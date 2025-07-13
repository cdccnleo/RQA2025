"""数据元数据测试模块"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.data_manager import DataModel, DataManager
from src.infrastructure.utils import DataLoaderError


@pytest.fixture
def sample_data():
    """样本数据fixture"""
    dates = pd.date_range('2023-01-01', periods=10)
    return pd.DataFrame({
        'close': np.random.randn(10) + 100,
        'volume': np.random.randint(1000, 10000, 10),
        'open': np.random.randn(10) + 100,
        'high': np.random.randn(10) + 102,
        'low': np.random.randn(10) + 98
    }, index=dates)


@pytest.fixture
def sample_metadata():
    """样本元数据fixture"""
    return {
        'source': 'test_source',
        'symbol': '000001.SZ',
        'frequency': '1d',
        'adjust_type': 'none',
        'data_quality': 'good'
    }


@pytest.fixture
def data_model(sample_data, sample_metadata):
    """数据模型fixture"""
    return DataModel(sample_data, '1d', sample_metadata)


class TestDataModel:
    """测试数据模型"""
    
    def test_data_model_init(self, sample_data, sample_metadata):
        """测试数据模型初始化"""
        model = DataModel(sample_data, '1d', sample_metadata)
        
        assert model.data is not None
        assert model.get_frequency() == '1d'
        assert model.get_metadata()['source'] == 'test_source'
        assert model.get_metadata()['symbol'] == '000001.SZ'
    
    def test_data_model_validation(self, sample_data, sample_metadata):
        """测试数据模型验证"""
        model = DataModel(sample_data, '1d', sample_metadata)
        assert model.validate() is True
        
        # 测试空数据
        empty_model = DataModel(pd.DataFrame(), '1d', sample_metadata)
        assert empty_model.validate() is False
    
    def test_data_model_metadata(self, data_model):
        """测试元数据操作"""
        metadata = data_model.get_metadata()
        
        assert 'source' in metadata
        assert 'symbol' in metadata
        assert 'created_at' in metadata
        assert 'data_shape' in metadata
        assert 'data_columns' in metadata
    
    def test_data_model_frequency(self, data_model):
        """测试频率获取"""
        assert data_model.get_frequency() == '1d'


class TestDataManager:
    """测试数据管理器"""
    
    @patch('src.data.data_manager.DataRegistry')
    @patch('src.data.data_manager.DataValidator')
    @patch('src.data.data_manager.DataQualityMonitor')
    @patch('src.data.data_manager.CacheManager')
    def test_data_manager_init(self, mock_cache, mock_quality, mock_validator, mock_registry):
        """测试数据管理器初始化"""
        manager = DataManager()
        
        assert manager.registry is not None
        assert manager.validator is not None
        assert manager.quality_monitor is not None
        assert manager.cache_manager is not None
    
    def test_data_manager_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = {
            "General": {
                'max_concurrent_workers': '4',
                'cache_dir': 'cache'
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3'
            }
        }
        
        manager = DataManager(config_dict=valid_config)
        assert manager.config is not None
    
    @patch('src.data.data_manager.ThreadPoolExecutor')
    def test_data_manager_thread_pool(self, mock_executor):
        """测试线程池初始化"""
        manager = DataManager()
        assert manager.thread_pool is not None
    
    def test_data_manager_register_loader(self):
        """测试加载器注册"""
        manager = DataManager()
        
        # 创建模拟加载器
        mock_loader = Mock()
        mock_loader.name = "test_loader"
        
        manager.register_loader("test_loader", mock_loader)
        
        # 验证注册
        registered_loaders = manager.registry.list_registered_loaders()
        assert "test_loader" in registered_loaders
    
    def test_data_manager_cache_key_generation(self):
        """测试缓存键生成"""
        manager = DataManager()
        
        cache_key = manager._generate_cache_key(
            data_type="stock",
            start_date="2023-01-01",
            end_date="2023-01-31",
            frequency="1d",
            symbol="000001.SZ"
        )
        
        assert "stock" in cache_key
        assert "2023-01-01" in cache_key
        assert "2023-01-31" in cache_key
        assert "000001.SZ" in cache_key
    
    def test_data_manager_data_lineage(self):
        """测试数据血缘记录"""
        manager = DataManager()
        
        # 创建模拟数据模型
        mock_data_model = Mock()
        mock_data_model.get_metadata.return_value = {'source': 'test'}
        
        manager._record_data_lineage(
            data_type="stock",
            data_model=mock_data_model,
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert "stock" in manager.data_lineage
    
    def test_data_manager_cache_operations(self):
        """测试缓存操作"""
        manager = DataManager()
        
        # 测试缓存统计
        stats = manager.get_cache_stats()
        assert isinstance(stats, dict)
        
        # 测试清理过期缓存
        cleaned_count = manager.clean_expired_cache()
        assert isinstance(cleaned_count, int)
    
    def test_data_manager_shutdown(self):
        """测试关闭操作"""
        manager = DataManager()
        
        # 测试正常关闭
        manager.shutdown()
        
        # 验证线程池已关闭
        if hasattr(manager, 'thread_pool') and manager.thread_pool:
            assert manager.thread_pool._shutdown