"""
提高stock_loader测试覆盖率的补充测试
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import os
import pickle

from src.data.loader.stock_loader import StockDataLoader
from src.data.interfaces.standard_interfaces import DataSourceType


class TestStockLoaderCoverageImprovement:
    """提高stock_loader测试覆盖率的测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def loader(self, temp_dir):
        """StockDataLoader fixture"""
        return StockDataLoader(save_path=temp_dir)

    def test_initialization(self, temp_dir):
        """测试初始化"""
        loader = StockDataLoader(save_path=temp_dir, max_retries=5, cache_days=60)
        assert loader.save_path == Path(temp_dir)
        assert loader.max_retries == 5
        assert loader.cache_days == 60

    def test_create_from_config(self, temp_dir):
        """测试从配置创建"""
        config = {
            'save_path': temp_dir,
            'max_retries': 3,
            'cache_days': 30,
            'frequency': 'daily',
            'adjust_type': 'qfq'
        }
        loader = StockDataLoader.create_from_config(config)
        assert isinstance(loader, StockDataLoader)

    def test_get_required_config_fields(self, loader):
        """测试获取必需配置字段"""
        fields = loader.get_required_config_fields()
        assert 'save_path' in fields
        assert isinstance(fields, list)

    def test_validate_config_valid(self, loader, temp_dir):
        """测试有效配置验证"""
        config = {'save_path': temp_dir, 'max_retries': 3, 'cache_days': 30}
        assert loader.validate_config(config) == True

    def test_validate_config_invalid(self, loader):
        """测试无效配置验证"""
        config = {}  # 缺少必需字段
        assert loader.validate_config(config) == False

    def test_get_metadata(self, loader):
        """测试获取元数据"""
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert 'loader_type' in metadata

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_load_single_stock_success(self, mock_ak, loader, temp_dir):
        """测试成功加载单只股票"""
        # Mock API response
        mock_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'open': [100.0, 101.0],
            'close': [101.0, 102.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'volume': [1000, 1100]
        })
        mock_ak.return_value = mock_df

        result = loader.load_single_stock('000001')
        assert isinstance(result, dict)
        assert 'data' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) > 0

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_load_single_stock_api_error(self, mock_ak, loader):
        """测试API错误情况"""
        mock_ak.side_effect = Exception('API Error')

        with pytest.raises(Exception):
            loader.load_single_stock('000001')

    def test_validate_data_valid(self, loader):
        """测试有效数据验证"""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'close': [101.0, 102.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'volume': [1000, 1100]
        })
        assert loader.validate_data(df) == True

    def test_validate_data_invalid_empty(self, loader):
        """测试空数据验证"""
        df = pd.DataFrame()
        assert loader.validate_data(df) == False

    def test_validate_data_invalid_missing_columns(self, loader):
        """测试缺失列数据验证"""
        df = pd.DataFrame({'invalid_col': [1, 2, 3]})
        assert loader.validate_data(df) == False

    def test_get_file_path(self, loader):
        """测试获取文件路径"""
        path = loader._get_file_path('000001', 'daily')
        assert isinstance(path, Path)
        assert '000001' in str(path)

    def test_is_cache_valid_valid(self, loader, temp_dir):
        """测试有效缓存验证"""
        # Create a recent cache file
        cache_file = Path(temp_dir) / 'cache' / '000001_daily.pkl'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('dummy')

        assert loader._is_cache_valid('000001', 'daily') == True

    def test_is_cache_valid_expired(self, loader, temp_dir):
        """测试过期缓存验证"""
        # Create an old cache file
        cache_file = Path(temp_dir) / 'cache' / '000001_daily.pkl'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('dummy')

        # Mock old timestamp
        old_time = datetime.now() - timedelta(days=40)  # Older than cache_days
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))

        assert loader._is_cache_valid('000001', 'daily') == False

    def test_get_cache_key(self, loader):
        """测试获取缓存键"""
        key = loader._get_cache_key('000001', 'daily', 'none')
        assert isinstance(key, str)
        assert '000001' in key

    def test_load_cache_payload_valid(self, loader, temp_dir):
        """测试加载有效缓存数据"""
        cache_file = Path(temp_dir) / 'cache' / 'test.pkl'
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        test_data = {'test': 'data'}
        with open(cache_file, 'wb') as f:
            pickle.dump(test_data, f)

        result = loader._load_cache_payload(cache_file)
        assert result == test_data

    def test_load_cache_payload_invalid(self, loader, temp_dir):
        """测试加载无效缓存数据"""
        cache_file = Path(temp_dir) / 'cache' / 'invalid.pkl'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('invalid pickle data')

        result = loader._load_cache_payload(cache_file)
        assert result is None
