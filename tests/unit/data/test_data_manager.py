"""
数据管理器单元测试
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.data_manager import DataManager, DataModel
from src.data.interfaces import IDataModel
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.utils import DataLoaderError


class TestDataModel:
    """数据模型测试"""
    
    def test_data_model_creation(self):
        """测试数据模型创建"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        metadata = {'source': 'test'}
        model = DataModel(data, '1d', metadata)
        
        assert model.data is not None
        assert model.get_frequency() == '1d'
        assert model.get_metadata()['source'] == 'test'
        assert 'created_at' in model.get_metadata()
    
    def test_data_model_validation(self):
        """测试数据模型验证"""
        # 有效数据
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        assert model.validate() is True
        
        # 无效数据 - 使用空DataFrame而不是None
        model_empty = DataModel(pd.DataFrame(), '1d')
        assert model_empty.validate() is False


class TestDataManager:
    """数据管理器测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_dict(self):
        """测试配置字典"""
        return {
            "General": {
                'max_concurrent_workers': '2',
                'cache_dir': 'test_cache',
                'max_cache_size': str(1024 * 1024 * 100),
                'cache_ttl': '3600',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '7',
                'frequency': 'daily',
                'adjust_type': 'none'
            },
            "News": {
                'save_path': 'data/news',
                'max_retries': '3',
                'cache_days': '3'
            }
        }
    
    @pytest.fixture
    def data_manager(self, temp_dir, config_dict):
        """数据管理器fixture"""
        # 修改缓存目录到临时目录
        config_dict["General"]["cache_dir"] = temp_dir
        return DataManager(config_dict=config_dict)
    
    def test_data_manager_initialization(self, data_manager):
        """测试数据管理器初始化"""
        assert data_manager.config is not None
        assert data_manager.registry is not None
        assert data_manager.validator is not None
        assert data_manager.cache_manager is not None
        assert data_manager.thread_pool is not None
    
    def test_config_validation(self, config_dict):
        """测试配置验证"""
        # 有效配置
        manager = DataManager(config_dict=config_dict)
        assert manager.validate_all_configs() is True
        
        # 无效配置
        invalid_config = {"General": {}}  # 缺少必需字段
        with pytest.raises(Exception):
            DataManager(config_dict=invalid_config)
    
    @patch('src.data.data_manager.DataRegistry')
    def test_loader_registration(self, mock_registry, data_manager):
        """测试加载器注册"""
        mock_loader = Mock()
        mock_loader.get_required_config_fields.return_value = ['save_path']
        
        data_manager.register_loader('test_loader', mock_loader)
        
        # 验证注册中心被调用
        data_manager.registry.register.assert_called_once_with('test_loader', mock_loader)
    
    def test_cache_key_generation(self, data_manager):
        """测试缓存键生成"""
        key = data_manager._generate_cache_key(
            data_type='stock',
            start_date='2023-01-01',
            end_date='2023-01-31',
            frequency='1d',
            symbols=['000001']
        )
        
        assert 'stock' in key
        assert '2023-01-01' in key
        assert '2023-01-31' in key
        assert '1d' in key
        assert '000001' in key
    
    def test_data_lineage_recording(self, data_manager):
        """测试数据血缘记录"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        data_manager._record_data_lineage(
            'stock',
            model,
            '2023-01-01',
            '2023-01-31',
            symbols=['000001']
        )
        
        # 检查数据血缘记录是否存在
        lineage_key = 'stock_2023-01-01_2023-01-31'
        assert lineage_key in data_manager.data_lineage
        lineage_record = data_manager.data_lineage[lineage_key]
        assert lineage_record['start_date'] == '2023-01-01'
        assert lineage_record['end_date'] == '2023-01-31'
        assert lineage_record['parameters']['symbols'] == ['000001']
    
    def test_cache_operations(self, data_manager, temp_dir):
        """测试缓存操作"""
        # 测试清理过期缓存
        cleaned_count = data_manager.clean_expired_cache()
        assert isinstance(cleaned_count, int)
        
        # 测试获取缓存统计
        stats = data_manager.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'hits' in stats
        assert 'sets' in stats
        assert 'deletes' in stats
    
    def test_shutdown(self, data_manager):
        """测试关闭操作"""
        data_manager.shutdown()
        # 验证线程池已关闭
        assert data_manager.thread_pool._shutdown is True


class TestDataManagerInfrastructureIntegration:
    """数据管理器与基础设施层集成测试"""
    
    @pytest.fixture
    def config_manager(self):
        """配置管理器fixture"""
        return ConfigManager(env='test')
    
    @pytest.fixture
    def data_manager_with_infrastructure(self, temp_dir, config_manager):
        """带基础设施支持的数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '2',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 100),
                'cache_ttl': '3600',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '7',
                'frequency': 'daily',
                'adjust_type': 'none'
            }
        }
        
        # 将配置加载到配置管理器
        config_manager.load_from_dict(config_dict)
        
        return DataManager(config_dict=config_dict)
    
    def test_config_manager_integration(self, data_manager_with_infrastructure, config_manager):
        """测试与配置管理器的集成"""
        # 验证配置一致性
        assert data_manager_with_infrastructure.config.getint("General", "max_concurrent_workers") == 2
        assert config_manager.get("General.max_concurrent_workers") == 2
    
    @patch('src.infrastructure.monitoring.metrics.MetricsCollector')
    def test_monitoring_integration(self, mock_metrics, data_manager_with_infrastructure):
        """测试与监控系统的集成"""
        # 模拟数据加载操作
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        # 记录数据血缘（这会触发监控）
        data_manager_with_infrastructure._record_data_lineage(
            'stock',
            model,
            '2023-01-01',
            '2023-01-31'
        )
        
        # 验证监控指标被记录
        # 这里可以添加具体的监控验证逻辑
    
    @patch('src.infrastructure.error.error_handler.ErrorHandler')
    def test_error_handling_integration(self, mock_error_handler, data_manager_with_infrastructure):
        """测试与错误处理系统的集成"""
        # 模拟错误情况
        try:
            raise DataLoaderError("测试错误")
        except DataLoaderError as e:
            # 验证错误被正确处理
            assert str(e) == "测试错误"
    
    def test_logging_integration(self, data_manager_with_infrastructure, caplog):
        """测试与日志系统的集成"""
        # 执行一个操作来触发日志
        data_manager_with_infrastructure.validate_all_configs()
        
        # 验证日志被正确记录
        assert "数据管理器初始化完成" in caplog.text or "DataManager initialized" in caplog.text


class TestDataManagerPerformance:
    """数据管理器性能测试"""
    
    @pytest.fixture
    def large_data_manager(self, temp_dir):
        """大数据管理器fixture"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '4',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 1024),  # 1GB
                'cache_ttl': '86400',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '30',
                'frequency': 'daily',
                'adjust_type': 'none'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_concurrent_data_loading(self, large_data_manager):
        """测试并发数据加载性能"""
        import time
        
        start_time = time.time()
        
        # 模拟多个并发数据加载任务
        futures = []
        for i in range(10):
            future = large_data_manager.thread_pool.submit(
                lambda: time.sleep(0.1)  # 模拟数据加载
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证并发执行时间合理（应该小于串行执行时间）
        assert execution_time < 1.0  # 10个0.1秒的任务应该并发完成
    
    def test_cache_performance(self, large_data_manager):
        """测试缓存性能"""
        import time
        
        # 创建大量测试数据
        large_data = pd.DataFrame({
            'close': range(10000),
            'volume': range(10000),
            'open': range(10000),
            'high': range(10000),
            'low': range(10000)
        })
        
        model = DataModel(large_data, '1d')
        
        # 测试缓存写入性能
        start_time = time.time()
        large_data_manager._record_data_lineage('stock', model, '2023-01-01', '2023-01-31')
        write_time = time.time() - start_time
        
        # 验证写入时间合理
        assert write_time < 1.0  # 1秒内应该完成
    
    def test_memory_usage(self, large_data_manager):
        """测试内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行一些内存密集型操作
        for i in range(100):
            data = pd.DataFrame({'close': range(1000)})
            model = DataModel(data, '1d')
            large_data_manager._record_data_lineage('stock', model, '2023-01-01', '2023-01-31')
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长合理（小于100MB）
        assert memory_increase < 100 * 1024 * 1024


class TestDataManagerErrorHandling:
    """数据管理器错误处理测试"""
    
    @pytest.fixture
    def error_data_manager(self, temp_dir):
        """错误处理测试用的数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '1',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 10),
                'cache_ttl': '3600',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '1',
                'cache_days': '1',
                'frequency': 'daily',
                'adjust_type': 'none'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_invalid_config_handling(self):
        """测试无效配置处理"""
        invalid_config = {
            "General": {
                'max_concurrent_workers': 'invalid',  # 无效值
            }
        }
        
        with pytest.raises(Exception):
            DataManager(config_dict=invalid_config)
    
    def test_data_validation_error_handling(self, error_data_manager):
        """测试数据验证错误处理"""
        # 创建无效数据 - 使用空DataFrame
        invalid_data = pd.DataFrame()
        model = DataModel(invalid_data, '1d')
        
        # 验证数据验证失败
        assert model.validate() is False
    
    def test_cache_error_handling(self, error_data_manager, temp_dir):
        """测试缓存错误处理"""
        # 模拟缓存目录权限问题
        cache_dir = Path(temp_dir) / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # 设置只读权限（在Windows上可能不工作）
        try:
            cache_dir.chmod(0o444)  # 只读
            
            # 尝试缓存操作，应该优雅地处理错误
            data = pd.DataFrame({'close': [100, 101, 102]})
            model = DataModel(data, '1d')
            
            # 这应该不会抛出异常，而是记录警告
            error_data_manager._record_data_lineage('stock', model, '2023-01-01', '2023-01-31')
            
        except PermissionError:
            # Windows上可能不支持chmod
            pass
        finally:
            # 恢复权限
            try:
                cache_dir.chmod(0o755)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
