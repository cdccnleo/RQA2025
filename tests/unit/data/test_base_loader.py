"""
基础数据加载器单元测试
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.base_loader import BaseDataLoader, LoaderError, ConfigError, DataLoadError
from src.data.interfaces import IDataModel


class MockDataLoader(BaseDataLoader):
    """模拟数据加载器实现"""
    
    def load(self, start_date: str, end_date: str, frequency: str) -> IDataModel:
        """模拟数据加载"""
        # 创建模拟数据
        dates = pd.date_range(start_date, end_date, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': [100 + i for i in range(len(dates))],
            'volume': [1000 + i * 10 for i in range(len(dates))]
        })
        
        # 创建数据模型
        from src.data.data_manager import DataModel
        return DataModel(data, frequency)
    
    def get_required_config_fields(self) -> list:
        """获取必需的配置字段"""
        return ['save_path', 'max_retries']


class TestBaseDataLoader:
    """基础数据加载器测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def valid_config(self, temp_dir):
        """有效配置fixture"""
        return {
            'save_path': f'{temp_dir}/data',
            'max_retries': 3,
            'cache_dir': temp_dir
        }
    
    @pytest.fixture
    def invalid_config(self):
        """无效配置fixture"""
        return {
            'save_path': '/data'  # 缺少max_retries
        }
    
    def test_base_loader_initialization(self, valid_config):
        """测试基础加载器初始化"""
        loader = MockDataLoader(valid_config)
        
        assert loader.config == valid_config
        assert loader.max_retries == 3
        assert loader.cache_dir.exists()
        assert 'loader_type' in loader.metadata
        assert 'config_timestamp' in loader.metadata
    
    def test_config_validation_success(self, valid_config):
        """测试配置验证成功"""
        loader = MockDataLoader(valid_config)
        assert loader._validate_config() is True
    
    def test_config_validation_failure(self, invalid_config):
        """测试配置验证失败"""
        with pytest.raises(ConfigError, match="Missing required config field: max_retries"):
            MockDataLoader(invalid_config)
    
    def test_cache_path_generation(self, valid_config, temp_dir):
        """测试缓存路径生成"""
        loader = MockDataLoader(valid_config)
        
        cache_path = loader._get_cache_path("test_key")
        assert cache_path.parent == Path(temp_dir)
        assert "MockDataLoader_test_key.parquet" in str(cache_path)
    
    def test_cache_operations(self, valid_config, temp_dir):
        """测试缓存操作"""
        loader = MockDataLoader(valid_config)
        
        # 测试数据
        test_data = pd.DataFrame({'close': [100, 101, 102]})
        cache_key = "test_cache"
        
        # 测试保存到缓存
        success = loader._save_to_cache(cache_key, test_data)
        assert success is True
        
        # 验证文件存在
        cache_path = loader._get_cache_path(cache_key)
        assert cache_path.exists()
        
        # 测试从缓存加载
        loaded_data = loader._load_from_cache(cache_key)
        assert loaded_data is not None
        pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_cache_key_generation(self, valid_config):
        """测试缓存键生成"""
        loader = MockDataLoader(valid_config)
        
        key = loader._generate_cache_key(
            start_date='2023-01-01',
            end_date='2023-01-31',
            symbols=['000001']
        )
        
        # 验证键包含所有参数
        assert '2023-01-01' in key
        assert '2023-01-31' in key
        assert '000001' in key
        
        # 验证相同参数生成相同键
        key2 = loader._generate_cache_key(
            start_date='2023-01-01',
            end_date='2023-01-31',
            symbols=['000001']
        )
        assert key == key2
    
    def test_metadata_update(self, valid_config):
        """测试元数据更新"""
        loader = MockDataLoader(valid_config)
        
        initial_timestamp = loader.metadata['config_timestamp']
        
        # 更新元数据
        loader.update_metadata(source='test', version='1.0')
        
        assert loader.metadata['source'] == 'test'
        assert loader.metadata['version'] == '1.0'
        assert loader.metadata['last_updated'] != initial_timestamp
    
    def test_cache_error_handling(self, valid_config):
        """测试缓存错误处理"""
        loader = MockDataLoader(valid_config)
        
        # 测试无效数据保存
        invalid_data = "not a dataframe"
        success = loader._save_to_cache("test", invalid_data)
        assert success is False
        
        # 测试不存在的缓存文件加载
        loaded_data = loader._load_from_cache("nonexistent")
        assert loaded_data is None


class TestDataLoaderIntegration:
    """数据加载器集成测试"""
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """集成测试配置"""
        return {
            'save_path': f'{temp_dir}/data',
            'max_retries': 3,
            'cache_dir': temp_dir,
            'frequency': 'daily',
            'adjust_type': 'none'
        }
    
    def test_data_loading_integration(self, integration_config):
        """测试数据加载集成"""
        loader = MockDataLoader(integration_config)
        
        # 加载数据
        model = loader.load('2023-01-01', '2023-01-10', '1d')
        
        # 验证返回的数据模型
        assert isinstance(model, IDataModel)
        assert model.validate() is True
        assert model.get_frequency() == '1d'
        
        # 验证数据内容
        data = model.data
        assert len(data) == 10  # 10天的数据
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert 'date' in data.columns
    
    def test_cache_integration(self, integration_config):
        """测试缓存集成"""
        loader = MockDataLoader(integration_config)
        
        # 第一次加载（无缓存）
        model1 = loader.load('2023-01-01', '2023-01-05', '1d')
        
        # 第二次加载（应该使用缓存）
        model2 = loader.load('2023-01-01', '2023-01-05', '1d')
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(model1.data, model2.data)
    
    def test_error_handling_integration(self, integration_config):
        """测试错误处理集成"""
        # 创建会抛出异常的加载器
        class ErrorLoader(MockDataLoader):
            def load(self, start_date: str, end_date: str, frequency: str) -> IDataModel:
                raise DataLoadError("模拟加载错误")
        
        loader = ErrorLoader(integration_config)
        
        # 验证异常被正确抛出
        with pytest.raises(DataLoadError, match="模拟加载错误"):
            loader.load('2023-01-01', '2023-01-05', '1d')


class TestDataLoaderPerformance:
    """数据加载器性能测试"""
    
    @pytest.fixture
    def performance_config(self, temp_dir):
        """性能测试配置"""
        return {
            'save_path': f'{temp_dir}/data',
            'max_retries': 3,
            'cache_dir': temp_dir,
            'frequency': 'daily',
            'adjust_type': 'none'
        }
    
    def test_large_data_loading_performance(self, performance_config):
        """测试大数据加载性能"""
        import time
        
        loader = MockDataLoader(performance_config)
        
        # 测试大日期范围
        start_time = time.time()
        model = loader.load('2020-01-01', '2023-12-31', '1d')
        load_time = time.time() - start_time
        
        # 验证加载时间合理（小于5秒）
        assert load_time < 5.0
        
        # 验证数据量
        data = model.data
        assert len(data) > 1000  # 应该有超过1000天的数据
    
    def test_cache_performance(self, performance_config):
        """测试缓存性能"""
        import time
        
        loader = MockDataLoader(performance_config)
        
        # 创建大量数据
        large_data = pd.DataFrame({
            'close': range(10000),
            'volume': range(10000),
            'open': range(10000),
            'high': range(10000),
            'low': range(10000)
        })
        
        # 测试缓存写入性能
        start_time = time.time()
        success = loader._save_to_cache("large_test", large_data)
        write_time = time.time() - start_time
        
        assert success is True
        assert write_time < 2.0  # 写入时间应该小于2秒
        
        # 测试缓存读取性能
        start_time = time.time()
        loaded_data = loader._load_from_cache("large_test")
        read_time = time.time() - start_time
        
        assert loaded_data is not None
        assert read_time < 1.0  # 读取时间应该小于1秒
        pd.testing.assert_frame_equal(loaded_data, large_data)


class TestDataLoaderErrorHandling:
    """数据加载器错误处理测试"""
    
    @pytest.fixture
    def error_config(self, temp_dir):
        """错误处理测试配置"""
        return {
            'save_path': f'{temp_dir}/data',
            'max_retries': 1,
            'cache_dir': temp_dir
        }
    
    def test_invalid_date_format(self, error_config):
        """测试无效日期格式"""
        loader = MockDataLoader(error_config)
        
        # 测试无效日期格式
        with pytest.raises(Exception):
            loader.load('invalid-date', '2023-01-05', '1d')
    
    def test_invalid_frequency(self, error_config):
        """测试无效频率"""
        loader = MockDataLoader(error_config)
        
        # 测试无效频率
        with pytest.raises(Exception):
            loader.load('2023-01-01', '2023-01-05', 'invalid')
    
    def test_cache_directory_permission_error(self, error_config):
        """测试缓存目录权限错误"""
        # 创建只读目录
        cache_dir = Path(error_config['cache_dir'])
        cache_dir.mkdir(exist_ok=True)
        
        try:
            cache_dir.chmod(0o444)  # 只读
            
            loader = MockDataLoader(error_config)
            test_data = pd.DataFrame({'close': [100, 101, 102]})
            
            # 应该优雅地处理权限错误
            success = loader._save_to_cache("test", test_data)
            assert success is False
            
        except PermissionError:
            # Windows上可能不支持chmod
            pass
        finally:
            # 恢复权限
            try:
                cache_dir.chmod(0o755)
            except:
                pass
    
    def test_missing_required_methods(self):
        """测试缺少必需方法"""
        class IncompleteLoader(BaseDataLoader):
            def get_required_config_fields(self) -> list:
                return ['save_path']
            # 缺少load方法
        
        config = {'save_path': '/data', 'max_retries': 3}
        
        # 应该能够创建实例，但调用load时会失败
        loader = IncompleteLoader(config)
        
        with pytest.raises(TypeError):
            loader.load('2023-01-01', '2023-01-05', '1d')


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 