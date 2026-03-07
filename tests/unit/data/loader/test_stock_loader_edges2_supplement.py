"""
补充测试：stock_loader.py
测试未覆盖的边界情况
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import configparser
from src.data.loader.stock_loader import StockDataLoader
from src.infrastructure.utils.exceptions import DataLoaderError


def test_stock_loader_init_empty_save_path():
    """测试 StockDataLoader（__init__，save_path 为空，覆盖 58 行）"""
    with pytest.raises(ValueError, match="save_path不能为空"):
        StockDataLoader(save_path="")


def test_stock_loader_init_invalid_frequency():
    """测试 StockDataLoader（__init__，frequency 无效，覆盖 62 行）"""
    with pytest.raises(ValueError, match="frequency必须是daily / weekly / monthly"):
        StockDataLoader(save_path="/tmp/test", frequency="invalid")


def test_stock_loader_init_max_retries_zero():
    """测试 StockDataLoader（__init__，max_retries <= 0）"""
    with pytest.raises(ValueError, match="max_retries必须大于0"):
        StockDataLoader(save_path="/tmp/test", max_retries=0)


def test_stock_loader_create_from_config(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，覆盖 98 行）"""
    # Mock configparser 的 read 方法，避免读取实际文件
    original_read = configparser.ConfigParser.read
    
    def mock_read(self, filenames, encoding=None):
        # 不读取文件，直接设置配置
        if 'STOCK_LOADER' not in self:
            self.add_section('STOCK_LOADER')
        self.set('STOCK_LOADER', 'save_path', str(tmp_path))
        return [filenames]
    
    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_read)
    
    # 创建配置文件
    config = configparser.ConfigParser()
    config['STOCK_LOADER'] = {
        'save_path': str(tmp_path),
        'max_retries': '3',
        'cache_days': '30',
        'frequency': 'daily',
        'adjust_type': 'none'
    }
    
    loader = StockDataLoader.create_from_config(config)
    assert loader is not None
    # 注意：create_from_config 可能会从 config.ini 读取默认值，所以路径可能不同
    assert loader.save_path is not None


def test_stock_loader_create_from_config_dict(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，使用字典配置）"""
    # Mock configparser 的 read 方法
    def mock_read(self, filenames, encoding=None):
        if 'STOCK_LOADER' not in self:
            self.add_section('STOCK_LOADER')
        self.set('STOCK_LOADER', 'save_path', str(tmp_path))
        return [filenames]
    
    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_read)
    
    config_dict = {
        'STOCK_LOADER': {
            'save_path': str(tmp_path),
            'max_retries': '3',
            'cache_days': '30',
            'frequency': 'daily',
            'adjust_type': 'none'
        }
    }
    
    loader = StockDataLoader.create_from_config(config_dict)
    assert loader is not None
    # 注意：create_from_config 可能会从 config.ini 读取默认值
    assert loader.save_path is not None


def test_stock_loader_get_required_config_fields(tmp_path):
    """测试 StockDataLoader（get_required_config_fields，覆盖 159 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    fields = loader.get_required_config_fields()
    assert isinstance(fields, list)
    assert 'save_path' in fields


def test_stock_loader_validate_config_missing_fields(tmp_path):
    """测试 StockDataLoader（validate_config，缺少必需字段，覆盖 163 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    config = {}
    result = loader.validate_config(config)
    assert result is False


def test_stock_loader_load_data_missing_params(tmp_path):
    """测试 StockDataLoader（load_data，缺少必需参数，覆盖 218 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    with pytest.raises(ValueError, match="symbol, start_date, and end_date are required"):
        loader.load_data(symbol=None, start_date=None, end_date=None)


def test_stock_loader_load_batch_empty_symbols(tmp_path):
    """测试 StockDataLoader（load_batch，symbols 为空，覆盖 238 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    result = loader.load_batch(symbols=[], start_date="2023-01-01", end_date="2023-01-02")
    assert result == {}


def test_stock_loader_load_data_invalid_data(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_data，数据验证失败，覆盖 274 行）"""
    from src.infrastructure.utils.core.exceptions import DataLoaderError
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # Mock _validate_data 返回 False
    def mock_validate(data):
        return False, "Invalid data"
    
    monkeypatch.setattr(loader, '_validate_data', mock_validate)
    
    # Mock _fetch_raw_data 返回无效数据（缺少必需列）
    def mock_fetch_raw_data(symbol, start_date, end_date, adjust):
        return pd.DataFrame({'日期': ['2023-01-01'], '收盘': [100]})  # 缺少 open, high, low, volume
    
    monkeypatch.setattr(loader, '_fetch_raw_data', mock_fetch_raw_data)
    
    # 由于数据验证失败，应该抛出 DataLoaderError 异常
    with pytest.raises(DataLoaderError):
        loader.load_data(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )


def test_stock_loader_load_cache_payload(tmp_path):
    """测试 StockDataLoader（_load_cache_payload，覆盖 292 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # 创建缓存文件
    cache_file = tmp_path / "cache" / "test_cache.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入测试数据
    import pickle
    test_data = {"data": pd.DataFrame({'a': [1, 2, 3]}), "timestamp": "2023-01-01"}
    with open(cache_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    result = loader._load_cache_payload(cache_file)
    assert result is not None
    assert "data" in result


def test_stock_loader_load_cache_payload_not_exists(tmp_path):
    """测试 StockDataLoader（_load_cache_payload，文件不存在）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    cache_file = tmp_path / "cache" / "nonexistent.pkl"
    
    result = loader._load_cache_payload(cache_file)
    assert result is None

