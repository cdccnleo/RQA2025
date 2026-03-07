"""
补充测试：stock_loader.py（第三批）
测试未覆盖的边界情况和异常路径
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
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import configparser
import concurrent.futures
from src.data.loader.stock_loader import StockDataLoader
from src.infrastructure.utils.core.exceptions import DataLoaderError


def test_stock_loader_create_from_config_section_object(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，使用 Section 对象，覆盖 107-110 行）"""
    # Mock configparser 的 read 方法
    def mock_read(self, filenames, encoding=None):
        if 'Stock' not in self:
            self.add_section('Stock')
        self.set('Stock', 'save_path', str(tmp_path))
        return [filenames]
    
    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_read)
    
    # 创建一个类似 Section 的对象（有 items 方法）
    class SectionLike:
        def items(self):
            return [('save_path', str(tmp_path)), ('max_retries', '3'), ('cache_days', '30')]
    
    config = configparser.ConfigParser()
    config['Stock'] = SectionLike()
    
    loader = StockDataLoader.create_from_config(config)
    assert loader is not None
    assert loader.save_path is not None


def test_stock_loader_create_from_config_invalid_type(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，不支持的配置类型，覆盖 110 行）"""
    # Mock configparser 的 read 方法
    def mock_read(self, filenames, encoding=None):
        return [filenames]
    
    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_read)
    
    # 使用不支持的配置类型（既不是 ConfigParser，也不是 dict，也没有 items 方法）
    invalid_config = "invalid_config"
    
    with pytest.raises(ValueError, match="不支持的配置类型"):
        StockDataLoader.create_from_config(invalid_config)


def test_stock_loader_create_from_config_invalid_int_value(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，无效的整数值，覆盖 124-129 行）"""
    # Mock configparser 的 read 方法，确保默认配置有有效的整数值
    def mock_read(self, filenames, encoding=None):
        if 'Stock' not in self:
            self.add_section('Stock')
        self.set('Stock', 'save_path', str(tmp_path))
        self.set('Stock', 'max_retries', '3')  # 默认值有效
        return [filenames]
    
    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_read)
    
    # 创建一个配置对象，其中 max_retries 是无效的整数值
    # 使用字典配置，这样 safe_getint 会从 loader_config 中获取值
    config_dict = {
        'Stock': {
            'save_path': str(tmp_path),
            'max_retries': 'invalid_int',  # 无效的整数值
            'cache_days': '30',
            'frequency': 'daily',
            'adjust_type': 'none'
        }
    }
    
    with pytest.raises(DataLoaderError, match="配置项.*的值.*无效"):
        StockDataLoader.create_from_config(config_dict)


def test_stock_loader_load_batch_with_thread_pool(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_batch，使用外部线程池，覆盖 257-263 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # 创建一个真实的 ThreadPoolExecutor（不是 Mock）
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loader.thread_pool = executor
    
    # Mock load 方法返回测试数据
    def mock_load(symbol, start_date, end_date, adjust=None, force_refresh=False):
        return pd.DataFrame({
            'date': pd.date_range(start_date, end_date, freq='D'),
            'open': [100] * 2,
            'high': [110] * 2,
            'low': [90] * 2,
            'close': [105] * 2,
            'volume': [1000] * 2
        })
    
    monkeypatch.setattr(loader, 'load', mock_load)
    
    result = loader.load_batch(
        symbols=['000001', '000002'],
        start_date='2023-01-01',
        end_date='2023-01-02'
    )
    
    assert len(result) == 2
    assert '000001' in result
    assert '000002' in result
    
    executor.shutdown(wait=True)


def test_stock_loader_load_batch_with_exception(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_batch，任务执行异常，覆盖 251-253 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # Mock load 方法抛出异常
    def mock_load(symbol, start_date, end_date, adjust=None, force_refresh=False):
        raise Exception("Load failed")
    
    monkeypatch.setattr(loader, 'load', mock_load)
    
    result = loader.load_batch(
        symbols=['000001'],
        start_date='2023-01-01',
        end_date='2023-01-02'
    )
    
    assert '000001' in result
    assert result['000001'] is None


def test_stock_loader_load_single_stock_cache_hit(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_single_stock，缓存命中，覆盖 292-302 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # 创建缓存文件
    cache_file = loader._get_cache_file_path('000001', 'daily', 'none')
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入缓存数据
    import pickle
    import time
    from datetime import datetime
    cached_data = {
        "data": pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-02', freq='D'),
            'open': [100] * 2,
            'high': [110] * 2,
            'low': [90] * 2,
            'close': [105] * 2,
            'volume': [1000] * 2
        }),
        "metadata": {
            "symbol": "000001",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "frequency": "daily",
            "adjust_type": "none",
            "cached_time": datetime.now()
        },
        "cache_info": {"is_from_cache": True}
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    # Mock _load_cache_payload 返回缓存数据
    def mock_load_cache_payload(file_path):
        return cached_data
    
    monkeypatch.setattr(loader, '_load_cache_payload', mock_load_cache_payload)
    
    result = loader.load_single_stock(
        symbol='000001',
        start_date='2023-01-01',
        end_date='2023-01-02',
        force_refresh=False
    )
    
    assert result is not None
    assert result.get("cache_info", {}).get("is_from_cache") is True
    assert result.get("metadata", {}).get("performance", {}).get("cache_hit") is True


def test_stock_loader_load_single_stock_validation_failed(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_single_stock，数据验证失败，覆盖 307 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # Mock _load_data_impl 返回数据
    def mock_load_data_impl(symbol, start_date, end_date, adjust):
        return pd.DataFrame({'date': ['2023-01-01'], 'close': [100]})
    
    monkeypatch.setattr(loader, '_load_data_impl', mock_load_data_impl)
    
    # Mock _validate_data 返回验证失败
    def mock_validate_data(data):
        return False, ["Missing required columns"]
    
    monkeypatch.setattr(loader, '_validate_data', mock_validate_data)
    
    with pytest.raises(DataLoaderError, match="数据验证失败"):
        loader.load_single_stock(
            symbol='000001',
            start_date='2023-01-01',
            end_date='2023-01-02',
            force_refresh=True
        )


def test_stock_loader_load_multiple_stocks_empty(tmp_path):
    """测试 StockDataLoader（load_multiple_stocks，symbols 为空，覆盖 340-341 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    result = loader.load_multiple_stocks(symbols=[])
    assert result == {}


def test_stock_loader_load_multiple_stocks_with_thread_pool(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_multiple_stocks，使用外部线程池，覆盖 364-372 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # 创建一个真实的 ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loader.thread_pool = executor
    
    # Mock _load_single_stock_with_cache 返回测试数据
    def mock_load_single_stock_with_cache(symbol, start_date=None, end_date=None, adjust=None, force_refresh=False):
        return {
            "data": pd.DataFrame({
                'date': pd.date_range('2023-01-01', '2023-01-02', freq='D'),
                'open': [100] * 2,
                'high': [110] * 2,
                'low': [90] * 2,
                'close': [105] * 2,
                'volume': [1000] * 2
            }),
            "metadata": {"symbol": symbol}
        }
    
    monkeypatch.setattr(loader, '_load_single_stock_with_cache', mock_load_single_stock_with_cache)
    
    result = loader.load_multiple_stocks(symbols=['000001', '000002'], max_workers=2)
    
    assert len(result) == 2
    assert '000001' in result
    assert '000002' in result
    
    executor.shutdown(wait=True)


def test_stock_loader_load_multiple_stocks_with_exception(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_multiple_stocks，任务执行异常，覆盖 346-348 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # Mock _load_single_stock_with_cache 抛出异常
    def mock_load_single_stock_with_cache(symbol, start_date=None, end_date=None, adjust=None, force_refresh=False):
        raise Exception("Load failed")
    
    monkeypatch.setattr(loader, '_load_single_stock_with_cache', mock_load_single_stock_with_cache)
    
    result = loader.load_multiple_stocks(symbols=['000001'], max_workers=1)
    
    assert '000001' in result
    assert "error" in result['000001']


def test_stock_loader_load_multiple_stocks_thread_pool_unexpected_result(tmp_path, monkeypatch):
    """测试 StockDataLoader（load_multiple_stocks，线程池返回非预期结果，覆盖 371-372 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # 创建一个真实的 ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loader.thread_pool = executor
    
    # Mock submit 返回一个返回非元组结果的 future
    original_submit = executor.submit
    
    def mock_submit(fn, *args, **kwargs):
        future = original_submit(fn, *args, **kwargs)
        # 修改 future 的 result 方法，返回非元组结果
        original_result = future.result
        
        def mock_result():
            return "unexpected_result"  # 非元组结果
        
        future.result = mock_result
        return future
    
    monkeypatch.setattr(executor, 'submit', mock_submit)
    
    # Mock _load_single_stock_with_cache 返回测试数据
    def mock_load_single_stock_with_cache(symbol, start_date=None, end_date=None, adjust=None, force_refresh=False):
        return {"data": pd.DataFrame(), "metadata": {"symbol": symbol}}
    
    monkeypatch.setattr(loader, '_load_single_stock_with_cache', mock_load_single_stock_with_cache)
    
    result = loader.load_multiple_stocks(symbols=['000001'], max_workers=2)
    
    executor.shutdown(wait=True)


def test_stock_loader_get_metadata(tmp_path):
    """测试 StockDataLoader（get_metadata，覆盖 170-188 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    metadata = loader.get_metadata()
    
    assert isinstance(metadata, dict)
    assert metadata["loader_type"] == "StockDataLoader"
    assert metadata["data_frequency"] == "daily"
    assert "max_retries" in metadata
    assert "cache_days" in metadata
    assert "supported_features" in metadata


def test_stock_loader_load_wrapper(tmp_path, monkeypatch):
    """测试 StockDataLoader（load，包装 load_data，覆盖 190-207 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    # Mock load_data 返回测试数据
    def mock_load_data(**kwargs):
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-02', freq='D'),
            'open': [100] * 2,
            'high': [110] * 2,
            'low': [90] * 2,
            'close': [105] * 2,
            'volume': [1000] * 2
        })
    
    monkeypatch.setattr(loader, 'load_data', mock_load_data)
    
    result = loader.load(
        symbol='000001',
        start_date='2023-01-01',
        end_date='2023-01-02',
        adjust='none',
        extra_param='test'
    )
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)

