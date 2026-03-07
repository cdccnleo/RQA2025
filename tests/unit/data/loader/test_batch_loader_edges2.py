"""
边界测试：batch_loader.py
测试边界情况和异常场景
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
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.data.loader.batch_loader import BatchDataLoader
from src.data.loader.stock_loader import StockDataLoader


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def batch_loader(temp_dir):
    """创建批量加载器实例"""
    return BatchDataLoader(save_path=temp_dir)


@pytest.fixture
def mock_stock_loader():
    """创建模拟股票加载器"""
    loader = Mock(spec=StockDataLoader)
    loader.load = Mock(return_value=pd.DataFrame({'close': [100, 101, 102]}))
    return loader


def test_batch_loader_init_default():
    """测试 BatchDataLoader（初始化，默认参数）"""
    loader = BatchDataLoader()
    assert loader is not None
    assert loader.max_workers == 4
    assert loader.max_retries == 3


def test_batch_loader_init_custom(temp_dir):
    """测试 BatchDataLoader（初始化，自定义参数）"""
    loader = BatchDataLoader(
        save_path=temp_dir,
        max_retries=5,
        cache_days=2,
        timeout=60,
        max_workers=8
    )
    assert loader.save_path == Path(temp_dir)
    assert loader.max_retries == 5
    assert loader.cache_days == 2
    assert loader.timeout == 60
    assert loader.max_workers == 8


def test_batch_loader_init_with_stock_loader(temp_dir, mock_stock_loader):
    """测试 BatchDataLoader（初始化，自定义股票加载器）"""
    loader = BatchDataLoader(save_path=temp_dir, stock_loader=mock_stock_loader)
    assert loader.stock_loader == mock_stock_loader


def test_batch_loader_init_none_save_path():
    """测试 BatchDataLoader（初始化，None 保存路径）"""
    loader = BatchDataLoader(save_path=None)
    assert loader.save_path is not None
    assert loader.save_path.exists() or loader.save_path.parent.exists()


def test_batch_loader_load_batch_empty_symbols(batch_loader):
    """测试 BatchDataLoader（批量加载，空股票列表）"""
    result = batch_loader.load_batch([], "2023-01-01", "2023-01-31")
    assert result == {}


def test_batch_loader_load_batch_single_symbol(batch_loader):
    """测试 BatchDataLoader（批量加载，单个股票）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(["000001"], "2023-01-01", "2023-01-31")
        assert "000001" in result
        assert isinstance(result["000001"], pd.DataFrame)


def test_batch_loader_load_batch_multiple_symbols(batch_loader):
    """测试 BatchDataLoader（批量加载，多个股票）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(["000001", "000002", "000003"], "2023-01-01", "2023-01-31")
        assert len(result) == 3
        assert all(isinstance(df, pd.DataFrame) for df in result.values())


def test_batch_loader_load_batch_with_failure(batch_loader):
    """测试 BatchDataLoader（批量加载，部分失败）"""
    def mock_load(symbol, start_date, end_date, adjust):
        if symbol == "000002":
            raise Exception("Load failed")
        return pd.DataFrame({'close': [100]})
    
    with patch.object(batch_loader.stock_loader, 'load', side_effect=mock_load):
        result = batch_loader.load_batch(["000001", "000002", "000003"], "2023-01-01", "2023-01-31")
        assert len(result) == 3
        assert result["000001"] is not None
        assert result["000002"] is None
        assert result["000003"] is not None


def test_batch_loader_load_batch_all_failures(batch_loader):
    """测试 BatchDataLoader（批量加载，全部失败）"""
    with patch.object(batch_loader.stock_loader, 'load', side_effect=Exception("Load failed")):
        result = batch_loader.load_batch(["000001", "000002"], "2023-01-01", "2023-01-31")
        assert len(result) == 2
        assert all(value is None for value in result.values())


def test_batch_loader_load_batch_custom_max_workers(batch_loader):
    """测试 BatchDataLoader（批量加载，自定义工作线程数）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(
            ["000001", "000002", "000003", "000004"],
            "2023-01-01",
            "2023-01-31",
            max_workers=2
        )
        assert len(result) == 4


def test_batch_loader_load_batch_zero_workers(batch_loader):
    """测试 BatchDataLoader（批量加载，零工作线程）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(
            ["000001"],
            "2023-01-01",
            "2023-01-31",
            max_workers=0
        )
        # 应该使用默认值或最小值为1
        assert "000001" in result


def test_batch_loader_load_batch_datetime_dates(batch_loader):
    """测试 BatchDataLoader（批量加载，datetime 日期）"""
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(["000001"], start, end)
        assert "000001" in result


def test_batch_loader_load_batch_string_dates(batch_loader):
    """测试 BatchDataLoader（批量加载，字符串日期）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(["000001"], "2023-01-01", "2023-01-31")
        assert "000001" in result


def test_batch_loader_load_batch_custom_adjust(batch_loader):
    """测试 BatchDataLoader（批量加载，自定义复权方式）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(
            ["000001"],
            "2023-01-01",
            "2023-01-31",
            adjust="qfq"
        )
        assert "000001" in result


def test_batch_loader_load_method(batch_loader):
    """测试 BatchDataLoader（load 方法，兼容接口）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load(["000001"], "2023-01-01", "2023-01-31")
        assert "000001" in result


def test_batch_loader_validate_dict_dataframes(batch_loader):
    """测试 BatchDataLoader（验证，字典包含 DataFrame）"""
    data = {
        "000001": pd.DataFrame({'close': [100]}),
        "000002": pd.DataFrame({'close': [200]})
    }
    assert batch_loader.validate(data) is True


def test_batch_loader_validate_dict_with_none(batch_loader):
    """测试 BatchDataLoader（验证，字典包含 None）"""
    data = {
        "000001": pd.DataFrame({'close': [100]}),
        "000002": None
    }
    assert batch_loader.validate(data) is True


def test_batch_loader_validate_dict_all_none(batch_loader):
    """测试 BatchDataLoader（验证，字典全部为 None）"""
    data = {
        "000001": None,
        "000002": None
    }
    assert batch_loader.validate(data) is True


def test_batch_loader_validate_empty_dict(batch_loader):
    """测试 BatchDataLoader（验证，空字典）"""
    data = {}
    assert batch_loader.validate(data) is True


def test_batch_loader_validate_not_dict(batch_loader):
    """测试 BatchDataLoader（验证，非字典）"""
    data = pd.DataFrame({'close': [100]})
    assert batch_loader.validate(data) is False


def test_batch_loader_validate_dict_with_invalid_value(batch_loader):
    """测试 BatchDataLoader（验证，字典包含无效值）"""
    data = {
        "000001": pd.DataFrame({'close': [100]}),
        "000002": "invalid"
    }
    assert batch_loader.validate(data) is False


def test_batch_loader_get_metadata(batch_loader):
    """测试 BatchDataLoader（获取元数据）"""
    metadata = batch_loader.get_metadata()
    assert isinstance(metadata, dict)
    assert metadata["loader_type"] == "BatchDataLoader"
    assert metadata["supports_batch"] is True
    assert "max_workers" in metadata
    assert "timeout" in metadata


def test_batch_loader_load_batch_large_symbol_list(batch_loader):
    """测试 BatchDataLoader（批量加载，大量股票）"""
    symbols = [f"{i:06d}" for i in range(100)]
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(symbols, "2023-01-01", "2023-01-31")
        assert len(result) == 100


def test_batch_loader_load_batch_duplicate_symbols(batch_loader):
    """测试 BatchDataLoader（批量加载，重复股票代码）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame({'close': [100]})):
        result = batch_loader.load_batch(["000001", "000001", "000001"], "2023-01-01", "2023-01-31")
        # 重复的股票代码在结果字典中会被覆盖，但会多次调用 load 方法
        assert len(result) == 1  # 字典键唯一，后面的会覆盖前面的
        assert "000001" in result
        assert isinstance(result["000001"], pd.DataFrame)


def test_batch_loader_load_batch_empty_dataframe(batch_loader):
    """测试 BatchDataLoader（批量加载，返回空 DataFrame）"""
    with patch.object(batch_loader.stock_loader, 'load', return_value=pd.DataFrame()):
        result = batch_loader.load_batch(["000001"], "2023-01-01", "2023-01-31")
        assert "000001" in result
        assert isinstance(result["000001"], pd.DataFrame)
        assert len(result["000001"]) == 0


def test_batch_loader_init_zero_max_workers(temp_dir):
    """测试 BatchDataLoader（初始化，零工作线程）"""
    loader = BatchDataLoader(save_path=temp_dir, max_workers=0)
    # 应该使用默认值或最小值
    assert loader.max_workers >= 1


def test_batch_loader_init_negative_max_workers(temp_dir):
    """测试 BatchDataLoader（初始化，负工作线程）"""
    loader = BatchDataLoader(save_path=temp_dir, max_workers=-1)
    # 负值会被保留，但在使用时会被处理（max_workers or 4）
    assert loader.max_workers == -1


def test_batch_loader_init_negative_max_retries(temp_dir):
    """测试 BatchDataLoader（初始化，负重试次数）"""
    # StockDataLoader 不允许负值，会抛出异常
    with pytest.raises(ValueError, match="max_retries必须大于0"):
        BatchDataLoader(save_path=temp_dir, max_retries=-1)


def test_batch_loader_init_zero_timeout(temp_dir):
    """测试 BatchDataLoader（初始化，零超时）"""
    loader = BatchDataLoader(save_path=temp_dir, timeout=0)
    assert loader.timeout == 0


def test_batch_loader_init_negative_cache_days(temp_dir):
    """测试 BatchDataLoader（初始化，负缓存天数）"""
    loader = BatchDataLoader(save_path=temp_dir, cache_days=-1)
    assert loader.cache_days == -1  # 允许负值，由调用方处理

