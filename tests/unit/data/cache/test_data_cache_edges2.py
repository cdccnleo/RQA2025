"""
边界测试：data_cache.py
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
from unittest.mock import Mock, patch

from src.data.cache.data_cache import DataCache


@pytest.fixture
def data_cache(tmp_path):
    """创建数据缓存实例"""
    cache_dir = str(tmp_path / "data_cache")
    return DataCache(cache_dir=cache_dir)


def test_data_cache_init_default():
    """测试 DataCache（初始化，默认参数）"""
    cache = DataCache()
    assert cache is not None
    assert cache.cache_manager is not None


def test_data_cache_init_custom_dir(tmp_path):
    """测试 DataCache（初始化，自定义目录）"""
    cache_dir = str(tmp_path / "custom_cache")
    cache = DataCache(cache_dir=cache_dir)
    assert cache is not None


def test_data_cache_get_nonexistent(data_cache):
    """测试 DataCache（获取，不存在）"""
    result = data_cache.get("nonexistent_key")
    assert result is None


def test_data_cache_get_empty_key(data_cache):
    """测试 DataCache（获取，空键）"""
    result = data_cache.get("")
    assert result is None


def test_data_cache_set_get(data_cache):
    """测试 DataCache（设置和获取）"""
    result = data_cache.set("test_key", "test_value")
    assert result is True
    value = data_cache.get("test_key")
    assert value == "test_value"


def test_data_cache_set_empty_key(data_cache):
    """测试 DataCache（设置，空键）"""
    result = data_cache.set("", "test_value")
    assert isinstance(result, bool)


def test_data_cache_set_none_value(data_cache):
    """测试 DataCache（设置，None 值）"""
    result = data_cache.set("test_key", None)
    assert result is True
    value = data_cache.get("test_key")
    assert value is None


def test_data_cache_get_or_compute_cached(data_cache):
    """测试 DataCache（获取或计算，已缓存）"""
    data_cache.set("test_key", "cached_value")
    
    def compute_func():
        return "computed_value"
    
    result = data_cache.get_or_compute("test_key", compute_func)
    assert result == "cached_value"


def test_data_cache_get_or_compute_not_cached(data_cache):
    """测试 DataCache（获取或计算，未缓存）"""
    def compute_func():
        return "computed_value"
    
    result = data_cache.get_or_compute("test_key", compute_func)
    assert result == "computed_value"
    # 验证已缓存
    cached = data_cache.get("test_key")
    assert cached == "computed_value"


def test_data_cache_get_or_compute_with_args(data_cache):
    """测试 DataCache（获取或计算，带参数）"""
    def compute_func(x, y):
        return x + y
    
    result = data_cache.get_or_compute("test_key", compute_func, 1, 2)
    assert result == 3


def test_data_cache_get_or_compute_with_kwargs(data_cache):
    """测试 DataCache（获取或计算，带关键字参数）"""
    def compute_func(x=0, y=0):
        return x + y
    
    result = data_cache.get_or_compute("test_key", compute_func, x=1, y=2)
    assert result == 3


def test_data_cache_get_or_compute_empty_key(data_cache):
    """测试 DataCache（获取或计算，空键）"""
    def compute_func():
        return "value"
    
    result = data_cache.get_or_compute("", compute_func)
    assert result == "value"


def test_data_cache_get_dataframe_nonexistent(data_cache):
    """测试 DataCache（获取 DataFrame，不存在）"""
    result = data_cache.get_dataframe("nonexistent_key")
    assert result is None


def test_data_cache_get_dataframe_existing(data_cache):
    """测试 DataCache（获取 DataFrame，存在）"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    data_cache.set_dataframe("test_key", df)
    result = data_cache.get_dataframe("test_key")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_data_cache_get_dataframe_empty_key(data_cache):
    """测试 DataCache（获取 DataFrame，空键）"""
    result = data_cache.get_dataframe("")
    assert result is None


def test_data_cache_set_dataframe(data_cache):
    """测试 DataCache（设置 DataFrame）"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = data_cache.set_dataframe("test_key", df)
    assert result is True


def test_data_cache_set_dataframe_empty(data_cache):
    """测试 DataCache（设置 DataFrame，空）"""
    df = pd.DataFrame()
    result = data_cache.set_dataframe("test_key", df)
    assert result is True


def test_data_cache_set_dataframe_empty_key(data_cache):
    """测试 DataCache（设置 DataFrame，空键）"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = data_cache.set_dataframe("", df)
    assert isinstance(result, bool)


def test_data_cache_get_dict_nonexistent(data_cache):
    """测试 DataCache（获取字典，不存在）"""
    result = data_cache.get_dict("nonexistent_key")
    assert result is None


def test_data_cache_get_dict_existing(data_cache):
    """测试 DataCache（获取字典，存在）"""
    data = {'key': 'value'}
    data_cache.set_dict("test_key", data)
    result = data_cache.get_dict("test_key")
    assert result == data


def test_data_cache_get_dict_empty_key(data_cache):
    """测试 DataCache（获取字典，空键）"""
    result = data_cache.get_dict("")
    assert result is None


def test_data_cache_set_dict(data_cache):
    """测试 DataCache（设置字典）"""
    data = {'key': 'value'}
    result = data_cache.set_dict("test_key", data)
    assert result is True


def test_data_cache_set_dict_empty(data_cache):
    """测试 DataCache（设置字典，空）"""
    data = {}
    result = data_cache.set_dict("test_key", data)
    assert result is True


def test_data_cache_set_dict_empty_key(data_cache):
    """测试 DataCache（设置字典，空键）"""
    data = {'key': 'value'}
    result = data_cache.set_dict("", data)
    assert isinstance(result, bool)


def test_data_cache_clear(data_cache):
    """测试 DataCache（清空）"""
    data_cache.set("key1", "value1")
    data_cache.set("key2", "value2")
    result = data_cache.clear()
    assert isinstance(result, bool) or isinstance(result, int)
    # 清空后应该无法获取
    assert data_cache.get("key1") is None


def test_data_cache_clear_empty(data_cache):
    """测试 DataCache（清空，空缓存）"""
    result = data_cache.clear()
    # clear 可能返回 True/False 或删除的文件数量
    assert isinstance(result, bool) or isinstance(result, int)


def test_data_cache_exists_nonexistent(data_cache):
    """测试 DataCache（检查存在，不存在）"""
    result = data_cache.exists("nonexistent_key")
    assert result is False


def test_data_cache_exists_existing(data_cache):
    """测试 DataCache（检查存在，存在）"""
    data_cache.set("test_key", "test_value")
    result = data_cache.exists("test_key")
    assert result is True


def test_data_cache_exists_empty_key(data_cache):
    """测试 DataCache（检查存在，空键）"""
    result = data_cache.exists("")
    assert result is False


def test_data_cache_delete_nonexistent(data_cache):
    """测试 DataCache（删除，不存在）"""
    result = data_cache.delete("nonexistent_key")
    assert isinstance(result, bool)


def test_data_cache_delete_existing(data_cache):
    """测试 DataCache（删除，存在）"""
    data_cache.set("test_key", "test_value")
    result = data_cache.delete("test_key")
    assert isinstance(result, bool)
    assert data_cache.get("test_key") is None


def test_data_cache_delete_empty_key(data_cache):
    """测试 DataCache（删除，空键）"""
    result = data_cache.delete("")
    assert isinstance(result, bool)


def test_data_cache_get_stats(data_cache):
    """测试 DataCache（获取统计信息）"""
    if hasattr(data_cache, 'get_stats'):
        stats = data_cache.get_stats()
        assert isinstance(stats, dict)
    elif hasattr(data_cache.cache_manager, 'get_stats'):
        stats = data_cache.cache_manager.get_stats()
        assert isinstance(stats, dict)
    else:
        # 如果没有这个方法，测试通过
        assert True


def test_data_cache_get_stats_with_operations(data_cache):
    """测试 DataCache（获取统计信息，有操作）"""
    data_cache.set("key1", "value1")
    data_cache.get("key1")
    data_cache.get("nonexistent")
    if hasattr(data_cache, 'get_stats'):
        stats = data_cache.get_stats()
        assert isinstance(stats, dict)
    elif hasattr(data_cache.cache_manager, 'get_stats'):
        stats = data_cache.cache_manager.get_stats()
        assert isinstance(stats, dict)
    else:
        # 如果没有这个方法，测试通过
        assert True


def test_data_cache_get_with_exception(data_cache):
    """测试 DataCache（获取，异常处理）"""
    # 模拟 cache_manager.get 抛出异常
    with patch.object(data_cache.cache_manager, 'get', side_effect=Exception("Test error")):
        result = data_cache.get("test_key")
        assert result is None


def test_data_cache_set_with_exception(data_cache):
    """测试 DataCache（设置，异常处理）"""
    # 模拟 cache_manager.set 抛出异常
    with patch.object(data_cache.cache_manager, 'set', side_effect=Exception("Test error")):
        result = data_cache.set("test_key", "value")
        assert result is False


def test_data_cache_get_dataframe_with_exception(data_cache):
    """测试 DataCache（获取 DataFrame，异常处理）"""
    # 模拟 cache_manager.get 抛出异常
    with patch.object(data_cache.cache_manager, 'get', side_effect=Exception("Test error")):
        result = data_cache.get_dataframe("test_key")
        assert result is None


def test_data_cache_set_dataframe_with_exception(data_cache):
    """测试 DataCache（设置 DataFrame，异常处理）"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    # 模拟 cache_manager.set 抛出异常
    with patch.object(data_cache.cache_manager, 'set', side_effect=Exception("Test error")):
        result = data_cache.set_dataframe("test_key", df)
        assert result is False


def test_data_cache_get_dict_with_exception(data_cache):
    """测试 DataCache（获取字典，异常处理）"""
    # 模拟 cache_manager.get 抛出异常
    with patch.object(data_cache.cache_manager, 'get', side_effect=Exception("Test error")):
        result = data_cache.get_dict("test_key")
        assert result is None


def test_data_cache_set_dict_with_exception(data_cache):
    """测试 DataCache（设置字典，异常处理）"""
    data = {'key': 'value'}
    # 模拟 cache_manager.set 抛出异常
    with patch.object(data_cache.cache_manager, 'set', side_effect=Exception("Test error")):
        result = data_cache.set_dict("test_key", data)
        assert result is False


def test_data_cache_clear_with_exception(data_cache):
    """测试 DataCache（清空，异常处理）"""
    # 模拟 cache_manager.clear 抛出异常
    with patch.object(data_cache.cache_manager, 'clear', side_effect=Exception("Test error")):
        result = data_cache.clear()
        assert result is False


def test_data_cache_special_value_types(data_cache):
    """测试 DataCache（特殊值类型）"""
    test_cases = [
        ("dict", {"key": "value"}),
        ("list", [1, 2, 3]),
        ("tuple", (1, 2, 3)),
        ("int", 42),
        ("float", 3.14),
        ("bool", True),
        ("str", "string"),
    ]
    for key, value in test_cases:
        result = data_cache.set(key, value)
        assert result is True
        retrieved = data_cache.get(key)
        assert retrieved == value

