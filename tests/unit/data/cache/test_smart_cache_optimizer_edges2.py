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
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.data.cache.smart_cache_optimizer import (
    get_smart_cache_optimizer, SmartCacheOptimizer, PreloadRule, CacheEntry
)
from src.data.interfaces.standard_interfaces import DataSourceType


@pytest.fixture
def optimizer():
    """创建智能缓存优化器实例"""
    opt = get_smart_cache_optimizer()
    yield opt
    opt.shutdown()


def test_access_pattern_invalidation_and_metrics(optimizer):
    """测试访问模式失效和性能指标"""
    key = "k-access"
    # 先写入
    assert optimizer.smart_set(key, {"v": 1}, DataSourceType.NEWS) is True
    # 访问以更新统计
    assert optimizer.smart_get(key, DataSourceType.NEWS) == {"v": 1}
    # 设置为很久未访问且访问次数低，触发访问模式失效
    entry = optimizer._cache_entries[key]
    entry.last_access = datetime.now() - timedelta(days=2)
    entry.access_count = 0
    assert optimizer.smart_get(key, DataSourceType.NEWS) is None
    # 性能指标存在
    metrics = optimizer.get_performance_metrics()
    assert "cache_status" in metrics and "preload_status" in metrics


def test_preload_rule_should_execute_no_last_execution(optimizer):
    """测试 PreloadRule（should_execute，无上次执行）"""
    def condition(context):
        return True
    
    def preload_func():
        return "data"
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    # 无上次执行，应该返回True
    assert rule.should_execute({}) is True


def test_preload_rule_should_execute_condition_false(optimizer):
    """测试 PreloadRule（should_execute，条件不满足）"""
    def condition(context):
        return False
    
    def preload_func():
        return "data"
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    # 条件不满足，应该返回False
    assert rule.should_execute({}) is False


def test_preload_rule_should_execute_interval_not_met(optimizer):
    """测试 PreloadRule（should_execute，间隔未满足）"""
    def condition(context):
        return True
    
    def preload_func():
        return "data"
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func,
        interval_seconds=300
    )
    
    # 设置上次执行时间为最近
    rule.last_execution = datetime.now()
    
    # 间隔未满足，应该返回False
    assert rule.should_execute({}) is False


def test_preload_rule_execute_success(optimizer):
    """测试 PreloadRule（execute，成功）"""
    def condition(context):
        return True
    
    def preload_func():
        return "data"
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    # 执行应该返回数据
    result = rule.execute({})
    assert result == "data"
    assert rule.last_execution is not None


def test_preload_rule_execute_exception(optimizer):
    """测试 PreloadRule（execute，异常处理）"""
    def condition(context):
        return True
    
    def preload_func():
        raise Exception("Preload error")
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    # 执行应该返回None（异常被捕获）
    result = rule.execute({})
    assert result is None


def test_smart_get_exception(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（smart_get，异常处理）"""
    # 模拟cache.get抛出异常
    def mock_get(key):
        raise Exception("Cache error")
    
    monkeypatch.setattr(optimizer.cache, "get", mock_get)
    
    # 获取应该返回None（异常被捕获）
    result = optimizer.smart_get("key1", DataSourceType.STOCK)
    assert result is None


def test_smart_set_exception(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（smart_set，异常处理）"""
    # 模拟cache.set抛出异常
    def mock_set(key, value, ttl=None):
        raise Exception("Cache error")
    
    monkeypatch.setattr(optimizer.cache, "set", mock_set)
    
    # 设置应该返回False（异常被捕获）
    result = optimizer.smart_set("key1", "value1", DataSourceType.STOCK)
    assert result is False


def test_invalidate_exception(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（invalidate_cache_entry，异常处理）"""
    # 设置一个缓存项
    optimizer.smart_set("key1", "value1", DataSourceType.STOCK)
    
    # 模拟cache.delete抛出异常
    def mock_delete(key):
        raise Exception("Delete error")
    
    monkeypatch.setattr(optimizer.cache, "delete", mock_delete)
    
    # 失效应该返回False（异常被捕获）
    result = optimizer.invalidate_cache_entry("key1", DataSourceType.STOCK)
    assert result is False


def test_should_invalidate_smart_no_entry(optimizer):
    """测试 SmartCacheOptimizer（_should_invalidate_smart，无条目）"""
    # 无条目，应该返回False
    result = optimizer._should_invalidate_smart("nonexistent_key", DataSourceType.STOCK)
    assert result is False


def test_should_invalidate_smart_expired(optimizer):
    """测试 SmartCacheOptimizer（_should_invalidate_smart，已过期）"""
    # 设置一个缓存项
    optimizer.smart_set("key1", "value1", DataSourceType.STOCK, ttl_seconds=1)
    
    # 等待过期
    time.sleep(1.1)
    
    # 应该返回True（已过期）
    result = optimizer._should_invalidate_smart("key1", DataSourceType.STOCK)
    assert result is True


def test_is_market_active_asia_market(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（_is_market_active，亚洲市场时段）"""
    # 模拟当前时间为亚洲市场时段（UTC 0-8）
    mock_now = Mock()
    mock_now.hour = 4
    mock_now.weekday = Mock(return_value=0)
    
    with patch('src.data.cache.smart_cache_optimizer.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = optimizer._is_market_active()
        assert result is True


def test_is_market_active_europe_market(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（_is_market_active，欧洲市场时段）"""
    # 模拟当前时间为欧洲市场时段（UTC 8-16）
    mock_now = Mock()
    mock_now.hour = 12
    mock_now.weekday = Mock(return_value=0)
    
    with patch('src.data.cache.smart_cache_optimizer.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = optimizer._is_market_active()
        assert result is True


def test_is_market_active_us_market(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（_is_market_active，美国市场时段）"""
    # 模拟当前时间为美国市场时段（UTC 14-21）
    mock_now = Mock()
    mock_now.hour = 18
    mock_now.weekday = Mock(return_value=0)
    
    with patch('src.data.cache.smart_cache_optimizer.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = optimizer._is_market_active()
        assert result is True


def test_is_market_active_inactive(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（_is_market_active，非活跃时段）"""
    # 模拟当前时间为非活跃时段
    mock_now = Mock()
    mock_now.hour = 22
    mock_now.weekday = Mock(return_value=0)
    
    with patch('src.data.cache.smart_cache_optimizer.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = optimizer._is_market_active()
        assert result is False


def test_preload_frequent_stock_data(optimizer):
    """测试 SmartCacheOptimizer（_preload_frequent_stock_data）"""
    # 执行预加载
    result = optimizer._preload_frequent_stock_data()
    assert result["status"] == "success"
    assert result["data_type"] == "stock"


def test_in_memory_data_cache_methods(optimizer):
    """测试 _InMemoryDataCache（各种方法）"""
    from src.data.cache.smart_cache_optimizer import _InMemoryDataCache
    
    cache = _InMemoryDataCache()
    
    # 测试set和get
    assert cache.set("key1", "value1") is True
    assert cache.get("key1") == "value1"
    
    # 测试delete
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    # 测试clear
    cache.set("key2", "value2")
    assert cache.clear() is True
    assert cache.get("key2") is None
    
    # 测试get_stats
    stats = cache.get_stats()
    assert "size" in stats
    assert stats["ttl_support"] is False


def test_smart_cache_optimizer_preload_scheduler_exception(optimizer, monkeypatch):
    """测试 SmartCacheOptimizer（预加载调度器，异常处理，覆盖 157-159 行）"""
    # 模拟调度器运行时抛出异常
    # 通过 patch asyncio.run 来触发异常
    import asyncio
    call_count = [0]
    
    original_run = asyncio.run
    def mock_asyncio_run(coro):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Scheduler error")
        return original_run(coro)
    
    monkeypatch.setattr(asyncio, 'run', mock_asyncio_run)
    
    # 直接调用 _preload_scheduler_loop 来触发异常处理
    # 创建一个线程来运行调度器
    import threading
    scheduler_thread = threading.Thread(target=optimizer._preload_scheduler_loop, daemon=True)
    scheduler_thread.start()
    
    # 等待一下让调度器运行并触发异常
    time.sleep(0.1)
    
    # 停止调度器（通过设置标志）
    optimizer._stop_scheduler.set()


def test_smart_cache_optimizer_smart_get_none(optimizer):
    """测试 SmartCacheOptimizer（smart_get，返回 None，覆盖 243 行）"""
    # 获取不存在的 key
    result = optimizer.smart_get("nonexistent_key", DataSourceType.STOCK)
    assert result is None


def test_smart_cache_optimizer_execute_preload_async_duplicate_task(optimizer):
    """测试 SmartCacheOptimizer（异步预加载执行，重复任务，覆盖 192 行）"""
    def condition(context):
        return True
    
    def preload_func():
        return {"data": "test"}
    
    rule = PreloadRule(
        name="test_rule",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    context = optimizer._get_current_context()
    
    # 第一次执行
    import asyncio
    asyncio.run(optimizer._execute_preload_async(rule, context))
    
    # 立即再次执行，应该因为任务已存在而直接返回（覆盖 192 行）
    asyncio.run(optimizer._execute_preload_async(rule, context))


def test_smart_cache_optimizer_execute_preload_async_exception(optimizer):
    """测试 SmartCacheOptimizer（异步预加载执行，异常处理，覆盖 207-208 行）"""
    def condition(context):
        return True
    
    def preload_func():
        raise Exception("Preload async error")
    
    rule = PreloadRule(
        name="test_rule_async",
        data_type=DataSourceType.STOCK,
        condition=condition,
        preload_func=preload_func
    )
    
    context = optimizer._get_current_context()
    
    # 执行异步预加载，应该能处理异常（覆盖 207-208 行）
    import asyncio
    asyncio.run(optimizer._execute_preload_async(rule, context))
    # 应该不会抛出异常


def test_smart_cache_optimizer_should_invalidate_by_freshness(optimizer):
    """测试 SmartCacheOptimizer（基于数据新鲜度的失效，覆盖 313 行）"""
    # 设置一个缓存条目
    key = "test_freshness_key"
    optimizer.smart_set(key, {"data": "test"}, DataSourceType.STOCK)
    
    # 获取缓存条目
    entry = optimizer._cache_entries.get(key)
    assert entry is not None, "缓存条目应该存在"
    
    # 设置条目为过期状态（超过 STOCK 的 600 秒新鲜度要求）
    # STOCK 类型要求 600 秒（10分钟）内的数据，需要超过 600 * 1.2 = 720 秒
    entry.timestamp = datetime.now() - timedelta(seconds=800)
    
    # 检查是否应该失效（覆盖 313 行）
    # _should_invalidate_smart 需要 key 和 data_type 参数
    result = optimizer._should_invalidate_smart(key, DataSourceType.STOCK)
    # 应该返回 True（因为数据超过新鲜度要求）
    assert result is True, "超过新鲜度要求的数据应该被标记为失效"


