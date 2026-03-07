import asyncio
import pandas as pd
from unittest.mock import Mock
from typing import Any, Optional

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


import asyncio
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch

import pytest

import src.data.cache.smart_cache_optimizer as optimizer_module
from src.data.cache.smart_cache_optimizer import (
    CacheEntry,
    PreloadRule,
    SmartCacheOptimizer,
    get_smart_cache_optimizer,
)
from src.data.interfaces.standard_interfaces import DataSourceType
# 强制使用标准接口的DataSourceType，确保包含MACRO
import sys
if 'src.data.sources.intelligent_source_manager' in sys.modules:
    del sys.modules['src.data.sources.intelligent_source_manager']
from src.data.interfaces.data_interfaces import IDataCache


class DummyCache(IDataCache):
    """简易缓存实现，满足 IDataCache 接口"""

    def __init__(self):
        self._store = {}

    def get_cached_data(self, key: str):
        return self._store.get(key)

    def set_cached_data(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        self._store[key] = data
        return True

    def invalidate_cache(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear_cache(self) -> bool:
        self._store.clear()
        return True

    def get_cache_stats(self):
        return {"size": len(self._store)}

    # 兼容性方法
    def get(self, key: str):
        return self.get_cached_data(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return self.set_cached_data(key, value, ttl)

    def delete(self, key: str) -> bool:
        return self.invalidate_cache(key)

    def clear(self) -> bool:
        return self.clear_cache()

    def get_stats(self):
        return self.get_cache_stats()


class SpyCache(DummyCache):

    def __init__(self):

        super().__init__()
        self.deleted = []

    def delete(self, key: str) -> bool:

        self.deleted.append(key)
        return super().delete(key)


def _build_optimizer(cache: IDataCache) -> SmartCacheOptimizer:
    """禁用后台线程的 SmartCacheOptimizer 构造辅助函数"""
    with patch.object(SmartCacheOptimizer, "_initialize_preload_rules", lambda self: None), \
         patch.object(SmartCacheOptimizer, "_start_preload_scheduler", lambda self: None):
        optimizer = SmartCacheOptimizer(cache)
    return optimizer


def test_preload_rule_should_execute_and_execute():
    call_log = []

    def _loader():
        call_log.append("loaded")
        return {"status": "ok"}

    rule = PreloadRule(
        name="rule",
        data_type=DataSourceType.STOCK,
        condition=lambda ctx: ctx.get("allow", False),
        preload_func=_loader,
        interval_seconds=60
    )

    assert rule.should_execute({"allow": False}) is False
    assert rule.should_execute({"allow": True}) is True

    # 执行成功会刷新 last_execution
    result = rule.execute({})
    assert result == {"status": "ok"}
    assert call_log == ["loaded"]
    assert rule.last_execution is not None

    # 时间未到，不应重复执行
    rule.last_execution = datetime.now()
    assert rule.should_execute({"allow": True}) is False

    # 当超过间隔时应允许执行
    rule.last_execution = datetime.now() - timedelta(seconds=120)
    assert rule.should_execute({"allow": True}) is True

    # 执行失败时返回 None 且 last_execution 不更新
    rule.last_execution = datetime.now()

    def _failing():
        raise RuntimeError("boom")

    rule.preload_func = _failing
    last_exec = rule.last_execution
    assert rule.execute({}) is None
    assert rule.last_execution == last_exec


def test_preload_scheduler_loop_triggers_rule(monkeypatch):
    optimizer = _build_optimizer(DummyCache())
    try:
        rule = PreloadRule(
            name="scheduled",
            data_type=DataSourceType.NEWS,
            condition=lambda ctx: True,
            preload_func=lambda: {"data": 1},
            interval_seconds=0
        )
        optimizer._preload_rules = [rule]

        class Stopper:
            def __init__(self):
                self.calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls > 1

            def set(self):
                self.calls = 2

        optimizer._stop_scheduler = Stopper()
        run_calls = []

        async def fake_execute(self, rule, context):
            run_calls.append((rule.name, context["hour"]))

        monkeypatch.setattr(SmartCacheOptimizer, "_execute_preload_async", fake_execute)
        monkeypatch.setattr(optimizer_module.time, "sleep", lambda *_: None)

        optimizer._preload_scheduler_loop()
        assert run_calls and run_calls[0][0] == "scheduled"
    finally:
        optimizer.shutdown()


@pytest.mark.asyncio
async def test_execute_preload_async_handles_success_and_duplicates(monkeypatch):
    optimizer = _build_optimizer(DummyCache())
    try:
        rule = PreloadRule(
            name="async_rule",
            data_type=DataSourceType.NEWS,
            condition=lambda _: True,
            preload_func=lambda: "ok",
            interval_seconds=0
        )

        await optimizer._execute_preload_async(rule, {})
        assert not optimizer._preload_tasks

        fixed_now = datetime(2025, 1, 1, 0, 0, 0)

        class FixedDateTime(type(fixed_now)):
            @classmethod
            def now(cls, tz=None):
                return fixed_now

        monkeypatch.setattr(optimizer_module, "datetime", FixedDateTime)
        duplicate_task_id = f"{rule.name}_{FixedDateTime.now().isoformat()}"
        optimizer._preload_tasks.add(duplicate_task_id)

        await optimizer._execute_preload_async(rule, {})
        assert duplicate_task_id in optimizer._preload_tasks
        optimizer._preload_tasks.clear()
    finally:
        optimizer.shutdown()


def test_get_current_context_and_market_activity(monkeypatch):
    optimizer = _build_optimizer(DummyCache())
    try:
        class DummyDateTime(datetime):
            current = datetime(2025, 1, 1, 0, 0, 0)

            @classmethod
            def now(cls, tz=None):
                return cls.current

        monkeypatch.setattr(optimizer_module, "datetime", DummyDateTime)

        for hour in (6, 12, 18):
            DummyDateTime.current = datetime(2025, 1, 1, hour, 0, 0)
            assert optimizer._is_market_active() is True

        DummyDateTime.current = datetime(2025, 1, 1, 23, 0, 0)
        assert optimizer._is_market_active() is False

        DummyDateTime.current = datetime(2025, 1, 1, 9, 0, 0)
        context = optimizer._get_current_context()
        assert context["hour"] == 9
        assert "day_of_week" in context
        assert context["market_active"] is True
    finally:
        optimizer.shutdown()


def test_add_preload_rule_and_metrics(monkeypatch):
    optimizer = _build_optimizer(DummyCache())
    try:
        optimizer.add_preload_rule(
            name="custom",
            data_type=DataSourceType.CRYPTO,
            condition=lambda ctx: ctx.get("market_active", False),
            preload_func=lambda: {"data": 42},
            priority=4,
            interval_seconds=10
        )

        optimizer.smart_set("btc", 123, DataSourceType.CRYPTO)
        entry = optimizer._cache_entries["btc"]
        assert entry.ttl_seconds == 300
        assert entry.priority == 3
        assert 295 <= (entry.expiry_time - entry.timestamp).total_seconds() <= 305

        metrics = optimizer.get_performance_metrics()
        assert metrics["cache_status"]["current_entries"] == 1
        assert metrics["preload_status"]["active_rules"] >= 1
    finally:
        optimizer.shutdown()


def test_smart_get_handles_expired_and_cache_hits(monkeypatch):
    spy_cache = SpyCache()
    optimizer = _build_optimizer(spy_cache)
    try:
        # 构造过期条目，smart_get 应触发智能失效并返回 None
        expired_entry = CacheEntry(
            key="legacy",
            data_type=DataSourceType.STOCK,
            value="stale",
            timestamp=datetime.now() - timedelta(hours=2),
            expiry_time=datetime.now() - timedelta(seconds=10),
        )
        optimizer._cache_entries["legacy"] = expired_entry
        spy_cache.set("legacy", "stale")

        assert optimizer.smart_get("legacy", DataSourceType.STOCK) is None
        assert "legacy" in spy_cache.deleted
        assert "legacy" not in optimizer._cache_entries

        # 构造有效条目，smart_get 应命中缓存并更新统计
        fresh_entry = CacheEntry(
            key="fresh",
            data_type=DataSourceType.CRYPTO,
            value="live",
            timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=1),
        )
        optimizer._cache_entries["fresh"] = fresh_entry
        spy_cache.set("fresh", "live")

        result = optimizer.smart_get("fresh", DataSourceType.CRYPTO)
        assert result == "live"
        assert fresh_entry.access_count == 1
        assert optimizer._access_history
    finally:
        optimizer.shutdown()


def test_invalidation_rules_for_access_pattern_and_freshness():
    optimizer = _build_optimizer(DummyCache())
    try:
        entry = CacheEntry(
            key="slow",
            data_type=DataSourceType.NEWS,
            value="data",
            timestamp=datetime.now() - timedelta(minutes=40),
        )
        entry.last_access = datetime.now() - timedelta(days=2)
        entry.access_count = 1

        assert optimizer._should_invalidate_by_access_pattern(entry) is True

        crypto_entry = CacheEntry(
            key="btc",
            data_type=DataSourceType.CRYPTO,
            value="data",
            timestamp=datetime.now() - timedelta(minutes=20),
        )
        assert optimizer._should_invalidate_by_freshness(crypto_entry) is True

        macro_entry = CacheEntry(
            key="macro",
            data_type=DataSourceType.NEWS,
            value="macro",
            timestamp=datetime.now() - timedelta(minutes=5),
        )
        assert optimizer._should_invalidate_by_freshness(macro_entry) is False
    finally:
        optimizer.shutdown()


def test_shutdown_waits_for_active_thread():
    optimizer = _build_optimizer(DummyCache())
    try:
        class DummyThread:
            def __init__(self):
                self.join_called = False

            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.join_called = True

        dummy_thread = DummyThread()
        optimizer._scheduler_thread = dummy_thread
        optimizer.shutdown()
        assert dummy_thread.join_called is True
    finally:
        optimizer.shutdown()


def test_get_smart_cache_optimizer_default_and_custom_cache():
    with patch.object(SmartCacheOptimizer, "_start_preload_scheduler", lambda self: None):
        default_optimizer = get_smart_cache_optimizer()

    try:
        assert default_optimizer.smart_set("symbol", "value", DataSourceType.STOCK) is True
        assert default_optimizer.smart_get("symbol", DataSourceType.STOCK) == "value"
    finally:
        default_optimizer.shutdown()

    custom_cache = DummyCache()
    with patch.object(SmartCacheOptimizer, "_start_preload_scheduler", lambda self: None):
        custom_optimizer = get_smart_cache_optimizer(cache=custom_cache, config={"foo": "bar"})

    try:
        assert custom_optimizer.cache is custom_cache
        custom_optimizer.smart_set("custom", "cache", DataSourceType.NEWS)
        assert custom_optimizer.smart_get("custom", DataSourceType.NEWS) == "cache"
    finally:
        custom_optimizer.shutdown()

