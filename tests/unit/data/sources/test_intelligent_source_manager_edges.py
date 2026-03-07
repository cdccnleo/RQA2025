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


import asyncio
import pytest

from src.data.sources.intelligent_source_manager import (
    IntelligentSourceManager,
    DataSourceConfig,
    DataSourceType,
)


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class _FailLoader:
    async def load_data(self, *args, **kwargs):
        raise RuntimeError("loader-fail")


class _OkLoader:
    async def load_data(self, *args, **kwargs):
        return {"ok": True}


def test_no_available_sources_raises():
    mgr = IntelligentSourceManager()
    try:
        with pytest.raises(Exception, match="没有可用的数据源"):
            _run(mgr.load_data("stock", "2024-01-01", "2024-01-31", "1d", ["A"]))
    finally:
        mgr.cleanup()


def test_fallback_second_source_success_and_health_records(tmp_path):
    mgr = IntelligentSourceManager()
    try:
        s1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        s2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        mgr.register_source("s1", s1, _FailLoader())
        mgr.register_source("s2", s2, _OkLoader())

        res = _run(mgr.load_data("stock", "2024-01-01", "2024-01-31", "1d", ["A"]))
        assert res == {"ok": True}

        report = mgr.get_source_info()["health_report"]
        # 两个源都应被记录一次请求（一个失败，一个成功）
        assert report["sources"]["s1"]["total_requests"] >= 1
        assert report["sources"]["s2"]["total_requests"] >= 1
    finally:
        mgr.cleanup()


def test_all_sources_fail_raise():
    mgr = IntelligentSourceManager()
    try:
        s1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        s2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        mgr.register_source("s1", s1, _FailLoader())
        mgr.register_source("s2", s2, _FailLoader())

        with pytest.raises(Exception, match="所有可用数据源都加载失败"):
            _run(mgr.load_data("stock", "2024-01-01", "2024-01-31", "1d", ["A"]))
    finally:
        mgr.cleanup()


def test_enable_disable_affects_best_source():
    mgr = IntelligentSourceManager()
    try:
        s1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        s2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        mgr.register_source("s1", s1, _OkLoader())
        mgr.register_source("s2", s2, _OkLoader())

        best = mgr.get_best_source("stock")
        assert best in {"s1", "s2"}

        mgr.disable_source("s1")
        best2 = mgr.get_best_source("stock")
        assert best2 == "s2"

        mgr.enable_source("s1")
        best3 = mgr.get_best_source("stock")
        assert best3 in {"s1", "s2"}
    finally:
        mgr.cleanup()

import asyncio
import pytest
from datetime import datetime, timedelta

from src.data.sources.intelligent_source_manager import (
    IntelligentSourceManager,
    DataSourceConfig,
    DataSourceType,
)


class _MockLoaderOK:
    async def load_data(self, data_type, start_date, end_date, frequency, symbols=None, **kwargs):
        await asyncio.sleep(0)
        return {"ok": True, "data_type": data_type}


class _MockLoaderFail:
    async def load_data(self, data_type, start_date, end_date, frequency, symbols=None, **kwargs):
        await asyncio.sleep(0)
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_load_data_best_source_then_fallback_and_all_fail():
    mgr = IntelligentSourceManager()
    try:
        # 注册两个数据源：首选失败，次选成功
        c1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        c2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        mgr.register_source("s1", c1, _MockLoaderFail())
        mgr.register_source("s2", c2, _MockLoaderOK())
        res = await mgr.load_data("stock", "2024-01-01", "2024-01-02", symbols=["000001.SZ"])
        assert res["ok"] is True
        # 禁用 s2，确保全部失败时抛出异常
        mgr.disable_source("s2")
        with pytest.raises(Exception):
            await mgr.load_data("stock", "2024-01-01", "2024-01-02")
    finally:
        mgr.cleanup()


def test_health_monitor_update_and_best_source():
    mgr = IntelligentSourceManager()
    try:
        c1 = DataSourceConfig(name="s1", source_type=DataSourceType.NEWS, priority=5, enabled=True)
        mgr.register_source("s1", c1, _MockLoaderOK())
        # 伪造健康记录：成功率高、响应低
        mgr.health_monitor.record_request("s1", response_time_ms=50, success=True)
        best = mgr.get_best_source("news")
        assert best == "s1"
        info = mgr.get_source_info()
        assert "sources" in info and "ranking" in info
    finally:
        mgr.cleanup()


