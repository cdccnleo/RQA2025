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
from typing import Any, Optional, List

import pytest

from src.data.sources.intelligent_source_manager import (
    IntelligentSourceManager,
    DataSourceConfig,
    DataSourceType,
)


class _OKLoader:
    async def load_data(self, data_type: str, start_date: str, end_date: str, frequency: str, symbols: Optional[List[str]] = None, **kwargs: Any):
        return {"ok": True, "from": "ok"}


class _FailLoader:
    async def load_data(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_register_and_best_source_and_load_success(monkeypatch):
    mgr = IntelligentSourceManager()
    try:
        cfg1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        mgr.register_source("s1", cfg1, _OKLoader())

        best = mgr.get_best_source("stock")
        assert best == "s1"

        result = await mgr.load_data("stock", "2025-01-01", "2025-01-02", "1d", ["AAA"])
        assert result and result["ok"] is True
    finally:
        mgr.cleanup()


@pytest.mark.asyncio
async def test_fail_then_fallback_to_next_source(monkeypatch):
    mgr = IntelligentSourceManager()
    try:
        # 优先级高但失败的源
        cfg_fail = DataSourceConfig(name="p1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        mgr.register_source("p1", cfg_fail, _FailLoader())

        # 次优先级但成功的源
        cfg_ok = DataSourceConfig(name="p2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        mgr.register_source("p2", cfg_ok, _OKLoader())

        result = await mgr.load_data("stock", "2025-01-01", "2025-01-02", "1d", ["AAA"])
        assert result and result["from"] == "ok"
    finally:
        mgr.cleanup()


def test_disable_source_and_info_snapshot():
    mgr = IntelligentSourceManager()
    try:
        cfg = DataSourceConfig(name="x", source_type=DataSourceType.CRYPTO, priority=3, enabled=True)
        mgr.register_source("x", cfg, _OKLoader())
        assert mgr.get_best_source("crypto") == "x"

        mgr.disable_source("x")
        assert mgr.get_best_source("crypto") is None

        info = mgr.get_source_info()
        assert "sources" in info and "ranking" in info and "health_report" in info
    finally:
        mgr.cleanup()


