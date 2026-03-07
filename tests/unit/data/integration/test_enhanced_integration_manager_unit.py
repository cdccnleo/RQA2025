#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import types
import pytest
import sys
import importlib

# 兼容相对导入路径差异：将 quality/monitor 与 cache/cache_manager 映射到真实模块
from src.data.quality import monitor as qm_mod
from src.data.cache import cache_manager as cm_mod
sys.modules.setdefault("src.data.integration.quality.monitor", qm_mod)
sys.modules.setdefault("src.data.integration.cache.cache_manager", cm_mod)
# 提供一个带 DataManagerSingleton 的桩模块，避免真实实现差异
_dm_stub_mod = types.SimpleNamespace(DataManagerSingleton=types.SimpleNamespace(get_instance=lambda: _DMStub()))
sys.modules["src.data.integration.data_manager"] = _dm_stub_mod

# 映射完成后再导入被测模块（若已导入则重载）
mod = importlib.import_module("src.data.integration.enhanced_integration_manager")
mod = importlib.reload(mod)
EnhancedDataIntegrationManager = getattr(mod, "EnhancedDataIntegrationManager")
DataStreamConfig = getattr(mod, "DataStreamConfig")


class _DMStub:
    async def load_data(self, data_type, start_date, end_date, frequency, **kwargs):
        return {"rows": 1, "type": data_type}


def test_stream_lifecycle_and_callbacks(monkeypatch):
    # 替换 DataManagerSingleton.get_instance
    dms = types.SimpleNamespace(get_instance=lambda: types.SimpleNamespace(load_data=None))
    monkeypatch.setattr(mod, "DataManagerSingleton", dms, raising=False)

    mgr = EnhancedDataIntegrationManager()
    sid = "s1"
    mgr.create_data_stream(DataStreamConfig(stream_id=sid, data_type="stock"))
    mgr.start_data_stream(sid)
    got = []
    mgr.add_stream_callback(sid, lambda d: got.append(d))
    mgr.data_streams[sid].emit_data({"k": "v"})
    mgr.stop_data_stream(sid)
    assert got and got[0]["k"] == "v"
    mgr.shutdown()


@pytest.mark.asyncio
async def test_distributed_load_local_and_metrics(monkeypatch):
    # 注入 DataManagerSingleton
    dms = types.SimpleNamespace(get_instance=lambda: _DMStub())
    monkeypatch.setattr(mod, "DataManagerSingleton", dms, raising=False)

    mgr = EnhancedDataIntegrationManager()
    # 清空节点以走本地分支
    mgr.node_manager.clear_all_nodes()
    out = await mgr.load_data_distributed("stock", "2025-01-01", "2025-01-02")
    assert out["node_id"] == "local"
    perf = mgr.get_performance_metrics()
    assert "performance" in perf and "cache" in perf
    mgr.shutdown()


def test_alert_history_empty_and_shutdown_idempotent(monkeypatch):
    dms = types.SimpleNamespace(get_instance=lambda: _DMStub())
    monkeypatch.setattr(mod, "DataManagerSingleton", dms, raising=False)
    mgr = EnhancedDataIntegrationManager()
    hist = mgr.get_alert_history(hours=1)
    assert isinstance(hist, list)
    # 重复关闭不应抛异常
    mgr.shutdown()
    mgr.shutdown()


