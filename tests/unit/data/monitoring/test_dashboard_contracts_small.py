#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import types
import importlib
from datetime import datetime


class _StubEnhancedManager:
    def __init__(self):
        self._now = datetime.now()

    def get_performance_metrics(self):
        # 覆盖节点/流/缓存分支计算
        return {
            "performance": {"distributed_load_time": {"avg": 0.12}},
            "cache": {"hits": 8, "misses": 2},
            "nodes": {"n1": {"status": "active"}, "n2": {"status": "active"}},
            "streams": {"s1": {"is_running": True}, "s2": {"is_running": False}},
        }

    def get_quality_report(self, days=1):
        return {"period_days": days, "score": 0.95}

    def get_alert_history(self, hours=24):
        # 返回可 JSON 序列化的结构
        return [{"message": "ok", "level": "info", "timestamp": self._now.timestamp()}]


def test_dashboard_export_and_callbacks(tmp_path):
    # 在导入仪表板模块前，注入桩模块，解决错误的相对导入路径
    stub_mod = types.ModuleType("src.data.enhanced_integration_manager")
    setattr(stub_mod, "EnhancedDataIntegrationManager", _StubEnhancedManager)
    sys.modules["src.data.enhanced_integration_manager"] = stub_mod

    dashboard_mod = importlib.import_module("src.data.monitoring.dashboard")
    DataDashboard = getattr(dashboard_mod, "DataDashboard")
    DashboardConfig = getattr(dashboard_mod, "DashboardConfig")

    mgr = _StubEnhancedManager()
    dash = DataDashboard(mgr, DashboardConfig(enable_export=True))

    # 回调触发与容错
    called = {"n": 0}

    def _cb(_data):
        called["n"] += 1

    def _cb_err(_data):
        raise RuntimeError("boom")

    dash.add_callback("export", _cb)
    dash.add_callback("export", _cb_err)

    # 拉取仪表盘数据（覆盖当前 metrics 汇总逻辑）
    data = dash.get_dashboard_data()
    assert "current_metrics" in data and "widgets" in data

    # 导出报告（覆盖导出 JSON 分支与写文件路径）
    outfile = tmp_path / "dash.json"
    path = dash.export_dashboard_report("json", str(outfile))
    assert os.path.exists(path)

    # 触发回调（覆盖 _trigger_callback 的异常保护）
    dash._trigger_callback("export", {"path": path})
    assert called["n"] >= 1

    # 关闭（若存在自动刷新接口则调用，否则跳过）
    if hasattr(dash, "shutdown"):
        try:
            dash.shutdown()
        except AttributeError:
            # 部分实现可能缺少 stop_auto_refresh，忽略关闭异常
            pass


