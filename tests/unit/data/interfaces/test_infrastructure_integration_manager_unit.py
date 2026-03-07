#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import types
import pytest

import src.data.infrastructure_integration_manager as iim


def test_get_manager_initialize_and_health_bridge(monkeypatch):
    # 模拟 DataManagerSingleton.get_instance() 提供 health_bridge
    class DMStub:
        def __init__(self):
            self.health_bridge = object()

    # 将 DataManagerSingleton 指向带 get_instance 的桩
    monkeypatch.setattr(iim, "DataManagerSingleton", types.SimpleNamespace(get_instance=lambda: DMStub()), raising=True)
    mgr = iim.get_data_integration_manager()
    assert mgr.initialize() is True
    hb = mgr.get_health_check_bridge()
    assert hb is not None


def test_publish_data_event_delegates_safely(monkeypatch):
    calls = {}

    class DMStub:
        def publish_data_event(self, et, ed):
            calls["last"] = (et, ed)

    monkeypatch.setattr(iim, "DataManagerSingleton", types.SimpleNamespace(get_instance=lambda: DMStub()), raising=True)
    iim.publish_data_event("evt", {"a": 1})
    assert calls["last"] == ("evt", {"a": 1})


def test_publish_data_event_swallowed_on_exception(monkeypatch):
    class DMStub:
        def publish_data_event(self, et, ed):
            raise RuntimeError("boom")

    monkeypatch.setattr(iim, "DataManagerSingleton", types.SimpleNamespace(get_instance=lambda: DMStub()), raising=True)
    # 不应抛出
    iim.publish_data_event("evt", {"a": 1})


def test_log_and_metric_delegate(monkeypatch):
    log_calls = []
    metric_calls = []
    monkeypatch.setattr(iim, "_log_data_operation", lambda *a, **k: log_calls.append((a, k)), raising=True)
    monkeypatch.setattr(iim, "_record_data_metric", lambda *a, **k: metric_calls.append((a, k)), raising=True)

    iim.log_data_operation("op", "stock", {"k": "v"}, "info")
    iim.record_data_metric("m", 1.23, "stock", {"t": "x"})

    assert log_calls and "op" in log_calls[0][0]
    assert metric_calls and "m" in metric_calls[0][0]


