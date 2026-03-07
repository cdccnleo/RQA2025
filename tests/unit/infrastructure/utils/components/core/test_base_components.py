#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""base_components 统一工厂逻辑测试。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.infrastructure.utils.components.core import base_components as bc


class _DummyComponent:
    def __init__(self, init_ok: bool = True) -> None:
        self.init_ok = init_ok
        self.init_calls: List[Dict[str, Any]] = []

    def initialize(self, config: Dict[str, Any]) -> bool:
        self.init_calls.append(config)
        return self.init_ok


class _DummyFactory(bc.ComponentFactory):
    def __init__(self) -> None:
        super().__init__()
        self.created: List[Dict[str, Any]] = []

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        self.created.append({"type": component_type, "config": config})
        return _DummyComponent(config.get("init_ok", True))


def test_create_component_with_registered_factory_records_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _DummyFactory()
    emitted: Dict[str, Any] = {}

    def _fake_debug(message: str) -> None:
        emitted.setdefault("debug", []).append(message)

    monkeypatch.setattr(bc.logger, "debug", _fake_debug)

    factory.register_factory("cache", lambda cfg: {"from": "factory", "cfg": cfg})
    result = factory.create_component("cache", {"size": 5})

    assert result == {"from": "factory", "cfg": {"size": 5}}
    assert factory.get_registered_types() == ["cache"]

    stats = factory.get_statistics()["component_stats"]["cache"]
    assert stats["count"] == 1
    assert stats["success_count"] == 1
    assert stats["last_creation"] is not None

    factory.unregister_factory("cache")
    assert factory.get_registered_types() == []


def test_fallback_creation_success_and_failure_paths() -> None:
    factory = _DummyFactory()

    ok_component = factory.create_component("processor", {"init_ok": True})
    assert isinstance(ok_component, _DummyComponent)
    assert ok_component.init_calls == [{"init_ok": True}]

    failed_component = factory.create_component("processor", {"init_ok": False})
    assert isinstance(failed_component, _DummyComponent)

    stats = factory.get_statistics()["component_stats"]["processor"]
    assert stats["count"] == 2
    assert stats["success_count"] == 1
    assert bc.ComponentFactory._calculate_success_rate(factory) == pytest.approx(0.5)
    assert bc.ComponentFactory._calculate_average_time(factory) >= 0.0


def test_create_component_handles_exception_and_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _DummyFactory()
    captured: Dict[str, Any] = {}

    factory.register_factory("boom", lambda cfg: (_ for _ in ()).throw(ValueError("boom")))

    def _fake_error(message: str) -> None:
        captured.setdefault("error", []).append(message)

    monkeypatch.setattr(bc.logger, "error", _fake_error)

    result = factory.create_component("boom")
    assert result is None

    stats = factory.get_statistics()["component_stats"]["boom"]
    assert stats["success_count"] == 0
    assert "创建组件失败" in captured["error"][0]


def test_clear_cache_and_statistics_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _DummyFactory()
    factory._components["x"] = object()  # 直接填充缓存以验证清理

    info_messages: List[str] = []

    def _fake_info(message: str) -> None:
        info_messages.append(message)

    monkeypatch.setattr(bc.logger, "info", _fake_info)

    factory.clear_cache()
    assert factory._components == {}
    assert info_messages == ["组件缓存已清空"]

    # 没有任何统计时，成功率应当为 1.0，平均时间为 0
    factory._statistics.clear()
    assert bc.ComponentFactory._calculate_success_rate(factory) == 1.0
    assert bc.ComponentFactory._calculate_average_time(factory) == 0.0


def test_base_component_factory_dynamic_creation_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    base_factory = bc.BaseComponentFactory()
    logged: List[str] = []

    def _fake_debug(message: str) -> None:
        logged.append(message)

    monkeypatch.setattr(base_factory._logger, "debug", _fake_debug)

    result = base_factory.create_component("dynamic", {})
    assert result is None
    assert any("尝试创建组件" in msg for msg in logged)
