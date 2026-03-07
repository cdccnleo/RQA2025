import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.infrastructure.monitoring.core.component_registry as registry_module


class DummyComponent:
    def __init__(self, value: int = 0):
        self.value = value
        self.started = False
        self.stopped = False
        self.config_history = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update_config(self, cfg):
        self.config_history.append(cfg)

    def get_health_status(self):
        return {"status": "healthy"}


class MonitoringComponent(DummyComponent):
    def start_monitoring(self):
        self.started = True

    def stop_monitoring(self):
        self.stopped = True


class FaultyInitComponent:
    def __init__(self):
        raise RuntimeError("init fail")


class FaultyConfigComponent(DummyComponent):
    def update_config(self, cfg):
        raise RuntimeError("config fail")


class FaultyHealthComponent(DummyComponent):
    def get_health_status(self):
        raise RuntimeError("health fail")


@pytest.fixture
def registry(monkeypatch):
    events = []

    def fake_publish(event_name, payload):
        events.append((event_name, payload))

    monkeypatch.setattr(registry_module, "publish_event", fake_publish)

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return False

    monkeypatch.setattr(registry_module.threading, "Thread", lambda *a, **kw: DummyThread())
    reg = registry_module.InfrastructureComponentRegistry()
    return reg, events


def test_register_component_success(registry):
    reg, events = registry
    assert reg.register_component("dummy", DummyComponent, version="2.0.0", capabilities=["monitor"]) is True
    assert "dummy" in reg.components
    assert events and events[0][0] == "component.registry.registered"
    meta = reg.metadata["dummy"]
    assert meta.version == "2.0.0"
    assert meta.capabilities == ["monitor"]


def test_register_component_dependency_warning(registry, caplog):
    reg, _ = registry
    with caplog.at_level(logging.WARNING):
        reg.register_component("needs", DummyComponent, dependencies=["missing"])
    assert "依赖 missing 未注册" in caplog.text


def test_register_component_failure(registry, monkeypatch, caplog):
    reg, _ = registry

    def fake_component_instance(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(registry_module, "ComponentInstance", fake_component_instance)

    with caplog.at_level(logging.ERROR):
        assert reg.register_component("fault", DummyComponent) is False
    assert "注册组件 fault 失败" in caplog.text


def test_start_stop_restart_and_update_config(registry):
    reg, events = registry
    reg.register_component("svc", DummyComponent, config={"value": 1})

    assert reg.start_component("svc") is True
    instance = reg.get_component_instance("svc")
    assert isinstance(instance, DummyComponent)
    assert instance.started is True
    assert reg.components["svc"].is_running is True
    assert any(evt[0] == "component.lifecycle.started" for evt in events)

    assert reg.update_component_config("svc", {"threshold": 5}) is True
    assert reg.components["svc"].config["threshold"] == 5
    assert instance.config_history[-1]["threshold"] == 5

    assert reg.stop_component("svc") is True
    assert instance.stopped is True
    assert any(evt[0] == "component.lifecycle.stopped" for evt in events)

    restart_result = reg.restart_component("svc")
    assert restart_result is True
    assert reg.components["svc"].start_count >= 1


def test_start_with_start_monitoring_and_stop_monitoring(registry):
    reg, events = registry
    reg.register_component("monitor", MonitoringComponent)
    assert reg.start_component("monitor") is True
    inst = reg.get_component_instance("monitor")
    assert inst.started is True
    assert any(evt[0] == "component.lifecycle.started" for evt in events)

    assert reg.stop_component("monitor") is True
    assert inst.stopped is True


def test_start_component_failure_on_init(registry):
    reg, _ = registry
    reg.register_component("bad", FaultyInitComponent)
    assert reg.start_component("bad") is False


def test_update_component_config_failure(registry):
    reg, _ = registry
    reg.register_component("bad_cfg", FaultyConfigComponent)
    reg.start_component("bad_cfg")
    assert reg.update_component_config("bad_cfg", {"x": 1}) is False


def test_component_operations_missing(registry):
    reg, _ = registry
    assert reg.start_component("missing") is False
    assert reg.stop_component("missing") is False
    assert reg.restart_component("missing") is False
    assert reg.update_component_config("missing", {"x": 1}) is False


def test_list_and_find_components(registry):
    reg, _ = registry
    reg.register_component("a", DummyComponent, capabilities=["metrics"])
    reg.register_component("b", DummyComponent, capabilities=["alerts"])

    names = {entry["name"] for entry in reg.list_components()}
    assert {"a", "b"} <= names

    assert reg.find_components_by_capability("metrics") == ["a"]
    assert reg.find_components_by_capability("alerts") == ["b"]


def test_dependency_checks(registry):
    reg, _ = registry
    reg.register_component("provider", DummyComponent)
    reg.start_component("provider")
    reg.register_component("consumer", DummyComponent, dependencies=["provider"])

    deps = reg.check_dependencies("consumer")
    assert deps["satisfied"] is True

    reg.register_component("client", DummyComponent, dependencies=["missing"])
    deps_missing = reg.check_dependencies("client")
    assert deps_missing["satisfied"] is False
    assert deps_missing["missing"] == ["missing"]


def test_dependency_not_running(registry):
    reg, _ = registry
    reg.register_component("base", DummyComponent)
    reg.register_component("upper", DummyComponent, dependencies=["base"])
    # base not started
    deps = reg.check_dependencies("upper")
    assert "base(not_running)" in deps["missing"]


def test_get_system_health_degraded(registry):
    reg, _ = registry
    reg.register_component("healthy", DummyComponent)
    reg.start_component("healthy")
    reg.metadata["healthy"].update_health_status("healthy")

    reg.register_component("dependent", DummyComponent, dependencies=["missing"])
    health = reg.get_system_health()
    assert health["overall_health"] == "degraded"
    assert health["total_components"] == 2


def test_health_check_loop_updates_metadata(registry, monkeypatch):
    reg, _ = registry
    reg.register_component("healthy", DummyComponent)
    reg.register_component("faulty", FaultyHealthComponent)
    reg.register_component("stopped", DummyComponent)
    reg.start_component("healthy")
    reg.start_component("faulty")

    def stop_after_one(_):
        if getattr(stop_after_one, "called", False):
            raise KeyboardInterrupt
        stop_after_one.called = True

    monkeypatch.setattr(time, "sleep", stop_after_one)

    with pytest.raises(KeyboardInterrupt):
        reg._health_check_loop()

    assert reg.metadata["healthy"].health_status == "healthy"
    assert reg.metadata["faulty"].health_status == "error"
    assert reg.metadata["stopped"].health_status == "unhealthy"


def test_unregister_component(registry):
    reg, events = registry
    reg.register_component("service", DummyComponent)
    assert reg.unregister_component("service") is True
    assert "service" not in reg.components
    assert any(evt[0] == "component.registry.unregistered" for evt in events)


def test_unregister_component_missing(registry):
    reg, _ = registry
    assert reg.unregister_component("missing") is False


def test_unregister_component_with_dependents(registry, caplog):
    reg, _ = registry
    reg.register_component("base", DummyComponent)
    reg.register_component("child", DummyComponent, dependencies=["base"])
    reg.start_component("base")

    with caplog.at_level(logging.WARNING):
        assert reg.unregister_component("base") is True
    assert "仍有依赖组件" in caplog.text
    reg.unregister_component("child")


def test_unregister_component_failure(registry, monkeypatch, caplog):
    reg, _ = registry
    reg.register_component("svc", DummyComponent)
    instance = reg.components["svc"]

    def boom():
        raise RuntimeError("stop fail")

    monkeypatch.setattr(instance, "stop", boom)
    with caplog.at_level(logging.ERROR):
        assert reg.unregister_component("svc") is False
    assert "注销组件 svc 失败" in caplog.text


def test_save_registry_state(tmp_path, registry):
    reg, _ = registry
    reg.register_component("svc", DummyComponent)
    reg.start_component("svc")
    file_path = tmp_path / "registry.json"
    assert reg.save_registry_state(str(file_path)) is True
    content = json.loads(file_path.read_text(encoding="utf-8"))
    assert "components" in content
    assert "svc" in content["components"]


def test_save_registry_state_failure(registry, monkeypatch):
    reg, _ = registry
    reg.register_component("svc", DummyComponent)

    def fake_open(*args, **kwargs):
        raise IOError("write blocked")

    monkeypatch.setattr("builtins.open", fake_open)
    assert reg.save_registry_state("any.json") is False


def test_load_registry_state_success(tmp_path, registry):
    reg, _ = registry
    file_path = tmp_path / "registry.json"
    file_path.write_text(json.dumps({"components": {}, "timestamp": "now"}), encoding="utf-8")
    assert reg.load_registry_state(str(file_path)) is True


def test_load_registry_state_failure(registry):
    reg, _ = registry
    assert reg.load_registry_state("not_existing.json") is False


def test_check_dependencies_component_missing(registry):
    reg, _ = registry
    result = reg.check_dependencies("ghost")
    assert result["reason"] == "component_not_found"
    assert result["satisfied"] is False


def test_find_dependents_helper(registry):
    reg, _ = registry
    reg.register_component("base", DummyComponent)
    reg.register_component("child", DummyComponent, dependencies=["base"])
    assert reg._find_dependents("base") == ["child"]


def test_health_check_loop_global_error(registry, monkeypatch):
    reg, _ = registry

    def fake_sleep(seconds):
        if getattr(fake_sleep, "count", 0) == 0:
            fake_sleep.count = 1
            raise ValueError("sleep fail")
        raise KeyboardInterrupt

    fake_sleep.count = 0
    monkeypatch.setattr("time.sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        reg._health_check_loop()


def test_global_helpers(monkeypatch, registry):
    reg, _ = registry
    monkeypatch.setattr(registry_module, "global_component_registry", reg)
    assert registry_module.register_component("global", DummyComponent) is True
    assert registry_module.get_component("global") is reg.get_component_instance("global")
    reg.unregister_component("global")

