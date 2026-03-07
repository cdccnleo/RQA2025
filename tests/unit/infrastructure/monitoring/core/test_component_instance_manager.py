import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.infrastructure.monitoring.core.component_instance_manager import (
    ComponentInstance,
    ComponentInstanceManager,
)


class DummyComponent:
    """简单组件，支持 start/stop/update_config/health_check。"""

    def __init__(self, should_fail=False):
        self.started = False
        self.stopped = False
        self.updated_configs = []
        self.should_fail = should_fail

    def start(self):
        if self.should_fail:
            raise RuntimeError("start failed")
        self.started = True

    def stop(self):
        if self.should_fail:
            raise RuntimeError("stop failed")
        self.stopped = True

    def update_config(self, cfg):
        if self.should_fail:
            raise RuntimeError("update failed")
        self.updated_configs.append(cfg)

    def health_check(self):
        if self.should_fail:
            raise RuntimeError("health failed")
        return {"status": "healthy", "message": "ok"}


# ---------- ComponentInstance ----------

def _make_instance(name="demo", instance=None):
    metadata = {"name": name}
    ci = ComponentInstance(metadata, instance=instance)
    return ci


def test_component_instance_create_start_stop_restart(caplog):
    ci = _make_instance()
    instance = ci.create_instance(DummyComponent, {"should_fail": False})
    assert isinstance(instance, DummyComponent)

    assert ci.start() is True
    assert ci.is_active is True

    assert ci.stop() is True
    assert ci.is_active is False

    assert ci.restart() is True
    assert ci.is_active is True


def test_component_instance_create_fail(caplog):
    ci = _make_instance()

    class FailingComponent:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ci.create_instance(FailingComponent, {})
    assert ci.error_count == 1
    assert "创建组件实例 demo 失败" in ci.last_error


def test_component_instance_start_stop_failures(caplog):
    ci = _make_instance(instance=DummyComponent(should_fail=True))

    assert ci.start() is False
    assert ci.stop() is False
    assert ci.restart() is False
    assert ci.error_count == 3
    assert "失败" in ci.last_error


def test_component_instance_restart_raises_exception(caplog):
    ci = _make_instance(instance=DummyComponent())

    def raising_stop():
        raise RuntimeError("stop boom")

    ci.stop = raising_stop  # type: ignore[assignment]

    assert ci.restart() is False
    assert "重启组件实例 demo 失败" in ci.last_error


def test_component_instance_update_config_success():
    component = DummyComponent()
    ci = _make_instance(instance=component)

    assert ci.update_config({"retry": 3}) is True
    assert ci.config["retry"] == 3
    assert component.updated_configs[-1] == {"retry": 3}


def test_component_instance_update_config_failure(caplog):
    component = DummyComponent(should_fail=True)
    ci = _make_instance(instance=component)

    assert ci.update_config({"retry": 3}) is False
    assert ci.error_count == 1
    assert "更新组件实例 demo 配置失败" in ci.last_error


def test_component_instance_get_status_and_health():
    ci = _make_instance(instance=DummyComponent())
    ci.start()
    status = ci.get_status()
    assert status["is_active"] is True
    assert status["name"] == "demo"

    health = ci.health_check()
    assert health["status"] == "healthy"


def test_component_instance_health_default_path():
    class NoHealthComponent:
        def start(self):
            pass

    ci = _make_instance(instance=NoHealthComponent())
    ci.start()
    result = ci.health_check()
    assert result["status"] == "healthy"
    assert result["message"] == "实例运行正常"


def test_component_instance_health_inactive_and_errors():
    ci = _make_instance()
    health = ci.health_check()
    assert health["status"] == "error"

    ci.instance = DummyComponent()
    health = ci.health_check()
    assert health["status"] == "inactive"

    ci.start()
    ci.instance.should_fail = True
    health = ci.health_check()
    assert health["status"] == "error"
    assert ci.error_count > 0


# ---------- ComponentInstanceManager ----------

@pytest.fixture
def manager():
    return ComponentInstanceManager()


def test_manager_create_and_start_stop_restart(manager):
    instance = manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    assert isinstance(instance, ComponentInstance)

    assert manager.start_instance("dummy") is True
    status = manager.get_instance_status("dummy")
    assert status["is_active"] is True

    assert manager.restart_instance("dummy") is True
    assert manager.stop_instance("dummy") is True

    assert manager.stop_instance("missing") is False


def test_manager_create_duplicate(manager, caplog):
    manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    with caplog.at_level(logging.WARNING):
        manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    assert "重新创建" in caplog.text
    assert manager.get_instance("dummy") is not None


def test_manager_create_instance_failure(manager, caplog):
    class FailingComponent:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        assert manager.create_instance("fail", FailingComponent, {"name": "fail"}) is None
    assert "创建组件实例 fail 失败" in caplog.text


def test_manager_update_and_status(manager):
    manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    manager.start_instance("dummy")

    assert manager.update_instance_config("dummy", {"limit": 5}) is True

    status = manager.get_instance_status("dummy")
    assert status["current_config"]["limit"] == 5

    assert manager.list_instances()
    assert manager.get_active_instances() == ["dummy"]
    counts = manager.get_instance_count()
    assert counts["total"] == 1
    assert counts["active"] == 1
    assert counts["inactive"] == 0


def test_manager_update_missing_and_error(manager, caplog):
    with caplog.at_level(logging.ERROR):
        assert manager.update_instance_config("missing", {}) is False
    assert "未找到" in caplog.text


def test_manager_start_missing(manager, caplog):
    with caplog.at_level(logging.ERROR):
        assert manager.start_instance("missing") is False
    assert "未找到" in caplog.text


def test_manager_restart_missing(manager, caplog):
    with caplog.at_level(logging.ERROR):
        assert manager.restart_instance("missing") is False
    assert "未找到" in caplog.text


def test_manager_stop_missing(manager, caplog):
    with caplog.at_level(logging.WARNING):
        assert manager.stop_instance("missing") is False
    assert "未找到" in caplog.text


def test_manager_cleanup_failed_instances(manager, caplog):
    inst_a = manager.create_instance("a", DummyComponent, {"name": "a"})
    inst_b = manager.create_instance("b", DummyComponent, {"name": "b"})
    inst_a.error_count = 5
    inst_b.error_count = 1

    with caplog.at_level(logging.WARNING):
        removed = manager.cleanup_failed_instances(max_errors=5)
    assert removed == 1
    assert manager.get_instance("a") is None
    assert "清理失败实例: a" in caplog.text


def test_manager_health_check(manager):
    manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    manager.start_instance("dummy")
    health = manager.get_health_status()
    assert health["status"] == "healthy"
    assert health["instance_counts"]["active"] == 1


def test_manager_health_check_warnings(manager):
    health = manager.get_health_status()
    assert "没有运行中的实例" in health["issues"]


def test_manager_health_check_failures(manager, caplog):
    manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    manager.start_instance("dummy")
    manager.get_instance("dummy").is_active = False

    health = manager.get_health_status()
    assert health["status"] in {"warning", "error"}
    # 至少包含未激活提示
    assert any("未激活" in issue for issue in health["issues"])


def test_manager_health_check_error_instances(manager):
    component = DummyComponent()
    ci = manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    ci.instance.health_check = MagicMock(return_value={"status": "error"})
    manager.start_instance("dummy")

    health = manager.get_health_status()
    assert "存在错误实例" in health["issues"][0]


def test_manager_health_check_exception(manager, monkeypatch):
    monkeypatch.setattr(manager, "get_instance_count", MagicMock(side_effect=RuntimeError("boom")))
    health = manager.get_health_status()
    assert health["status"] == "error"
    assert health["error"] == "boom"


def test_manager_shutdown(manager):
    manager.create_instance("dummy", DummyComponent, {"name": "dummy"})
    manager.start_instance("dummy")

    results = manager.stop_all_instances()
    assert results["dummy"] is True
    assert manager.get_instance("dummy") is None

