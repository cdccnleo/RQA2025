import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.infrastructure.error.recovery.recovery import (
    AutoRecoveryStrategy,
    ComponentHealth,
    ComponentStatus,
    FallbackManager,
    RecoveryAction,
    RecoveryPriority,
    UnifiedRecoveryManager,
)


class DummyThread:
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        # 不启动后台循环，保持同步测试
        pass

    def is_alive(self):
        return False


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr("threading.Thread", DummyThread)
    mgr = UnifiedRecoveryManager()
    return mgr


def test_fallback_manager_activation_and_deactivation():
    fallback = FallbackManager()
    called = {}

    def fallback_fn():
        called["value"] = True

    fallback.register_fallback("service", fallback_fn)
    assert fallback.activate_fallback("service") is True
    assert "service" in fallback.get_active_fallbacks()
    assert fallback.deactivate_fallback("service") is True
    assert fallback.get_active_fallbacks() == []

    def failing():
        raise RuntimeError("boom")

    fallback.register_fallback("broken", failing)
    assert fallback.activate_fallback("broken") is False


def test_recovery_queue_and_force_recovery(manager):
    component = ComponentHealth(
        component_name="comp",
        status=ComponentStatus.DEGRADED,
        last_check=time.time() - 120,
        failure_count=2,
    )
    manager._component_health["comp"] = component

    manager._check_recovery(component)
    assert manager._recovery_queue

    class FakeStrategy(AutoRecoveryStrategy):
        def can_recover(self, comp):
            return True

        def execute_recovery(self, comp):
            comp.status = ComponentStatus.HEALTHY
            return True

    manager.register_recovery_strategy("fake", FakeStrategy())
    manager._component_health["comp"].status = ComponentStatus.FAILED
    assert manager.force_recovery("comp", "fake") is True
    assert manager._component_health["comp"].status == ComponentStatus.HEALTHY


def test_execute_recovery_action_handles_exceptions(manager):
    action = RecoveryAction(
        action_type="test",
        component_name="comp",
        priority=RecoveryPriority.MEDIUM,
        description="raise error",
        action_function=lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    manager._execute_recovery_action(action)


def test_perform_health_check_updates_timestamp(manager):
    component = ComponentHealth(
        component_name="comp",
        status=ComponentStatus.FAILED,
        last_check=time.time() - 400,
        failure_count=1,
    )
    manager._perform_health_check(component)
    assert component.last_check > time.time() - 5
    assert component.status in {ComponentStatus.RECOVERING, ComponentStatus.FAILED}


def test_apply_auto_recovery_retry_success(manager):
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("retry")
        return "ok"

    result = manager.apply_auto_recovery("retry", flaky)
    assert result == "ok"
    assert attempts["count"] == 2


def test_apply_auto_recovery_retry_failure(manager):
    def always_fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        manager.apply_auto_recovery("retry", always_fail)


def test_get_recovery_stats_structure(manager):
    stats = manager.get_recovery_stats()
    assert {"total_components", "recovery_queue_size", "recovery_strategies"}.issubset(stats.keys())

