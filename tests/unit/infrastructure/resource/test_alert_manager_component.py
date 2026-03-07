"""
测试告警管理组件

验证AlertManager类的功能，包括规则管理、告警触发、状态管理等
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
import sys

import pytest

import src.infrastructure.resource.models.alert_dataclasses as model_alert_dataclasses
import src.infrastructure.resource.models.alert_enums as model_alert_enums
import src.infrastructure.resource.core.shared_interfaces as model_shared_interfaces

sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.alert_dataclasses",
    model_alert_dataclasses,
)
sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.alert_enums",
    model_alert_enums,
)
sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.shared_interfaces",
    model_shared_interfaces,
)

import src.infrastructure.resource.monitoring.alerts.alert_manager_component as alert_manager_module
from src.infrastructure.resource.monitoring.alerts.alert_manager_component import MonitoringAlertManager
from src.infrastructure.resource.models.alert_dataclasses import AlertRule, PerformanceMetrics, Alert
from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel


class DummyLogger:
    def __init__(self):
        self.log_info = MagicMock()
        self.log_warning = MagicMock()
        self.log_error = MagicMock()
        self.warning = MagicMock()


class DummyErrorHandler:
    def __init__(self):
        self.handle_error = MagicMock()


class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


@pytest.fixture
def manager(monkeypatch):
    logger = DummyLogger()
    error_handler = DummyErrorHandler()
    monkeypatch.setattr(alert_manager_module, "StandardLogger", lambda name=None: logger)
    monkeypatch.setattr(alert_manager_module, "BaseErrorHandler", lambda: error_handler)
    monkeypatch.setattr(alert_manager_module.threading, "Thread", SyncThread)
    mgr = MonitoringAlertManager()
    mgr.logger = logger
    mgr.error_handler = error_handler
    return mgr, logger, error_handler


def make_rule(name="cpu_high", condition="cpu_usage > threshold", threshold=80.0, cooldown=300):
    return AlertRule(
        name=name,
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        alert_level=AlertLevel.WARNING,
        condition=condition,
        threshold=threshold,
        enabled=True,
        cooldown=cooldown,
    )


def make_metrics(**kwargs):
    defaults = {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "disk_usage": 0.0,
        "network_latency": 0.0,
        "test_execution_time": 0.0,
        "test_success_rate": 1.0,
        "active_threads": 0,
    }
    defaults.update(kwargs)
    return PerformanceMetrics(**defaults)


def test_add_alert_rule_replaces_existing(manager):
    mgr, logger, _ = manager
    first = make_rule(threshold=70.0)
    second = make_rule(threshold=90.0)

    mgr.add_alert_rule(first)
    mgr.add_alert_rule(second)

    assert len(mgr.alert_rules) == 1
    assert mgr.alert_rules[0].threshold == 90.0
    logger.warning.assert_called_once()
    messages = [args[0] for args, _ in logger.log_info.call_args_list]
    assert messages.count("添加告警规则: cpu_high") == 2
    assert any(msg.startswith("移除告警规则") for msg in messages)


def test_check_alerts_triggers_handler_once(manager):
    mgr, logger, _ = manager
    rule = make_rule(threshold=50.0)
    metrics = make_metrics(cpu_usage=72.5)
    captured_alerts = []

    mgr.add_alert_rule(rule)
    mgr.register_alert_handler(AlertType.PERFORMANCE_DEGRADATION, captured_alerts.append)

    mgr.check_alerts(metrics)

    assert len(captured_alerts) == 1
    generated_alert = captured_alerts[0]
    assert generated_alert.details["current_value"] == pytest.approx(72.5)
    assert rule.last_triggered is not None
    assert list(mgr.active_alerts.keys())[0] == generated_alert.id
    logger.warning.assert_called_once()


def test_check_alerts_respects_cooldown(manager):
    mgr, _, _ = manager
    rule = make_rule(threshold=10.0)
    metrics = make_metrics(cpu_usage=20.0)
    calls = []

    mgr.add_alert_rule(rule)
    mgr.register_alert_handler(AlertType.PERFORMANCE_DEGRADATION, calls.append)
    mgr.check_alerts(metrics)

    assert len(calls) == 1
    calls.clear()
    mgr.check_alerts(metrics)
    assert len(calls) == 0


def test_evaluate_condition_handles_errors(manager):
    mgr, _, error_handler = manager
    rule = make_rule()

    class BadMetrics:
        pass

    result = mgr._evaluate_condition(rule, BadMetrics(), None)
    assert result is False
    error_handler.handle_error.assert_called_once()


def test_resolve_alert_and_statistics(manager):
    mgr, _, _ = manager
    now = datetime.now()
    recent_alert = Alert(
        id="a1",
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        alert_level=AlertLevel.ERROR,
        message="recent",
        details={},
        timestamp=now,
        source="unit-test",
    )
    old_alert = Alert(
        id="a2",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.CRITICAL,
        message="old",
        details={},
        timestamp=now - timedelta(hours=5),
        source="unit-test",
    )
    mgr.active_alerts = {recent_alert.id: recent_alert, old_alert.id: old_alert}

    mgr.resolve_alert("a2")
    stats = mgr.get_alert_statistics()
    history = mgr.get_alert_history(hours=2)

    assert mgr.active_alerts["a2"].resolved is True
    assert stats["total_alerts"] == 2
    assert stats["resolved_alerts"] == 1
    assert stats["type_distribution"][AlertType.PERFORMANCE_DEGRADATION.value] == 1
    assert stats["level_distribution"][AlertLevel.ERROR.value] == 1
    assert len(history) == 1 and history[0].id == "a1"
