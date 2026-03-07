import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.infrastructure.monitoring.core.parameter_objects import (
    AlertConditionConfig,
    AlertRuleConfig,
)
from src.infrastructure.monitoring.services.intelligent_alert_system_refactored import (
    AlertHistoryManager,
    AlertProcessor,
    AlertRuleManager,
    IntelligentAlertSystemRefactored,
    NotificationManager,
)


def _make_rule(
    rule_id: str = "rule-1",
    *,
    field: str = "metric",
    operator: str = "gt",
    value: float = 5.0,
    enabled: bool = True,
    cooldown: int = 0,
) -> AlertRuleConfig:
    condition = AlertConditionConfig(field=field, operator=operator, value=value)
    return AlertRuleConfig(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="Test rule",
        condition=condition,
        level="warning",
        enabled=enabled,
        cooldown=cooldown,
    )


def test_alert_rule_manager_basic_operations():
    manager = AlertRuleManager()
    rule = _make_rule()

    assert manager.add_rule(rule) is True
    assert manager.add_rule(rule) is False  # duplicate

    fetched = manager.get_rule(rule.rule_id)
    assert fetched is rule

    assert manager.disable_rule(rule.rule_id) is True
    assert manager.enable_rule(rule.rule_id) is True

    assert manager.remove_rule(rule.rule_id) is True
    assert manager.get_rule(rule.rule_id) is None


def test_alert_rule_manager_failure_and_missing_cases():
    manager = AlertRuleManager()

    class FailingDict(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("fail to store")

    manager.rules = FailingDict()
    assert manager.add_rule(_make_rule("bad-rule")) is False

    # missing rule behaviours
    assert manager.remove_rule("missing") is False
    assert manager.enable_rule("missing") is False
    assert manager.disable_rule("missing") is False


def test_alert_processor_triggers_with_cooldown(monkeypatch):
    manager = AlertRuleManager()
    rule = _make_rule(cooldown=60)
    manager.add_rule(rule)
    processor = AlertProcessor(manager)

    data = {"metric": 10}
    alerts = processor.process_alerts(data)
    assert len(alerts) == 1

    # second call blocked by cooldown
    assert processor.process_alerts(data) == []

    # advance time past cooldown
    processor.last_trigger_times[rule.rule_id] -= timedelta(seconds=61)
    alerts_after_cooldown = processor.process_alerts(data)
    assert len(alerts_after_cooldown) == 1


def test_alert_processor_handles_missing_field():
    manager = AlertRuleManager()
    rule = _make_rule(field="other")
    manager.add_rule(rule)
    processor = AlertProcessor(manager)

    assert processor.process_alerts({"metric": 5}) == []


def test_alert_processor_handles_invalid_operator(monkeypatch):
    manager = AlertRuleManager()
    rule = _make_rule(operator="unknown")
    manager.add_rule(rule)
    processor = AlertProcessor(manager)

    assert processor.process_alerts({"metric": 10}) == []

    # force evaluate condition to raise
    monkeypatch.setattr(processor, "_evaluate_condition", MagicMock(side_effect=RuntimeError("boom")))
    assert processor.process_alerts({"metric": 10}) == []


def test_alert_processor_skips_disabled_rule():
    manager = AlertRuleManager()
    rule = _make_rule(enabled=False)
    manager.add_rule(rule)
    processor = AlertProcessor(manager)

    assert processor.process_alerts({"metric": 10}) == []


@pytest.mark.parametrize(
    "operator,value,data,expected",
    [
        ("gt", 5, {"metric": 6}, True),
        ("lt", 5, {"metric": 4}, True),
        ("eq", 5, {"metric": 5}, True),
        ("ne", 5, {"metric": 4}, True),
        ("ge", 5, {"metric": 5}, True),
        ("le", 5, {"metric": 5}, True),
    ],
)
def test_evaluate_condition_variants(operator, value, data, expected):
    manager = AlertRuleManager()
    rule = _make_rule(operator=operator, value=value)
    processor = AlertProcessor(manager)

    assert processor._evaluate_condition(rule.condition, data) is expected
    # unknown operator path already validated above


def test_alert_history_manager_filters_and_updates():
    history = AlertHistoryManager(max_history_size=2)

    alert_active = {
        "alert_id": "a1",
        "rule_name": "Rule A",
        "description": "desc",
        "level": "warning",
        "status": "active",
        "triggered_at": datetime.now().isoformat(),
    }
    history.add_alert(alert_active)

    alert_resolved = {
        "alert_id": "a2",
        "rule_name": "Rule B",
        "description": "desc",
        "level": "error",
        "status": "resolved",
        "triggered_at": datetime.now().isoformat(),
    }
    history.add_alert(alert_resolved)

    assert len(history.get_alert_history()) == 2
    assert len(history.get_alert_history(level="warning")) == 1
    assert len(history.get_alert_history(status="resolved")) == 1

    assert history.get_active_alerts() == [alert_active]
    assert history.acknowledge_alert("a1") is True
    assert history.resolve_alert("a1") is True

    stats = history.get_alert_statistics()
    assert stats["total_alerts"] == 2
    assert stats["level_distribution"]["warning"] == 1
    assert history.acknowledge_alert("missing") is False
    assert history.resolve_alert("missing") is False


def test_alert_history_manager_limits_size():
    history = AlertHistoryManager(max_history_size=1)
    alert_template = {
        "rule_name": "Rule",
        "description": "desc",
        "level": "warning",
        "status": "active",
        "channels": [],
        "triggered_at": datetime.now().isoformat(),
    }
    history.add_alert({"alert_id": "a1", **alert_template})
    history.add_alert({"alert_id": "a2", **alert_template})

    assert len(history.alert_history) == 1
    assert history.alert_history[0]["alert_id"] == "a2"


def test_notification_manager_handles_channels(caplog):
    manager = NotificationManager()
    alert = {
        "rule_name": "Test",
        "description": "desc",
        "level": "info",
        "channels": ["console", "unknown"],
    }

    assert manager.send_notification(alert) is True

    def failing_channel(alert):
        raise RuntimeError("fail")

    manager.notification_channels["console"] = failing_channel
    assert manager.send_notification({"channels": ["console"]}) is False
    manager.notification_channels["console"] = NotificationManager()._notify_console
    manager._notify_email({"rule_name": "Rule"})
    manager._notify_webhook({"rule_name": "Rule"})
    manager._notify_slack({"rule_name": "Rule"})


def test_system_process_monitoring_data_and_history(monkeypatch):
    system = IntelligentAlertSystemRefactored()
    rule = _make_rule()
    assert system.add_alert_rule(rule) is True

    send_mock = MagicMock(return_value=True)
    system.notification_manager.send_notification = send_mock

    alerts = system.process_monitoring_data({"metric": 10})
    assert alerts and alerts[0]["rule_id"] == rule.rule_id
    assert system.get_active_alerts()
    send_mock.assert_called()


def test_system_process_monitoring_data_handles_exception(monkeypatch):
    system = IntelligentAlertSystemRefactored()
    monkeypatch.setattr(system.alert_processor, "process_alerts", MagicMock(side_effect=RuntimeError("boom")))

    assert system.process_monitoring_data({"metric": 1}) == []


def test_export_alert_history_json(tmp_path):
    system = IntelligentAlertSystemRefactored()
    alert = {
        "alert_id": "a1",
        "rule_id": "r1",
        "rule_name": "Rule",
        "description": "desc",
        "level": "warning",
        "status": "active",
        "channels": [],
        "triggered_at": datetime.now().isoformat(),
        "data": {},
    }
    system.history_manager.add_alert(alert)

    path = tmp_path / "alerts.json"
    assert system.export_alert_history(str(path), "json") is True
    assert path.exists()
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["total_alerts"] == 1


def test_export_alert_history_unsupported_format(tmp_path):
    system = IntelligentAlertSystemRefactored()
    assert system.export_alert_history(str(tmp_path / "alerts.txt"), "xml") is False


def test_export_alert_history_csv(tmp_path):
    system = IntelligentAlertSystemRefactored()
    system.history_manager.add_alert(
        {
            "alert_id": "a1",
            "rule_name": "Rule",
            "description": "desc",
            "level": "warning",
            "status": "active",
            "channels": [],
            "triggered_at": datetime.now().isoformat(),
            "data": {},
        }
    )

    path = tmp_path / "alerts.csv"
    assert system.export_alert_history(str(path), "csv") is True
    assert path.exists()


def test_export_alert_history_handles_exception(monkeypatch, tmp_path):
    system = IntelligentAlertSystemRefactored()
    system.history_manager.add_alert(
        {
            "alert_id": "a1",
            "rule_name": "Rule",
            "description": "desc",
            "level": "warning",
            "status": "active",
            "channels": [],
            "triggered_at": datetime.now().isoformat(),
            "data": {},
        }
    )

    def raise_io_error(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr("builtins.open", raise_io_error)

    assert system.export_alert_history(str(tmp_path / "alerts.json"), "json") is False


def test_system_wrapper_methods(monkeypatch):
    system = IntelligentAlertSystemRefactored()
    rule = _make_rule()
    system.add_alert_rule(rule)

    assert system.get_alert_rule(rule.rule_id) is rule
    assert system.get_all_alert_rules()

    alert = {
        "alert_id": "a1",
        "rule_name": "Rule",
        "description": "desc",
        "level": "warning",
        "status": "active",
        "channels": [],
    }
    system.history_manager.add_alert(alert)

    assert system.acknowledge_alert("a1") is True
    assert system.resolve_alert("a1") is True
    assert system.get_alert_statistics()["total_alerts"] >= 1
    assert system.remove_alert_rule(rule.rule_id) is True


