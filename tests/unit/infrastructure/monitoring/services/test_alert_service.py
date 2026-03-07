import importlib
import json
import sys
import types
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import Mock

import pytest

constants_module = types.ModuleType(
    "src.infrastructure.monitoring.services.core.constants"
)
constants_module.NOTIFICATION_MAX_RETRIES = 3
constants_module.NOTIFICATION_RETRY_DELAY = 1
constants_module.ALERT_LEVEL_INFO = "info"
constants_module.ALERT_LEVEL_WARNING = "warning"
constants_module.ALERT_LEVEL_ERROR = "error"
constants_module.ALERT_LEVEL_CRITICAL = "critical"
sys.modules["src.infrastructure.monitoring.services.core.constants"] = constants_module

exceptions_module = types.ModuleType(
    "src.infrastructure.monitoring.services.core.exceptions"
)


class MonitoringException(Exception):
    pass


class NotificationError(Exception):
    def __init__(self, message, notification_type=None, recipient=None, details=None):
        super().__init__(message)


def handle_monitoring_exception(context):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


exceptions_module.MonitoringException = MonitoringException
exceptions_module.NotificationError = NotificationError
exceptions_module.handle_monitoring_exception = handle_monitoring_exception
sys.modules[
    "src.infrastructure.monitoring.services.core.exceptions"
] = exceptions_module

alert_module = importlib.import_module(
    "src.infrastructure.monitoring.services.alert_service"
)


class DummyThread:
    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon
        self._started = False

    def start(self):
        self._started = True
        # 不自动运行 target，避免后台线程影响测试

    def join(self, timeout=None):
        self._started = False

    def is_alive(self) -> bool:
        return self._started


@pytest.fixture(autouse=True)
def disable_components(monkeypatch):
    monkeypatch.setattr(alert_module, "COMPONENTS_AVAILABLE", False)
    monkeypatch.setattr(alert_module.threading, "Thread", DummyThread)
    yield


@pytest.fixture
def alert_system():
    system = alert_module.IntelligentAlertSystem()
    yield system
    system.shutdown()


@pytest.fixture
def alert_module_fixture():
    """提供alert_module的fixture"""
    return alert_module


@pytest.fixture
def sample_alert():
    return alert_module.Alert(
        alert_id="alert_999999",
        rule_id="rule_sample",
        title="Sample",
        message="Sample message",
        level=alert_module.AlertLevel.WARNING,
        data={"value": 42},
        created_at=datetime(2025, 1, 1, 12, 0, 0),
    )


def test_evaluate_condition_operators(alert_system):
    data = {"value": 10, "text": "hello123"}

    def check(condition: Dict[str, Any]) -> bool:
        return alert_system.evaluate_condition(condition, data)

    assert check({"operator": "eq", "field": "value", "value": 10})
    assert not check({"operator": "eq", "field": "value", "value": 5})
    assert check({"operator": "gt", "field": "value", "value": 5})
    assert check({"operator": "gte", "field": "value", "value": 10})
    assert check({"operator": "regex", "field": "text", "value": r"hello\d{3}"})
    assert not check({"operator": "contains", "field": "text", "value": "world"})
    assert not check({"operator": "eq", "field": "missing", "value": 1})
    assert not check({"operator": "unknown", "field": "value", "value": 1})


def test_check_alerts_and_notifications(alert_system, monkeypatch):
    class DummyNotifier(alert_module.IAlertNotifier):
        def __init__(self):
            self.calls = []

        def send_notification(self, alert, recipient):
            self.calls.append((alert.alert_id, recipient))
            return True

    notifier = DummyNotifier()
    alert_system.register_notifier("console", notifier)

    rule = alert_module.AlertRule(
        rule_id="rule_cpu",
        name="CPU 高",
        description="CPU 超阈值",
        condition={"operator": "gt", "field": "cpu", "value": 80},
        level=alert_module.AlertLevel.WARNING,
        channels=[alert_module.AlertChannel.CONSOLE],
        cooldown=60,
    )
    alert_system.add_alert_rule(rule)

    data = {"cpu": 90}
    assert alert_system.check_alerts(data, source="unit_test") is True

    alert = alert_system.alert_queue.get_nowait()
    alert_system._send_notifications(alert)
    alert_system.alert_queue.task_done()

    assert notifier.calls
    assert alert_system.rule_last_triggered["rule_cpu"]

    assert alert_system.check_alerts(data, source="unit_test") is False


def test_acknowledge_and_resolve(alert_system):
    rule = alert_module.AlertRule(
        rule_id="rule_error",
        name="错误",
        description="出现错误",
        condition={"operator": "eq", "field": "status", "value": "error"},
        level=alert_module.AlertLevel.ERROR,
        channels=[alert_module.AlertChannel.CONSOLE],
    )
    alert_system.add_alert_rule(rule)

    alert_system.register_notifier(
        alert_module.AlertChannel.CONSOLE,
        alert_module.ConsoleNotifier(),
    )

    alert_system.check_alerts({"status": "error"})
    alert = alert_system.alert_queue.get_nowait()
    alert_system.alerts[alert.alert_id] = alert

    alert_system.acknowledge_alert(alert.alert_id, user="tester")
    assert alert.status == alert_module.AlertStatus.ACKNOWLEDGED
    assert alert.acknowledged_by == "tester"

    alert_system.resolve_alert(alert.alert_id)
    assert alert.status == alert_module.AlertStatus.RESOLVED


def test_send_notifications_multiple_channels(alert_system, sample_alert, monkeypatch):
    rule = alert_module.AlertRule(
        rule_id=sample_alert.rule_id,
        name="多渠道告警",
        description="Multi-channel alert",
        condition={"operator": "eq", "field": "value", "value": 42},
        level=alert_module.AlertLevel.ERROR,
        channels=[alert_module.AlertChannel.SLACK, alert_module.AlertChannel.CONSOLE],
    )
    alert_system.add_alert_rule(rule)
    alert_system.alerts[sample_alert.alert_id] = sample_alert

    recipients_map = {
        alert_module.AlertChannel.SLACK: ["#alerts", "#trading"],
        alert_module.AlertChannel.CONSOLE: ["console"],
    }
    monkeypatch.setattr(
        alert_system,
        "_get_recipients",
        lambda channel, level: recipients_map[channel],
    )

    class RecordingNotifier(alert_module.IAlertNotifier):
        def __init__(self):
            self.calls = []

        def send_notification(self, alert, recipient):
            self.calls.append(recipient)
            return True

    slack_notifier = RecordingNotifier()
    console_notifier = RecordingNotifier()

    alert_system.register_notifier(alert_module.AlertChannel.SLACK, slack_notifier)
    alert_system.register_notifier(alert_module.AlertChannel.CONSOLE, console_notifier)

    alert_system._send_notifications(sample_alert)

    assert slack_notifier.calls == ["#alerts", "#trading"]
    assert console_notifier.calls == ["console"]


def test_send_notifications_missing_notifier(alert_system, sample_alert, monkeypatch):
    rule = alert_module.AlertRule(
        rule_id=sample_alert.rule_id,
        name="缺少通知器",
        description="No notifier",
        condition={"operator": "eq", "field": "value", "value": 42},
        level=alert_module.AlertLevel.ERROR,
        channels=[alert_module.AlertChannel.SLACK],
    )
    alert_system.add_alert_rule(rule)
    alert_system.alerts[sample_alert.alert_id] = sample_alert

    # 没有注册 Slack 通知器，调用应静默返回且不抛异常
    alert_system._send_notifications(sample_alert)
    assert not alert_system.notifications


def test_email_notifier_success(monkeypatch, sample_alert):
    smtp_state = {"starttls": 0, "login": 0, "sendmail": []}

    class DummySMTP:
        def __init__(self, host, port):
            smtp_state["init"] = (host, port)

        def starttls(self):
            smtp_state["starttls"] += 1

        def login(self, username, password):
            smtp_state["login"] += 1
            smtp_state["credentials"] = (username, password)

        def sendmail(self, from_email, recipient, message):
            smtp_state["sendmail"].append((from_email, recipient))

    monkeypatch.setattr(alert_module.smtplib, "SMTP", DummySMTP)

    notifier = alert_module.EmailNotifier(
        {
            "host": "smtp.example.com",
            "port": 25,
            "username": "alert@example.com",
            "password": "secret",
            "use_tls": True,
        }
    )

    result = notifier.send_notification(sample_alert, "ops@example.com")

    assert result is True
    assert smtp_state["starttls"] == 1
    assert smtp_state["login"] == 1
    assert smtp_state["sendmail"] == [("alert@example.com", "ops@example.com")]


def test_email_notifier_failure(monkeypatch, sample_alert):
    class DummySMTP:
        def __init__(self, host, port):
            pass

        def login(self, username, password):
            raise RuntimeError("login failed")

    monkeypatch.setattr(alert_module.smtplib, "SMTP", DummySMTP)

    notifier = alert_module.EmailNotifier(
        {
            "host": "smtp.example.com",
            "port": 25,
            "username": "alert@example.com",
            "password": "secret",
        }
    )

    result = notifier.send_notification(sample_alert, "ops@example.com")
    assert result is False


def test_email_notifier_requires_config():
    with pytest.raises(ValueError):
        alert_module.EmailNotifier({"host": "smtp.example.com"})


def test_webhook_notifier_success(monkeypatch, sample_alert):
    calls = []

    class DummyResponse:
        status_code = 200

    def fake_post(url, json, headers, timeout):
        calls.append((url, json["alert_id"], timeout))
        return DummyResponse()

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.WebhookNotifier("https://hooks.example.com/alert")
    result = notifier.send_notification(sample_alert, "webhook-recipient")

    assert result is True
    assert calls and calls[0][0] == "https://hooks.example.com/alert"


def test_webhook_notifier_failure(monkeypatch, sample_alert):
    class DummyResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_post(url, json, headers, timeout):
        return DummyResponse(500)

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.WebhookNotifier("https://hooks.example.com/alert")
    assert notifier.send_notification(sample_alert, "webhook-recipient") is False


def test_webhook_notifier_exception(monkeypatch, sample_alert):
    def fake_post(url, json, headers, timeout):
        raise RuntimeError("network error")

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.WebhookNotifier("https://hooks.example.com/alert")
    assert notifier.send_notification(sample_alert, "webhook-recipient") is False


def test_slack_notifier_success(monkeypatch, sample_alert):
    class DummyResponse:
        status_code = 200

    def fake_post(url, json, timeout):
        return DummyResponse()

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.SlackNotifier("https://hooks.slack.com/services/abc")
    assert notifier.send_notification(sample_alert, "#alerts") is True


def test_slack_notifier_http_error(monkeypatch, sample_alert):
    class DummyResponse:
        status_code = 500

    def fake_post(url, json, timeout):
        return DummyResponse()

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.SlackNotifier("https://hooks.slack.com/services/abc")
    assert notifier.send_notification(sample_alert, "#alerts") is False


def test_slack_notifier_exception(monkeypatch, sample_alert):
    def fake_post(url, json, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(alert_module.requests, "post", fake_post)

    notifier = alert_module.SlackNotifier("https://hooks.slack.com/services/abc")

    with pytest.raises(alert_module.NotificationError):
        notifier.send_notification(sample_alert, "#alerts")


def test_register_notifier_accepts_string(alert_system):
    notifier = alert_module.ConsoleNotifier()
    alert_system.register_notifier("slack", notifier)

    assert alert_module.AlertChannel.SLACK in alert_system.notifiers
    assert alert_system.notifiers[alert_module.AlertChannel.SLACK] is notifier


def test_check_alerts_respects_cooldown(alert_system):
    rule = alert_module.AlertRule(
        rule_id="rule_cooldown",
        name="冷却告警",
        description="Cooldown test",
        condition={"operator": "gt", "field": "cpu", "value": 10},
        level=alert_module.AlertLevel.WARNING,
        channels=[alert_module.AlertChannel.CONSOLE],
        cooldown=300,
    )
    alert_system.add_alert_rule(rule)
    alert_system.register_notifier(
        alert_module.AlertChannel.CONSOLE, alert_module.ConsoleNotifier()
    )

    assert alert_system.check_alerts({"cpu": 20}) is True
    alert = alert_system.alert_queue.get_nowait()
    alert_system.alert_queue.task_done()

    assert alert.alert_id.startswith("alert_")
    assert alert_system.check_alerts({"cpu": 25}) is False


def test_remove_alert_rule(alert_system):
    rule = alert_module.AlertRule(
        rule_id="rule_remove",
        name="待移除",
        description="Remove rule",
        condition={"operator": "eq", "field": "status", "value": "bad"},
        level=alert_module.AlertLevel.ERROR,
        channels=[alert_module.AlertChannel.CONSOLE],
    )
    alert_system.add_alert_rule(rule)
    assert "rule_remove" in alert_system.rules

    alert_system.remove_alert_rule("rule_remove")
    assert "rule_remove" not in alert_system.rules


def test_get_active_alerts(alert_system):
    active = alert_module.Alert(
        alert_id="alert_active",
        rule_id="rule_a",
        title="Active",
        message="Active message",
        level=alert_module.AlertLevel.WARNING,
        status=alert_module.AlertStatus.ACTIVE,
    )
    resolved = alert_module.Alert(
        alert_id="alert_resolved",
        rule_id="rule_b",
        title="Resolved",
        message="Resolved message",
        level=alert_module.AlertLevel.ERROR,
        status=alert_module.AlertStatus.RESOLVED,
    )
    alert_system.alerts[active.alert_id] = active
    alert_system.alerts[resolved.alert_id] = resolved

    active_alerts = alert_system.get_active_alerts()
    assert len(active_alerts) == 1
    assert active_alerts[0].alert_id == "alert_active"


def test_get_alert_history_filters(alert_system):
    now = datetime.now()
    alert_recent = alert_module.Alert(
        alert_id="alert_recent",
        rule_id="rule_recent",
        title="Recent",
        message="Recent alert",
        level=alert_module.AlertLevel.ERROR,
        status=alert_module.AlertStatus.ACTIVE,
        created_at=now - timedelta(minutes=5),
    )
    alert_old = alert_module.Alert(
        alert_id="alert_old",
        rule_id="rule_old",
        title="Old",
        message="Old alert",
        level=alert_module.AlertLevel.WARNING,
        status=alert_module.AlertStatus.RESOLVED,
        created_at=now - timedelta(hours=2),
    )
    alert_system.alerts = {
        alert_recent.alert_id: alert_recent,
        alert_old.alert_id: alert_old,
    }

    history = alert_system.get_alert_history(start_time=now - timedelta(minutes=30))
    assert len(history) == 1
    assert history[0].alert_id == "alert_recent"

    warning_history = alert_system.get_alert_history(level=alert_module.AlertLevel.WARNING)
    assert len(warning_history) == 1
    assert warning_history[0].alert_id == "alert_old"


def test_export_alert_report(tmp_path, alert_system):
    now = datetime.now()
    alert = alert_module.Alert(
        alert_id="alert_export",
        rule_id="rule_export",
        title="导出告警",
        message="需要导出",
        level=alert_module.AlertLevel.CRITICAL,
        status=alert_module.AlertStatus.ACKNOWLEDGED,
        created_at=now - timedelta(minutes=1),
    )
    alert_system.alerts[alert.alert_id] = alert

    report_path = tmp_path / "alert_report.json"
    alert_system.export_alert_report(
        start_time=now - timedelta(minutes=5),
        end_time=now + timedelta(minutes=5),
        file_path=str(report_path),
    )

    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["total_alerts"] == 1
    assert data["alerts_by_level"]["critical"] == 1
    assert data["alerts_by_status"]["acknowledged"] == 1


def test_console_notifier_outputs(sample_alert, capsys):
    notifier = alert_module.ConsoleNotifier()
    assert notifier.send_notification(sample_alert, "console") is True

    captured = capsys.readouterr()
    assert "告警通知" in captured.out
    assert "Sample message" in captured.out


def test_get_recipients_fallback(alert_system):
    recipients_console = alert_system._get_recipients(
        alert_module.AlertChannel.CONSOLE, alert_module.AlertLevel.INFO
    )
    assert recipients_console == ["console"]

    recipients_sms = alert_system._get_recipients(
        alert_module.AlertChannel.SMS, alert_module.AlertLevel.WARNING
    )
    assert recipients_sms == []


def test_alert_json_encoder_with_enum(alert_module_fixture):
    """测试AlertJSONEncoder处理枚举类型"""
    encoder = alert_module_fixture.AlertJSONEncoder()
    result = encoder.default(alert_module_fixture.AlertLevel.CRITICAL)
    assert result == "critical"


def test_alert_json_encoder_with_datetime(alert_module_fixture):
    """测试AlertJSONEncoder处理datetime类型"""
    encoder = alert_module_fixture.AlertJSONEncoder()
    now = datetime.now()
    result = encoder.default(now)
    assert isinstance(result, str)
    assert "T" in result or "-" in result


def test_alert_json_encoder_fallback(alert_module_fixture):
    """测试AlertJSONEncoder的fallback到super().default"""
    encoder = alert_module_fixture.AlertJSONEncoder()
    # 传递一个不支持的类型，应该调用super().default
    try:
        result = encoder.default(object())
        # 如果成功，应该返回某种表示
        assert result is not None
    except TypeError:
        # 如果抛出TypeError，这也是预期的行为
        pass


def test_alert_rule_post_init_channels_conversion(alert_module_fixture):
    """测试AlertRule.__post_init__中channels从字符串转换为枚举"""
    rule = alert_module_fixture.AlertRule(
        rule_id="test_rule",
        name="Test Rule",
        description="Test",
        condition={"operator": "eq", "field": "test", "value": 1},
        level=alert_module_fixture.AlertLevel.INFO,
        channels=["console", "email"]  # 字符串列表
    )
    # 应该被转换为枚举列表
    assert all(isinstance(ch, alert_module_fixture.AlertChannel) for ch in rule.channels)


def test_slack_notifier_exception_handling(alert_module_fixture, sample_alert, monkeypatch):
    """测试SlackNotifier的异常处理"""
    notifier = alert_module_fixture.SlackNotifier(webhook_url="http://test.com")
    
    # 模拟requests.post抛出异常
    def failing_post(*args, **kwargs):
        raise Exception("Network error")
    
    monkeypatch.setattr("requests.post", failing_post)
    
    with pytest.raises(alert_module_fixture.NotificationError):
        notifier.send_notification(sample_alert, "#test-channel")


def test_console_notifier_exception_handling(alert_module_fixture, sample_alert, monkeypatch):
    """测试ConsoleNotifier的异常处理"""
    notifier = alert_module_fixture.ConsoleNotifier()
    
    # 模拟print抛出异常
    def failing_print(*args, **kwargs):
        raise IOError("Print error")
    
    monkeypatch.setattr("builtins.print", failing_print)
    
    result = notifier.send_notification(sample_alert, "console")
    assert result is False


def test_evaluate_condition_with_components(alert_system, monkeypatch):
    """测试使用条件评估器组件评估条件"""
    # 模拟组件可用
    mock_evaluator = Mock()
    mock_evaluator.evaluate.return_value = True
    alert_system._condition_evaluator = mock_evaluator
    
    condition = {"operator": "eq", "field": "test", "value": 1}
    data = {"test": 1}
    
    result = alert_system.evaluate_condition(condition, data)
    assert result is True
    mock_evaluator.evaluate.assert_called_once_with(condition, data)


def test_evaluate_condition_operators(alert_system):
    """测试各种运算符评估"""
    # 测试 ne (不等于)
    assert alert_system.evaluate_condition(
        {"operator": "ne", "field": "x", "value": 1},
        {"x": 2}
    ) is True
    
    # 测试 lt (小于)
    assert alert_system.evaluate_condition(
        {"operator": "lt", "field": "x", "value": 10},
        {"x": 5}
    ) is True
    
    # 测试 lte (小于等于)
    assert alert_system.evaluate_condition(
        {"operator": "lte", "field": "x", "value": 10},
        {"x": 10}
    ) is True


def test_evaluate_condition_exception_handling(alert_system):
    """测试条件评估的异常处理"""
    # 传递无效的条件，应该返回False而不是抛出异常
    result = alert_system.evaluate_condition(
        {"operator": "invalid", "field": "x", "value": 1},
        {"x": 1}
    )
    assert result is False


def test_check_alerts_exception_handling(alert_system, monkeypatch):
    """测试check_alerts的异常处理"""
    # 模拟_create_alert抛出异常
    def failing_create(*args, **kwargs):
        raise Exception("Create error")
    
    monkeypatch.setattr(alert_system, "_create_alert", failing_create)
    
    # 应该不会抛出异常，而是记录错误
    result = alert_system.check_alerts({"cpu_usage": 90}, "test")
    # 可能返回None或False
    assert result is None or result is False


def test_send_notifications_no_rule(alert_system, sample_alert):
    """测试_send_notifications当规则不存在时"""
    # 确保规则不存在
    alert_system.rules.clear()
    
    # 应该不会抛出异常
    alert_system._send_notifications(sample_alert)


def test_create_default_rules(alert_system):
    """测试创建默认规则"""
    alert_system.create_default_rules()
    
    # 应该创建了一些规则
    assert len(alert_system.rules) > 0
    assert "cpu_usage_high" in alert_system.rules
    assert "memory_usage_high" in alert_system.rules


def test_get_alert_statistics(alert_system, alert_module_fixture):
    """测试获取告警统计信息"""
    # 添加一些告警
    alert1 = alert_module_fixture.Alert(
        alert_id="alert1",
        rule_id="rule1",
        title="Alert 1",
        message="Message 1",
        level=alert_module_fixture.AlertLevel.WARNING,
        status=alert_module_fixture.AlertStatus.ACTIVE
    )
    alert2 = alert_module_fixture.Alert(
        alert_id="alert2",
        rule_id="rule2",
        title="Alert 2",
        message="Message 2",
        level=alert_module_fixture.AlertLevel.CRITICAL,
        status=alert_module_fixture.AlertStatus.RESOLVED
    )
    alert_system.alerts[alert1.alert_id] = alert1
    alert_system.alerts[alert2.alert_id] = alert2
    
    stats = alert_system.get_stats()
    
    assert stats["total_alerts"] == 2
    assert stats["active_alerts"] == 1
    assert "warning" in stats["alerts_by_level"]
    assert "critical" in stats["alerts_by_level"]
    assert "active" in stats["alerts_by_status"]
    assert "resolved" in stats["alerts_by_status"]


def test_alert_rule_configurator(alert_system, alert_module_fixture):
    """测试AlertRuleConfigurator"""
    configurator = alert_module_fixture.AlertRuleConfigurator(alert_system)
    
    config = {
        "metric": "cpu_usage",
        "threshold": 80,
        "level": "warning"
    }
    
    # 使用内置模板创建规则
    rule = configurator.create_rule_from_template("performance_threshold", config)
    
    assert rule is not None
    assert rule.name == "性能阈值告警"
    assert rule.level == alert_module_fixture.AlertLevel.WARNING


def test_add_alert_rule_with_rule_manager(alert_system, alert_module_fixture, monkeypatch):
    """测试使用规则管理器组件添加规则"""
    mock_rule_manager = Mock()
    alert_system._rule_manager = mock_rule_manager
    
    rule = alert_module_fixture.AlertRule(
        rule_id="test_rule",
        name="Test",
        description="Test",
        condition={"operator": "eq", "field": "x", "value": 1},
        level=alert_module_fixture.AlertLevel.INFO,
        channels=[alert_module_fixture.AlertChannel.CONSOLE]
    )
    
    alert_system.add_alert_rule(rule)
    
    mock_rule_manager.add_rule.assert_called_once_with(rule)
    assert "test_rule" in alert_system.rules


def test_remove_alert_rule_with_rule_manager(alert_system, monkeypatch):
    """测试使用规则管理器组件移除规则"""
    mock_rule_manager = Mock()
    alert_system._rule_manager = mock_rule_manager
    
    # 先添加一个规则
    alert_system.rules["test_rule"] = Mock()
    
    alert_system.remove_alert_rule("test_rule")
    
    mock_rule_manager.remove_rule.assert_called_once_with("test_rule")
    assert "test_rule" not in alert_system.rules
