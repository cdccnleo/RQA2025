"""
测试通知管理组件

验证NotificationManager类的功能，包括渠道注册、通知发送等
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

import src.infrastructure.resource.monitoring.notification_manager_component as notification_module
from src.infrastructure.resource.monitoring.notification_manager_component import (
    NotificationChannelRegistry,
    EmailNotificationHandler,
    WebhookNotificationHandler,
    WechatNotificationHandler,
    NotificationCoordinator,
    NotificationManager,
)
from src.infrastructure.resource.models.alert_dataclasses import Alert
from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
from src.infrastructure.resource.config.config_classes import AlertConfig


class DummyLogger:
    def __init__(self, name: str = "logger"):
        self.name = name
        self.infos: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log_info(self, message: str, **kwargs):
        self.infos.append(message)

    def log_error(self, message: str, **kwargs):
        self.errors.append(message)

    def log_warning(self, message: str, **kwargs):
        self.warnings.append(message)

    # 兼容直接调用
    def info(self, message: str, **kwargs):
        self.log_info(message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log_error(message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log_warning(message, **kwargs)


class DummyErrorHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def handle_error(self, error: Exception, context: Optional[Any] = None):
        self.calls.append({"error": error, "context": context})


@pytest.fixture(autouse=True)
def patch_logging(monkeypatch):
    created: List[DummyLogger] = []

    def factory(name: Optional[str] = None):
        logger = DummyLogger(name or "logger")
        created.append(logger)
        return logger

    monkeypatch.setattr(notification_module, "StandardLogger", factory)
    monkeypatch.setattr(notification_module, "BaseErrorHandler", lambda: DummyErrorHandler())
    return created


@pytest.fixture
def sample_alert():
    return Alert(
        id="alert-1",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.CRITICAL,
        message="系统出现严重错误",
        details={"system": "core"},
        timestamp=datetime.now(),
        source="unit-test",
    )


def test_channel_registry_register_and_unregister():
    logger = DummyLogger("registry")
    registry = NotificationChannelRegistry(logger=logger)
    handler = lambda alert, config: True
    config = {"smtp_server": "localhost"}

    registry.register_channel("email", handler, config)

    assert registry.get_channel("email") is handler
    assert registry.get_channel_config("email") == config
    assert registry.is_channel_available("email") is True
    assert "email" in registry.list_channels()

    assert registry.unregister_channel("email") is True
    assert registry.is_channel_available("email") is False
    assert registry.unregister_channel("email") is False


def test_email_notification_handler_send_success_and_failure(monkeypatch, sample_alert):
    logger = DummyLogger("email")
    handler = EmailNotificationHandler(logger=logger)
    captured: Dict[str, Any] = {}

    def fake_send(self, msg, smtp_config):
        captured["msg"] = msg
        captured["config"] = smtp_config

    monkeypatch.setattr(EmailNotificationHandler, "_send_via_smtp", fake_send)

    config = {
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "from_email": "monitor@example.com",
        "to_email": "ops@example.com",
        "use_tls": True,
    }

    assert handler.send_email_notification(sample_alert, config) is True
    assert "邮件通知发送成功" in logger.infos[-1]
    assert captured["config"]["server"] == "smtp.example.com"

    def failing_send(self, msg, smtp_config):
        raise RuntimeError("smtp failure")

    monkeypatch.setattr(EmailNotificationHandler, "_send_via_smtp", failing_send)
    assert handler.send_email_notification(sample_alert, config) is False
    assert any("发送失败" in msg for msg in logger.errors)


def test_email_notification_handler_builds_from_alert_config(monkeypatch, sample_alert):
    handler = EmailNotificationHandler(logger=DummyLogger("email"))
    routing_rules = [
        {
            "from_email": "noreply@example.com",
            "to_email": "team@example.com",
            "smtp_config": {"server": "smtp.alt", "port": 25},
        }
    ]
    alert_config = AlertConfig(routing_rules=routing_rules)
    captured: List[Any] = []

    def fake_send(self, msg, smtp_config):
        captured.append((msg, smtp_config))

    monkeypatch.setattr(EmailNotificationHandler, "_send_via_smtp", fake_send)

    assert handler.send_email_notification_with_config(sample_alert, alert_config) is True
    msg, smtp_config = captured[0]
    assert msg["From"] == "noreply@example.com"
    assert msg["To"] == "team@example.com"
    assert smtp_config["server"] == "smtp.alt"


def test_webhook_notification_handler(monkeypatch, sample_alert):
    logger = DummyLogger("webhook")
    handler = WebhookNotificationHandler(logger=logger)
    calls: List[Dict[str, Any]] = []

    class DummyResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return DummyResponse(200)

    monkeypatch.setattr(notification_module.requests, "post", fake_post)
    config = {"webhook_url": "https://api.example.com"}

    assert handler.send_webhook_notification(sample_alert, config) is True
    assert calls[0]["json"]["alert_id"] == sample_alert.id

    calls.clear()
    assert handler.send_webhook_notification(sample_alert, {}) is False
    assert any("未配置" in msg for msg in logger.errors)

    def failing_post(url, json=None, timeout=None):
        return DummyResponse(500)

    monster_logger = DummyLogger("webhook2")
    handler2 = WebhookNotificationHandler(logger=monster_logger)
    monkeypatch.setattr(notification_module.requests, "post", failing_post)
    assert handler2.send_webhook_notification(sample_alert, config) is False
    assert any("状态码" in msg for msg in monster_logger.errors)


def test_wechat_notification_handler(monkeypatch, sample_alert):
    logger = DummyLogger("wechat")
    handler = WechatNotificationHandler(logger=logger)
    calls: List[Dict[str, Any]] = []

    class DummyResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        return DummyResponse(200)

    monkeypatch.setattr(notification_module.requests, "post", fake_post)
    config = {"webhook_url": "https://qyapi.example.com"}

    assert handler.send_wechat_notification(sample_alert, config) is True
    assert calls[0]["json"]["markdown"]["content"].startswith("🚨 测试监控告警")

    assert handler.send_wechat_notification(sample_alert, {}) is False
    assert any("未配置" in msg for msg in logger.errors)

    def raising_post(url, json=None, timeout=None):
        raise RuntimeError("network")

    monkeypatch.setattr(notification_module.requests, "post", raising_post)
    assert handler.send_wechat_notification(sample_alert, config) is False
    assert any("失败" in msg for msg in logger.errors)


def test_notification_coordinator_send_and_status(sample_alert):
    registry = NotificationChannelRegistry(logger=DummyLogger("registry"))
    coordinator = NotificationCoordinator(registry, logger=DummyLogger("coord"))

    handled: List[str] = []

    def email_handler(alert, config):
        handled.append("email")
        return True

    registry.register_channel("email", email_handler, {"smtp_server": "smtp"})
    registry.register_channel("webhook", lambda a, c: True, {"webhook_url": "https://x"})
    registry.register_channel("wechat", lambda a, c: True, {"webhook_url": ""})
    registry.register_channel("custom", lambda a, c: False)

    results = coordinator.send_notification(sample_alert, ["email", "missing"])
    assert results["email"] is True
    assert results["missing"] is False
    assert "未注册" in coordinator.logger.warnings[-1]
    assert handled == ["email"]

    status = coordinator.get_channel_status()
    assert status["email"] is True
    assert status["webhook"] is True
    assert status["wechat"] is False
    assert status["custom"] is True


def test_notification_manager_defaults_and_configuration(monkeypatch, sample_alert):
    monkeypatch.setattr(
        notification_module.EmailNotificationHandler,
        "send_email_notification",
        lambda self, alert, config: True,
    )
    monkeypatch.setattr(
        notification_module.WebhookNotificationHandler,
        "send_webhook_notification",
        lambda self, alert, config: True,
    )
    monkeypatch.setattr(
        notification_module.WechatNotificationHandler,
        "send_wechat_notification",
        lambda self, alert, config: True,
    )

    config = {
        "email": {"smtp_server": "smtp.example.com"},
        "webhook": {"webhook_url": "https://hook"},
        "wechat": {"webhook_url": "https://wechat"},
    }

    manager = NotificationManager(config=config)

    default_channels = manager.channel_registry.list_channels()
    assert {"email", "webhook", "wechat"}.issubset(default_channels)
    assert manager.notification_config["email"]["smtp_server"] == "smtp.example.com"

    result = manager.send_notification(sample_alert, ["email", "wechat"])
    assert result == {"email": True, "wechat": True}

    assert manager.send_email_notification(sample_alert, config["email"]) is True
    assert manager.send_webhook_notification(sample_alert, config["webhook"]) is True
    assert manager.send_wechat_notification(sample_alert, config["wechat"]) is True

    with pytest.raises(ValueError):
        manager.configure_channel("unknown", {})

    manager.configure_channel("email", {"smtp_server": "smtp.alt"})
    assert manager.notification_config["email"]["smtp_server"] == "smtp.alt"
