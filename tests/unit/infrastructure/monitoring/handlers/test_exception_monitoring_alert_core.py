#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 异常监控告警系统核心功能

测试 handlers/exception_monitoring_alert.py 中的所有类和方法
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.handlers import exception_monitoring_alert
    return exception_monitoring_alert


@pytest.fixture
def monitor(module):
    """创建异常监控器实例"""
    return module.ExceptionMonitor()


class TestAlertLevel:
    """测试告警级别枚举"""

    def test_alert_level_values(self, module):
        """测试告警级别值"""
        assert module.AlertLevel.INFO.value == "info"
        assert module.AlertLevel.WARNING.value == "warning"
        assert module.AlertLevel.ERROR.value == "error"
        assert module.AlertLevel.CRITICAL.value == "critical"


class TestAlertChannel:
    """测试告警渠道枚举"""

    def test_alert_channel_values(self, module):
        """测试告警渠道值"""
        assert module.AlertChannel.EMAIL.value == "email"
        assert module.AlertChannel.WEBHOOK.value == "webhook"
        assert module.AlertChannel.LOG.value == "log"
        assert module.AlertChannel.CONSOLE.value == "console"
        assert module.AlertChannel.SMS.value == "sms"


class TestAlertRule:
    """测试告警规则"""

    def test_alert_rule_initialization(self, module):
        """测试告警规则初始化"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG]
        )

        assert rule.name == "test_rule"
        assert rule.level == module.AlertLevel.WARNING
        assert rule.channels == [module.AlertChannel.LOG]
        assert rule.cooldown == 300
        assert rule.enabled is True
        assert rule.last_triggered == 0.0

    def test_alert_rule_should_trigger_enabled(self, module):
        """测试告警规则触发检查 - 启用状态"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0  # 无冷却时间
        )

        assert rule.should_trigger({"test": "data"}) is True

    def test_alert_rule_should_trigger_disabled(self, module):
        """测试告警规则触发检查 - 禁用状态"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            enabled=False
        )

        assert rule.should_trigger({"test": "data"}) is False

    def test_alert_rule_should_trigger_cooldown(self, module):
        """测试告警规则触发检查 - 冷却时间"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=100
        )
        rule.last_triggered = time.time()

        assert rule.should_trigger({"test": "data"}) is False

    def test_alert_rule_should_trigger_condition_false(self, module):
        """测试告警规则触发检查 - 条件不满足"""
        def condition(context):
            return False

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0
        )

        assert rule.should_trigger({"test": "data"}) is False

    def test_alert_rule_should_trigger_condition_exception(self, module, monkeypatch):
        """测试告警规则触发检查 - 条件异常"""
        def condition(context):
            raise ValueError("Test error")

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0
        )

        # Mock logger.error 来捕获错误日志
        error_logs = []
        def mock_error(msg):
            error_logs.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)
        assert rule.should_trigger({"test": "data"}) is False
        assert len(error_logs) > 0

    def test_alert_rule_trigger(self, module):
        """测试告警规则触发"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0
        )

        initial_time = time.time()
        rule.trigger({"test": "data"})
        assert rule.last_triggered >= initial_time


class TestAlertMessage:
    """测试告警消息"""

    def test_alert_message_initialization(self, module):
        """测试告警消息初始化"""
        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert alert.level == module.AlertLevel.WARNING
        assert alert.timestamp is not None
        assert alert.context == {}
        assert alert.rule_name == ""

    def test_alert_message_with_timestamp(self, module):
        """测试告警消息带时间戳"""
        custom_timestamp = time.time()
        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING,
            timestamp=custom_timestamp
        )

        assert alert.timestamp == custom_timestamp


class TestAlertChannelConfig:
    """测试告警渠道配置"""

    def test_alert_channel_config_initialization(self, module):
        """测试告警渠道配置初始化"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.LOG,
            config={"enabled": True}
        )

        assert config.channel_type == module.AlertChannel.LOG
        assert config.config == {"enabled": True}

    def test_alert_channel_config_send_alert_log(self, module, monkeypatch):
        """测试告警渠道配置发送告警 - LOG"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.LOG,
            config={"enabled": True}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        logs = []
        def mock_log(level, msg, *args, **kwargs):
            logs.append((level, msg))
        
        monkeypatch.setattr(module.logger, "log", mock_log)
        result = config.send_alert(alert)
        assert result is True
        assert len(logs) > 0

    def test_alert_channel_config_send_alert_console(self, module, monkeypatch):
        """测试告警渠道配置发送告警 - CONSOLE"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.CONSOLE,
            config={"enabled": True}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock print 来捕获输出
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        result = config.send_alert(alert)
        assert result is True
        assert len(prints) > 0


class TestExceptionMonitor:
    """测试异常监控器"""

    def test_exception_monitor_initialization(self, monitor):
        """测试异常监控器初始化"""
        assert monitor.rules == []
        assert monitor.channels == {}
        assert len(monitor.exception_history) == 0
        assert monitor.stats == {}

    def test_add_rule(self, monitor, module):
        """测试添加告警规则"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG]
        )

        monitor.add_rule(rule)
        assert len(monitor.rules) == 1
        assert monitor.rules[0] == rule

    def test_remove_rule(self, monitor, module):
        """测试移除告警规则"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG]
        )

        monitor.add_rule(rule)
        monitor.remove_rule("test_rule")
        assert len(monitor.rules) == 0

    def test_configure_channel(self, monitor, module):
        """测试配置告警渠道"""
        monitor.configure_channel(
            module.AlertChannel.LOG,
            {"enabled": True}
        )

        assert module.AlertChannel.LOG in monitor.channels
        assert monitor.channels[module.AlertChannel.LOG].channel_type == module.AlertChannel.LOG

    def test_report_exception(self, monitor):
        """测试报告异常"""
        exception_context = {
            "type": "ValueError",
            "message": "Test error"
        }

        monitor.report_exception(exception_context)

        assert len(monitor.exception_history) == 1
        assert "ValueError" in monitor.stats
        assert monitor.stats["ValueError"] == 1

    def test_report_exception_triggers_alert(self, monitor, module, monkeypatch):
        """测试报告异常触发告警"""
        def condition(context):
            return context.get("type") == "ValueError"

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0
        )

        monitor.add_rule(rule)
        monitor.configure_channel(
            module.AlertChannel.LOG,
            {"enabled": True}
        )

        # Mock send_alert 来捕获调用
        sent_alerts = []
        def mock_send_alert(alert, channels):
            sent_alerts.append((alert, channels))
            return True
        
        monkeypatch.setattr(monitor, "_send_alert", mock_send_alert)

        exception_context = {
            "type": "ValueError",
            "message": "Test error"
        }

        monitor.report_exception(exception_context)

        assert len(sent_alerts) == 1
        assert sent_alerts[0][0].title.startswith("异常告警")

    def test_get_stats(self, monitor):
        """测试获取统计信息"""
        exception_context = {
            "type": "ValueError",
            "message": "Test error"
        }

        monitor.report_exception(exception_context)
        stats = monitor.get_stats()

        assert stats["total_exceptions"] == 1
        assert "ValueError" in stats["exception_types"]
        assert stats["exception_types"]["ValueError"] == 1
        assert stats["active_rules"] == 0
        assert stats["configured_channels"] == []

    def test_get_recent_exceptions(self, monitor):
        """测试获取最近的异常"""
        for i in range(5):
            monitor.report_exception({
                "type": f"Error{i}",
                "message": f"Test error {i}"
            })

        recent = monitor.get_recent_exceptions(limit=3)
        assert len(recent) == 3

    def test_start_monitoring(self, monitor, monkeypatch):
        """测试启动监控"""
        # Mock threading.Thread 来避免真实线程创建
        created_threads = []
        original_thread = threading.Thread

        def mock_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        monkeypatch.setattr(threading, "Thread", mock_thread)
        monitor.start_monitoring()

        assert monitor._monitoring_thread is not None
        assert len(created_threads) == 1

    def test_shutdown(self, monitor, monkeypatch):
        """测试关闭监控"""
        # Mock threading.Thread 和 Event
        mock_event = MagicMock()
        mock_event.is_set.return_value = False
        mock_event.set = MagicMock()

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread.join = MagicMock()

        monitor._shutdown_event = mock_event
        monitor._monitoring_thread = mock_thread

        monitor.shutdown()

        mock_event.set.assert_called_once()

    def test_perform_health_check_high_frequency(self, monitor, monkeypatch):
        """测试健康检查 - 高频异常"""
        # 添加多个异常
        for i in range(15):
            monitor.report_exception({
                "type": "TestError",
                "message": f"Error {i}"
            })

        # Mock report_exception 来捕获健康检查触发的告警
        health_alerts = []
        original_report = monitor.report_exception

        def mock_report(context):
            if context.get("type") == "high_exception_rate":
                health_alerts.append(context)
            return original_report(context)

        monkeypatch.setattr(monitor, "report_exception", mock_report)
        import time as time_module
        monkeypatch.setattr(time_module, "sleep", lambda *_: None)

        monitor._perform_health_check()

        # 由于我们添加了15个异常，应该触发高频告警
        assert len(health_alerts) > 0


class TestPredefinedRules:
    """测试预定义告警规则"""

    def test_create_high_frequency_rule(self, module):
        """测试创建高频异常告警规则"""
        rule = module.create_high_frequency_rule(
            name="test_high_freq",
            threshold=10,
            time_window=300
        )

        assert rule.name == "test_high_freq"
        assert rule.level == module.AlertLevel.WARNING
        assert module.AlertChannel.LOG in rule.channels
        assert rule.cooldown == 600

        # 测试条件函数
        assert rule.should_trigger({"type": "high_frequency_error"}) is True
        assert rule.should_trigger({"type": "normal_error"}) is False

    def test_create_critical_exception_rule(self, module):
        """测试创建严重异常告警规则"""
        rule = module.create_critical_exception_rule(name="test_critical")

        assert rule.name == "test_critical"
        assert rule.level == module.AlertLevel.CRITICAL
        assert module.AlertChannel.EMAIL in rule.channels
        assert rule.cooldown == 300

        # 测试条件函数
        assert rule.should_trigger({"severity": "critical"}) is True
        assert rule.should_trigger({"severity": "error"}) is True
        assert rule.should_trigger({"severity": "warning"}) is False

    def test_create_database_exception_rule(self, module):
        """测试创建数据库异常告警规则"""
        rule = module.create_database_exception_rule(name="test_db")

        assert rule.name == "test_db"
        assert rule.level == module.AlertLevel.ERROR

        # 测试条件函数
        assert rule.should_trigger({"type": "database_error"}) is True
        assert rule.should_trigger({"type": "connection_timeout"}) is True
        assert rule.should_trigger({"type": "normal_error"}) is False


class TestAlertChannelConfigAdvanced:
    """测试告警渠道配置高级功能"""

    def test_send_alert_email_success(self, module, monkeypatch):
        """测试发送邮件告警 - 成功"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.EMAIL,
            config={
                "smtp": {
                    "host": "smtp.example.com",
                    "port": 587,
                    "from": "test@example.com",
                    "username": "user",
                    "password": "pass",
                    "tls": True
                },
                "recipients": ["admin@example.com"]
            }
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock smtplib.SMTP
        mock_server = MagicMock()
        mock_smtp = MagicMock(return_value=mock_server)
        monkeypatch.setattr(module.smtplib, "SMTP", mock_smtp)

        result = config.send_alert(alert)
        assert result is True
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

    def test_send_alert_email_incomplete_config(self, module, monkeypatch):
        """测试发送邮件告警 - 配置不完整"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.EMAIL,
            config={}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)
        result = config.send_alert(alert)
        assert result is False
        assert len(warnings) > 0

    def test_send_alert_email_exception(self, module, monkeypatch):
        """测试发送邮件告警 - 异常处理"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.EMAIL,
            config={
                "smtp": {
                    "host": "smtp.example.com",
                    "port": 587,
                    "from": "test@example.com"
                },
                "recipients": ["admin@example.com"]
            }
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock smtplib.SMTP 抛出异常
        mock_smtp = MagicMock(side_effect=Exception("SMTP error"))
        monkeypatch.setattr(module.smtplib, "SMTP", mock_smtp)

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)
        result = config.send_alert(alert)
        assert result is False
        assert len(errors) > 0

    def test_send_alert_webhook_success(self, module, monkeypatch):
        """测试发送Webhook告警 - 成功"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.WEBHOOK,
            config={
                "url": "https://example.com/webhook",
                "timeout": 10
            }
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post = MagicMock(return_value=mock_response)
        monkeypatch.setattr(module.requests, "post", mock_post)

        result = config.send_alert(alert)
        assert result is True
        mock_post.assert_called_once()

    def test_send_alert_webhook_no_url(self, module, monkeypatch):
        """测试发送Webhook告警 - 无URL配置"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.WEBHOOK,
            config={}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)
        result = config.send_alert(alert)
        assert result is False
        assert len(warnings) > 0

    def test_send_alert_webhook_exception(self, module, monkeypatch):
        """测试发送Webhook告警 - 异常处理"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.WEBHOOK,
            config={
                "url": "https://example.com/webhook"
            }
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock requests.post 抛出异常
        mock_post = MagicMock(side_effect=Exception("Network error"))
        monkeypatch.setattr(module.requests, "post", mock_post)

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)
        result = config.send_alert(alert)
        assert result is False
        assert len(errors) > 0

    def test_send_alert_sms(self, module, monkeypatch):
        """测试发送SMS告警"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.SMS,
            config={}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        logs = []
        def mock_info(msg):
            logs.append(msg)
        
        monkeypatch.setattr(module.logger, "info", mock_info)
        result = config.send_alert(alert)
        assert result is True
        assert len(logs) > 0

    def test_send_alert_unsupported_channel(self, module, monkeypatch):
        """测试发送告警 - 不支持的渠道"""
        # 创建一个模拟的不支持的渠道值
        class MockChannel:
            value = "unsupported_channel"
        
        config = module.AlertChannelConfig(
            channel_type=MockChannel(),
            config={}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)
        result = config.send_alert(alert)
        assert result is False
        assert len(errors) > 0

    def test_send_alert_exception_handling(self, module, monkeypatch):
        """测试发送告警 - 异常处理"""
        config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.LOG,
            config={}
        )

        alert = module.AlertMessage(
            title="Test Alert",
            message="Test message",
            level=module.AlertLevel.WARNING
        )

        # Mock logger.log 抛出异常
        def mock_log(*args, **kwargs):
            raise Exception("Log error")
        
        monkeypatch.setattr(module.logger, "log", mock_log)

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)
        result = config.send_alert(alert)
        assert result is False
        assert len(errors) > 0


class TestExceptionMonitorAdvanced:
    """测试异常监控器高级功能"""

    def test_send_alert_with_channels(self, monitor, module, monkeypatch):
        """测试通过多个渠道发送告警"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG, module.AlertChannel.CONSOLE],
            cooldown=0
        )

        monitor.add_rule(rule)
        monitor.configure_channel(
            module.AlertChannel.LOG,
            {"enabled": True}
        )
        monitor.configure_channel(
            module.AlertChannel.CONSOLE,
            {"enabled": True}
        )

        sent_channels = []
        def mock_send_alert(alert, channels):
            sent_channels.extend(channels)
            return True
        
        monkeypatch.setattr(monitor, "_send_alert", mock_send_alert)

        exception_context = {
            "type": "TestError",
            "message": "Test error"
        }

        monitor.report_exception(exception_context)

        assert len(sent_channels) == 2
        assert module.AlertChannel.LOG in sent_channels
        assert module.AlertChannel.CONSOLE in sent_channels

    def test_send_alert_channel_failure(self, monitor, module, monkeypatch):
        """测试告警渠道发送失败"""
        def condition(context):
            return True

        rule = module.AlertRule(
            name="test_rule",
            condition=condition,
            level=module.AlertLevel.WARNING,
            channels=[module.AlertChannel.LOG],
            cooldown=0
        )

        monitor.add_rule(rule)
        
        # 创建一个配置，但 mock send_alert 返回 False
        channel_config = module.AlertChannelConfig(
            channel_type=module.AlertChannel.LOG,
            config={"enabled": True}
        )
        
        # Mock send_alert 返回 False
        original_send = channel_config.send_alert
        def mock_send_alert(alert):
            return False
        
        monkeypatch.setattr(channel_config, "send_alert", mock_send_alert)
        monitor.channels[module.AlertChannel.LOG] = channel_config

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)

        exception_context = {
            "type": "TestError",
            "message": "Test error"
        }

        monitor.report_exception(exception_context)

        # 验证错误日志被记录
        assert len(errors) > 0

    def test_monitoring_loop_exception_handling(self, monitor, module, monkeypatch):
        """测试监控循环异常处理"""
        # Mock _perform_health_check 抛出异常
        call_count = {"count": 0}
        def mock_health_check():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Health check error")
            # 第二次调用时设置 shutdown event 以便退出循环
            monitor._shutdown_event.set()
        
        monkeypatch.setattr(monitor, "_perform_health_check", mock_health_check)
        monkeypatch.setattr(module.time, "sleep", lambda *_: None)

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)

        # 确保 shutdown event 未设置，以便循环可以执行
        monitor._shutdown_event.clear()
        monitor._monitoring_loop()

        # 验证异常被记录
        assert len(errors) > 0

    def test_shutdown_with_thread(self, monitor, monkeypatch):
        """测试关闭监控 - 有活动线程"""
        # Mock threading.Thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = MagicMock()

        monitor._monitoring_thread = mock_thread
        monitor._shutdown_event = MagicMock()

        monitor.shutdown()

        mock_thread.join.assert_called_once_with(timeout=5)

    def test_perform_health_check_low_frequency(self, monitor):
        """测试健康检查 - 低频异常（不触发告警）"""
        # 添加少量异常
        for i in range(5):
            monitor.report_exception({
                "type": "TestError",
                "message": f"Error {i}"
            })

        # Mock report_exception 来捕获健康检查触发的告警
        health_alerts = []
        original_report = monitor.report_exception

        def mock_report(context):
            if context.get("type") == "high_exception_rate":
                health_alerts.append(context)
            return original_report(context)

        from unittest.mock import patch
        with patch.object(monitor, "report_exception", side_effect=mock_report):
            monitor._perform_health_check()

        # 由于只有5个异常，不应该触发高频告警
        assert len(health_alerts) == 0

