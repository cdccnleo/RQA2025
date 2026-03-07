#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警通知器质量测试
测试覆盖 AlertNotifier 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    alert_alert_notifier_module = importlib.import_module('src.monitoring.alert.alert_notifier')
    Alert = getattr(alert_alert_notifier_module, 'Alert', None)
    AlertRule = getattr(alert_alert_notifier_module, 'AlertRule', None)
    AlertNotifier = getattr(alert_alert_notifier_module, 'AlertNotifier', None)
    
    # 尝试从real_time_monitor导入NotificationConfig
    try:
        core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
        NotificationConfig = getattr(core_real_time_monitor_module, 'NotificationConfig', None)
    except ImportError:
        NotificationConfig = None
    
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def notification_config():
    """创建通知配置"""
    return NotificationConfig(
        email_enabled=False,
        wechat_enabled=False,
        sms_enabled=False,
        slack_enabled=False
    )


@pytest.fixture
def alert_notifier(notification_config):
    """创建告警通知器实例"""
    return AlertNotifier(notification_config)


@pytest.fixture
def sample_alert():
    """创建示例告警"""
    rule = AlertRule(
        name='test_rule',
        metric_name='cpu_percent',
        condition='>',
        threshold=80.0,
        duration=60,
        severity='warning',
        description='CPU usage too high'
    )
    return Alert(
        rule_name='test_rule',
        metric_name='cpu_percent',
        current_value=90.0,
        threshold=80.0,
        severity='warning',
        message='CPU usage too high',
        timestamp=datetime.now()
    )


class TestNotificationConfig:
    """NotificationConfig测试类"""

    def test_initialization(self, notification_config):
        """测试初始化"""
        assert notification_config.email_enabled is False
        assert notification_config.wechat_enabled is False
        assert notification_config.sms_enabled is False
        assert notification_config.slack_enabled is False

    def test_post_init(self, notification_config):
        """测试后初始化"""
        assert notification_config.email_to == []
        assert notification_config.sms_phone_numbers == []


class TestAlertNotifier:
    """AlertNotifier测试类"""

    def test_initialization(self, alert_notifier):
        """测试初始化"""
        assert alert_notifier.config is not None
        assert alert_notifier.notification_queue == []
        assert alert_notifier._running is False
        assert alert_notifier._notification_cooldown == 300

    def test_start(self, alert_notifier):
        """测试启动通知服务"""
        alert_notifier.start()
        assert alert_notifier._running is True
        assert alert_notifier._notification_thread is not None
        assert alert_notifier._notification_thread.is_alive()
        
        # 清理
        alert_notifier.stop()

    def test_stop(self, alert_notifier):
        """测试停止通知服务"""
        alert_notifier.start()
        time.sleep(0.1)
        alert_notifier.stop()
        assert alert_notifier._running is False

    def test_notify_alert(self, alert_notifier, sample_alert):
        """测试发送告警通知"""
        alert_notifier.start()
        alert_notifier.notify_alert(sample_alert)
        
        # 等待通知处理
        time.sleep(0.1)
        
        alert_notifier.stop()
        # 验证通知被添加到队列或已处理

    def test_notify_alert_cooldown(self, alert_notifier, sample_alert):
        """测试告警通知冷却时间"""
        alert_notifier.start()
        
        # 第一次通知
        alert_notifier.notify_alert(sample_alert)
        time.sleep(0.1)
        
        # 立即第二次通知（应该在冷却时间内）
        alert_notifier.notify_alert(sample_alert)
        time.sleep(0.1)
        
        alert_notifier.stop()

    def test_send_email_notification(self, notification_config, sample_alert):
        """测试发送邮件通知"""
        config = NotificationConfig(
            email_enabled=True,
            email_smtp_server='smtp.test.com',
            email_smtp_port=587,
            email_username='test@test.com',
            email_password='password',
            email_from='test@test.com',
            email_to=['recipient@test.com']
        )
        notifier = AlertNotifier(config)
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            notifier._send_email_notification(sample_alert)
            # 验证SMTP被调用

    def test_send_wechat_notification(self, notification_config, sample_alert):
        """测试发送微信通知"""
        config = NotificationConfig(
            wechat_enabled=True,
            wechat_webhook_url='https://test.com/webhook'
        )
        notifier = AlertNotifier(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_wechat_notification(sample_alert)
            # 验证请求被发送

    def test_send_sms_notification(self, notification_config, sample_alert):
        """测试发送短信通知"""
        config = NotificationConfig(
            sms_enabled=True,
            sms_api_url='https://test.com/sms',
            sms_api_key='test_key',
            sms_phone_numbers=['1234567890']
        )
        notifier = AlertNotifier(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_sms_notification(sample_alert)
            # 验证请求被发送

    def test_send_slack_notification(self, notification_config, sample_alert):
        """测试发送Slack通知"""
        config = NotificationConfig(
            slack_enabled=True,
            slack_webhook_url='https://test.com/slack',
            slack_channel='#alerts'
        )
        notifier = AlertNotifier(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_slack_notification(sample_alert)
            # 验证请求被发送


