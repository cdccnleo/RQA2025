#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlertNotifier通知渠道测试
覆盖各种通知渠道发送方法的详细场景和错误处理
"""

import sys
import importlib
from pathlib import Path
import pytest
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
    core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    AlertNotifier = getattr(core_real_time_monitor_module, 'AlertNotifier', None)
    NotificationConfig = getattr(core_real_time_monitor_module, 'NotificationConfig', None)
    create_default_config = getattr(core_real_time_monitor_module, 'create_default_config', None)
    get_notifier = getattr(core_real_time_monitor_module, 'get_notifier', None)
    _notifier_instance = getattr(core_real_time_monitor_module, '_notifier_instance', None)
    start_alert_notifications = getattr(core_real_time_monitor_module, 'start_alert_notifications', None)
    stop_alert_notifications = getattr(core_real_time_monitor_module, 'stop_alert_notifications', None)
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

# 动态导入alert_notifier模块
try:
    alert_notifier_module = importlib.import_module('src.monitoring.alert.alert_notifier')
    _notifier_instance = getattr(alert_notifier_module, '_notifier_instance', None)
except ImportError:
    alert_notifier_module = None
    _notifier_instance = None

# NotificationConfig已在上面导入


class TestAlertNotifierSendNotifications:
    """测试发送所有通知功能"""

    @pytest.fixture
    def config_all_enabled(self):
        """创建所有通知渠道启用的配置"""
        return NotificationConfig(
            email_enabled=True,
            email_smtp_server='smtp.example.com',
            email_smtp_port=587,
            email_username='test@example.com',
            email_password='password',
            email_from='test@example.com',
            email_to=['recipient@example.com'],
            wechat_enabled=True,
            wechat_webhook_url='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test',
            sms_enabled=True,
            sms_api_url='https://api.sms.example.com/send',
            sms_api_key='test_key',
            sms_phone_numbers=['13800138000'],
            slack_enabled=True,
            slack_webhook_url='https://hooks.slack.com/services/test'
        )

    @pytest.fixture
    def sample_alert(self):
        """创建示例告警"""
        return Alert(
            rule_name='test_rule',
            severity='critical',
            current_value=95.0,
            threshold=90.0,
            message='Critical alert',
            timestamp=datetime.now()
        )

    def test_send_notifications_all_channels(self, config_all_enabled, sample_alert):
        """测试发送所有通知（所有渠道）"""
        notifier = AlertNotifier(config_all_enabled)
        
        with patch.object(notifier, '_send_email_notification') as mock_email, \
             patch.object(notifier, '_send_wechat_notification') as mock_wechat, \
             patch.object(notifier, '_send_sms_notification') as mock_sms, \
             patch.object(notifier, '_send_slack_notification') as mock_slack:
            
            notifier._send_notifications(sample_alert)
            
            mock_email.assert_called_once_with(sample_alert)
            mock_wechat.assert_called_once_with(sample_alert)
            mock_sms.assert_called_once_with(sample_alert)
            mock_slack.assert_called_once_with(sample_alert)

    def test_send_notifications_partial_channels(self, sample_alert):
        """测试发送所有通知（部分渠道）"""
        config = NotificationConfig(
            email_enabled=True,
            email_smtp_server='smtp.example.com',
            email_smtp_port=587,
            email_username='test@example.com',
            email_password='password',
            email_from='test@example.com',
            email_to=['recipient@example.com'],
            wechat_enabled=False,
            sms_enabled=False,
            slack_enabled=True,
            slack_webhook_url='https://hooks.slack.com/services/test'
        )
        notifier = AlertNotifier(config)
        
        with patch.object(notifier, '_send_email_notification') as mock_email, \
             patch.object(notifier, '_send_wechat_notification') as mock_wechat, \
             patch.object(notifier, '_send_sms_notification') as mock_sms, \
             patch.object(notifier, '_send_slack_notification') as mock_slack:
            
            notifier._send_notifications(sample_alert)
            
            mock_email.assert_called_once()
            mock_wechat.assert_not_called()
            mock_sms.assert_not_called()
            mock_slack.assert_called_once()

    def test_send_notifications_exception_handling(self, config_all_enabled, sample_alert):
        """测试发送所有通知（异常处理）"""
        notifier = AlertNotifier(config_all_enabled)
        
        with patch.object(notifier, '_send_email_notification', side_effect=Exception("Email error")):
            # 应该优雅处理错误，继续发送其他渠道的通知
            with patch.object(notifier, '_send_wechat_notification') as mock_wechat:
                notifier._send_notifications(sample_alert)
                
                # 其他渠道应该仍然被调用
                mock_wechat.assert_called_once()



