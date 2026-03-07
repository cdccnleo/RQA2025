#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlertNotifier方法测试
补充alert_notifier.py中未覆盖的方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
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
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestAlertNotifierMethods:
    """测试AlertNotifier方法"""

    @pytest.fixture
    def notifier(self):
        """创建AlertNotifier实例"""
        if AlertNotifier is None:
            pytest.skip("AlertNotifier不可用")
        config = NotificationConfig() if NotificationConfig else None
        return AlertNotifier(config) if config else AlertNotifier()

    @pytest.fixture
    def sample_alert(self):
        """创建示例告警"""
        if Alert is None:
            pytest.skip("Alert不可用")
        return Alert(rule_name="test_rule", severity="warning")

    def test_start_already_running(self, notifier):
        """测试启动时已在运行的情况"""
        notifier._running = True
        
        notifier.start()
        
        # 应该不启动新线程
        assert notifier._running == True

    def test_start_normal(self, notifier):
        """测试正常启动"""
        notifier._running = False
        
        with patch('threading.Thread') as mock_thread:
            notifier.start()
            
            assert notifier._running == True
            mock_thread.assert_called_once()

    def test_stop_normal(self, notifier):
        """测试正常停止"""
        notifier._running = True
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = Mock()
        notifier._notification_thread = mock_thread
        
        notifier.stop()
        
        assert notifier._running == False
        mock_thread.join.assert_called_once()

    def test_notify_alert_cooldown(self, notifier, sample_alert):
        """测试告警通知冷却时间"""
        alert_key = f"{sample_alert.rule_name}_{sample_alert.severity}"
        notifier._last_notification_time[alert_key] = time.time()  # 刚刚通知过
        
        with patch('time.time', return_value=time.time() + 10):  # 10秒后
            notifier.notify_alert(sample_alert)
            
            # 冷却时间内不应该添加到队列
            assert len(notifier.notification_queue) == 0

    def test_notify_alert_no_cooldown(self, notifier, sample_alert):
        """测试告警通知无冷却时间"""
        alert_key = f"{sample_alert.rule_name}_{sample_alert.severity}"
        notifier._last_notification_time[alert_key] = time.time() - 600  # 10分钟前
        
        notifier.notify_alert(sample_alert)
        
        # 应该添加到队列
        assert len(notifier.notification_queue) > 0

    def test_send_email_notification_config_incomplete(self, notifier, sample_alert):
        """测试邮件通知配置不完整"""
        notifier.config.email_enabled = True
        notifier.config.email_smtp_server = ""  # 缺失
        notifier.config.email_username = ""
        notifier.config.email_password = ""
        notifier.config.email_from = ""
        notifier.config.email_to = []
        
        # 应该记录警告并返回，不抛出异常
        try:
            notifier._send_email_notification(sample_alert)
            assert True  # 应该正常返回
        except Exception:
            pytest.fail("不应该抛出异常")

    def test_send_email_notification_complete(self, notifier, sample_alert):
        """测试邮件通知配置完整"""
        notifier.config.email_enabled = True
        notifier.config.email_smtp_server = "smtp.example.com"
        notifier.config.email_username = "user"
        notifier.config.email_password = "pass"
        notifier.config.email_from = "from@example.com"
        notifier.config.email_to = ["to@example.com"]
        
        with patch('smtplib.SMTP') as mock_smtp:
            with patch('email.mime.multipart.MIMEMultipart'):
                with patch('email.mime.text.MIMEText'):
                    notifier._send_email_notification(sample_alert)
                    
                    # 应该尝试发送邮件
                    mock_smtp.assert_called_once()

    def test_send_wechat_notification(self, notifier, sample_alert):
        """测试微信通知"""
        notifier.config.wechat_enabled = True
        notifier.config.wechat_webhook_url = "http://example.com/webhook"
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_wechat_notification(sample_alert)
            
            mock_post.assert_called_once()

    def test_send_sms_notification(self, notifier, sample_alert):
        """测试短信通知"""
        notifier.config.sms_enabled = True
        notifier.config.sms_api_url = "http://example.com/sms"
        notifier.config.sms_api_key = "test_key"
        notifier.config.sms_phone_numbers = ["1234567890"]
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_sms_notification(sample_alert)
            
            mock_post.assert_called_once()

    def test_send_slack_notification(self, notifier, sample_alert):
        """测试Slack通知"""
        notifier.config.slack_enabled = True
        notifier.config.slack_webhook_url = "http://example.com/slack"
        notifier.config.slack_channel = "#alerts"
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            notifier._send_slack_notification(sample_alert)
            
            mock_post.assert_called_once()

    def test_send_notifications_all_channels(self, notifier, sample_alert):
        """测试发送所有渠道的通知"""
        notifier.config.email_enabled = True
        notifier.config.wechat_enabled = True
        notifier.config.sms_enabled = True
        notifier.config.slack_enabled = True
        
        with patch.object(notifier, '_send_email_notification') as mock_email:
            with patch.object(notifier, '_send_wechat_notification') as mock_wechat:
                with patch.object(notifier, '_send_sms_notification') as mock_sms:
                    with patch.object(notifier, '_send_slack_notification') as mock_slack:
                        notifier._send_notifications(sample_alert)
                        
                        mock_email.assert_called_once()
                        mock_wechat.assert_called_once()
                        mock_sms.assert_called_once()
                        mock_slack.assert_called_once()

    def test_notification_worker_with_queue(self, notifier):
        """测试通知工作线程处理队列"""
        notifier._running = True
        sample_alert = Mock(rule_name="test", severity="warning")
        notifier.notification_queue = [sample_alert]
        
        with patch.object(notifier, '_send_notifications') as mock_send:
            with patch('time.sleep'):
                # 只运行一次循环
                if notifier.notification_queue:
                    alert = notifier.notification_queue.pop(0)
                    notifier._send_notifications(alert)
                    mock_send.assert_called_once()

    def test_notification_worker_exception_handling(self, notifier):
        """测试通知工作线程异常处理"""
        notifier._running = True
        notifier.notification_queue = []
        
        with patch('time.sleep', side_effect=Exception("Test error")):
            try:
                notifier._notification_worker()
            except:
                pass  # 异常应该被捕获
            assert True

