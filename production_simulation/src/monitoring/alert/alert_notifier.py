# -*- coding: utf-8 -*-
"""
RQA2025 告警通知系统

支持多种通知渠道：邮件、微信、短信等
"""

import logging
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from ..core.real_time_monitor import Alert

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """通知配置"""
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = None

    wechat_enabled: bool = False
    wechat_webhook_url: str = ""
    wechat_corp_id: str = ""
    wechat_corp_secret: str = ""
    wechat_agent_id: int = 0

    sms_enabled: bool = False
    sms_api_url: str = ""
    sms_api_key: str = ""
    sms_phone_numbers: List[str] = None

    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = ""

    def __post_init__(self):
        if self.email_to is None:
            self.email_to = []
        if self.sms_phone_numbers is None:
            self.sms_phone_numbers = []


class AlertNotifier:
    """告警通知器"""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.notification_queue = []
        self._running = False
        self._notification_thread = None

        # 通知频率控制（避免告警轰炸）
        self._last_notification_time = {}
        self._notification_cooldown = 300  # 5分钟冷却时间

    def start(self):
        """启动通知服务"""
        if self._running:
            return

        self._running = True
        self._notification_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True
        )
        self._notification_thread.start()
        logger.info("Alert notifier started")

    def stop(self):
        """停止通知服务"""
        self._running = False
        if self._notification_thread:
            self._notification_thread.join(timeout=5)
        logger.info("Alert notifier stopped")

    def notify_alert(self, alert: Alert):
        """发送告警通知"""
        # 检查频率限制
        alert_key = f"{alert.rule_name}_{alert.severity}"
        last_time = self._last_notification_time.get(alert_key, 0)
        current_time = time.time()

        if current_time - last_time < self._notification_cooldown:
            logger.debug(f"Skipping notification for {alert_key} due to cooldown")
            return

        self._last_notification_time[alert_key] = current_time
        self.notification_queue.append(alert)

    def _notification_worker(self):
        """通知工作线程"""
        while self._running:
            try:
                if self.notification_queue:
                    alert = self.notification_queue.pop(0)
                    self._send_notifications(alert)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                time.sleep(5)

    def _send_notifications(self, alert: Alert):
        """发送所有配置的通知"""
        try:
            # 邮件通知
            if self.config.email_enabled:
                self._send_email_notification(alert)

            # 微信通知
            if self.config.wechat_enabled:
                self._send_wechat_notification(alert)

            # 短信通知
            if self.config.sms_enabled:
                self._send_sms_notification(alert)

            # Slack通知
            if self.config.slack_enabled:
                self._send_slack_notification(alert)

        except Exception as e:
            logger.error(f"Failed to send notifications for alert {alert.rule_name}: {e}")

    def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        try:
            if not all([
                self.config.email_smtp_server,
                self.config.email_username,
                self.config.email_password,
                self.config.email_from,
                self.config.email_to
            ]):
                logger.warning("Email configuration incomplete")
                return

            # 创建邮件内容
            subject = f"RQA2025 告警: {alert.rule_name} ({alert.severity.upper()})"

            body = f"""
RQA2025 系统告警通知

告警规则: {alert.rule_name}
严重程度: {alert.severity.upper()}
当前值: {alert.current_value}
阈值: {alert.threshold}
消息: {alert.message}

发生时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

请及时处理！

RQA2025 监控系统
"""

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 发送邮件
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            text = msg.as_string()
            server.sendmail(self.config.email_from, self.config.email_to, text)
            server.quit()

            logger.info(f"Email notification sent for alert: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _send_wechat_notification(self, alert: Alert):
        """发送微信通知"""
        try:
            if self.config.wechat_webhook_url:
                # 使用企业微信Webhook
                message = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": f"""# RQA2025 系统告警\n\n**告警规则**: {alert.rule_name}\n**严重程度**: {alert.severity.upper()}\n**当前值**: {alert.current_value}\n**阈值**: {alert.threshold}\n**消息**: {alert.message}\n\n**发生时间**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n请及时处理！"""
                    }
                }

                response = requests.post(
                    self.config.wechat_webhook_url,
                    json=message,
                    timeout=10
                )
                response.raise_for_status()

                logger.info(f"WeChat webhook notification sent for alert: {alert.rule_name}")

            elif all([
                self.config.wechat_corp_id,
                self.config.wechat_corp_secret,
                self.config.wechat_agent_id
            ]):
                # 使用企业微信API (需要access_token)
                logger.warning("Enterprise WeChat API not implemented yet")

            else:
                logger.warning("WeChat configuration incomplete")

        except Exception as e:
            logger.error(f"Failed to send WeChat notification: {e}")

    def _send_sms_notification(self, alert: Alert):
        """发送短信通知"""
        try:
            if not all([
                self.config.sms_api_url,
                self.config.sms_api_key,
                self.config.sms_phone_numbers
            ]):
                logger.warning("SMS configuration incomplete")
                return

            # 这里实现具体的短信API调用
            # 不同的短信服务商API不同，这里提供一个通用模板

            message = f"RQA2025告警: {alert.rule_name}({alert.severity.upper()}) - {alert.message}"

            # 示例API调用（需要根据实际服务商调整）
            payload = {
                'api_key': self.config.sms_api_key,
                'message': message,
                'recipients': self.config.sms_phone_numbers
            }

            response = requests.post(
                self.config.sms_api_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"SMS notification sent for alert: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")

    def _send_slack_notification(self, alert: Alert):
        """发送Slack通知"""
        try:
            if not self.config.slack_webhook_url:
                logger.warning("Slack webhook URL not configured")
                return

            severity_colors = {
                'info': 'good',
                'warning': 'warning',
                'error': 'danger',
                'critical': '#FF0000'
            }

            message = {
                "channel": self.config.slack_channel or "#alerts",
                "attachments": [{
                    "color": severity_colors.get(alert.severity, 'warning'),
                    "title": f"RQA2025 Alert: {alert.rule_name}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Message", "value": alert.message, "short": False},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": False}
                    ]
                }]
            }

            response = requests.post(
                self.config.slack_webhook_url,
                json=message,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"Slack notification sent for alert: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def get_notification_stats(self) -> Dict[str, Any]:
        """获取通知统计"""
        return {
            'queue_size': len(self.notification_queue),
            'last_notifications': {
                key: datetime.fromtimestamp(ts).isoformat()
                for key, ts in self._last_notification_time.items()
            },
            'cooldown_seconds': self._notification_cooldown
        }


# 默认通知器配置
def create_default_config() -> NotificationConfig:
    """创建默认通知配置"""
    return NotificationConfig(
        # 示例配置，请根据实际情况修改
        email_enabled=False,  # 默认禁用，需要配置后启用
        wechat_enabled=False,
        sms_enabled=False,
        slack_enabled=False
    )


# 全局通知器实例
_notifier_instance = None


def get_notifier(config: Optional[NotificationConfig] = None) -> AlertNotifier:
    """获取全局通知器实例"""
    global _notifier_instance
    if _notifier_instance is None:
        if config is None:
            config = create_default_config()
        _notifier_instance = AlertNotifier(config)
    return _notifier_instance


def start_alert_notifications(config: Optional[NotificationConfig] = None):
    """启动告警通知服务"""
    notifier = get_notifier(config)
    notifier.start()


def stop_alert_notifications():
    """停止告警通知服务"""
    global _notifier_instance
    if _notifier_instance:
        _notifier_instance.stop()
        _notifier_instance = None
