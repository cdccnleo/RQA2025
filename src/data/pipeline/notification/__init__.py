"""
通知系统模块

提供多渠道通知功能，支持邮件、Webhook和日志通知
"""

from .notification_service import (
    NotificationService,
    NotificationLevel,
    NotificationResult,
    NotificationChannel
)
from .email_channel import EmailNotificationChannel
from .webhook_channel import WebhookNotificationChannel
from .log_channel import LogNotificationChannel

__all__ = [
    'NotificationService',
    'NotificationLevel',
    'NotificationResult',
    'NotificationChannel',
    'EmailNotificationChannel',
    'WebhookNotificationChannel',
    'LogNotificationChannel',
]
