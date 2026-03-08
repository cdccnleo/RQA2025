"""
调度器告警模块

提供任务执行告警功能，支持邮件、Webhook、日志、短信等多种渠道
"""

from .alert_manager import (
    AlertLevel,
    AlertChannel,
    AlertMessage,
    AlertConfig,
    AlertHandler,
    EmailAlertHandler,
    WebhookAlertHandler,
    LogAlertHandler,
    SMSAlertHandler,
    AlertManager,
    get_alert_manager,
    reset_alert_manager
)

__all__ = [
    # 枚举
    'AlertLevel',
    'AlertChannel',
    # 数据类
    'AlertMessage',
    'AlertConfig',
    # 处理器
    'AlertHandler',
    'EmailAlertHandler',
    'WebhookAlertHandler',
    'LogAlertHandler',
    'SMSAlertHandler',
    # 管理器
    'AlertManager',
    'get_alert_manager',
    'reset_alert_manager'
]
