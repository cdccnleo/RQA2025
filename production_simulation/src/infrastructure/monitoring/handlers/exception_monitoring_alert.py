"""
exception_monitoring_alert 模块

提供 exception_monitoring_alert 相关功能和接口。
"""

import json
import logging
import requests

import smtplib
import threading
import time

from collections import deque
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层 - 异常监控和告警机制
实时监控异常情况并发出告警
"""

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """告警渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"
    SMS = "sms"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    channels: List[AlertChannel]
    cooldown: int = 300  # 冷却时间（秒）
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    last_triggered: float = 0.0

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """检查是否应该触发告警"""
        if not self.enabled:
            return False

        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_triggered < self.cooldown:
            return False

        # 检查条件
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(f"告警规则条件检查失败: {self.name} - {e}")
            return False

    def trigger(self, context: Dict[str, Any]):
        """触发告警"""
        self.last_triggered = time.time()
        logger.info(f"告警规则触发: {self.name} (级别: {self.level.value})")


@dataclass
class AlertMessage:
    """告警消息"""
    title: str
    message: str
    level: AlertLevel
    timestamp: float = None
    context: Dict[str, Any] = field(default_factory=dict)
    rule_name: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class AlertChannelConfig:
    """告警渠道配置"""

    def __init__(self, channel_type: AlertChannel, config: Dict[str, Any]):
        self.channel_type = channel_type
        self.config = config

    def send_alert(self, alert: AlertMessage) -> bool:
        """发送告警"""
        try:
            if self.channel_type == AlertChannel.EMAIL:
                return self._send_email_alert(alert)
            elif self.channel_type == AlertChannel.WEBHOOK:
                return self._send_webhook_alert(alert)
            elif self.channel_type == AlertChannel.LOG:
                return self._send_log_alert(alert)
            elif self.channel_type == AlertChannel.CONSOLE:
                return self._send_console_alert(alert)
            elif self.channel_type == AlertChannel.SMS:
                return self._send_sms_alert(alert)
            else:
                logger.error(f"不支持的告警渠道: {self.channel_type.value}")
                return False
        except Exception as e:
            logger.error(f"发送告警失败: {self.channel_type.value} - {e}")
            return False

    def _send_email_alert(self, alert: AlertMessage) -> bool:
        """发送邮件告警"""
        smtp_config = self.config.get('smtp', {})
        recipients = self.config.get('recipients', [])

        if not smtp_config or not recipients:
            logger.warning("邮件告警配置不完整")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from', '')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            body = f"""
异常告警通知

标题: {alert.title}
级别: {alert.level.value}
时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
规则: {alert.rule_name}

详细信息:
{alert.message}

上下文信息:
{json.dumps(alert.context, indent=2, ensure_ascii=False)}
            """

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            server = smtplib.SMTP(smtp_config.get('host', 'localhost'),
                                  smtp_config.get('port', 25))

            if smtp_config.get('tls', False):
                server.starttls()

            if smtp_config.get('username') and smtp_config.get('password'):
                server.login(smtp_config['username'], smtp_config['password'])

            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False

    def _send_webhook_alert(self, alert: AlertMessage) -> bool:
        """发送Webhook告警"""
        webhook_url = self.config.get('url')
        if not webhook_url:
            logger.warning("Webhook URL未配置")
            return False

        try:
            payload = {
                'title': alert.title,
                'message': alert.message,
                'level': alert.level.value,
                'timestamp': alert.timestamp,
                'rule_name': alert.rule_name,
                'context': alert.context
            }

            headers = {'Content-Type': 'application/json'}
            if 'headers' in self.config:
                headers.update(self.config['headers'])

            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=self.config.get('timeout', 10)
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
            return False

    def _send_log_alert(self, alert: AlertMessage) -> bool:
        """发送日志告警"""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.ERROR)

        logger.log(log_level,
                   f"异常告警 - {alert.title}: {alert.message}",
                   extra={
                       'alert_level': alert.level.value,
                       'rule_name': alert.rule_name,
                       'context': alert.context
                   })
        return True

    def _send_console_alert(self, alert: AlertMessage) -> bool:
        """发送控制台告警"""
        print(f"\n{'='*60}")
        print(f"🚨 异常告警 [{alert.level.value.upper()}]")
        print(f"📝 标题: {alert.title}")
        print(f"💬 消息: {alert.message}")
        print(f"🕐 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
        print(f"📋 规则: {alert.rule_name}")
        print(f"📊 上下文: {json.dumps(alert.context, indent=2, ensure_ascii=False)}")
        print(f"{'='*60}\n")
        return True

    def _send_sms_alert(self, alert: AlertMessage) -> bool:
        """发送短信告警"""
        # 这里可以集成短信服务，如阿里云SMS、腾讯云SMS等
        logger.info(f"SMS告警: {alert.title} - {alert.message}")
        # 实际实现需要根据具体的短信服务商API
        return True


class ExceptionMonitor:
    """异常监控器"""

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.channels: Dict[AlertChannel, AlertChannelConfig] = {}
        self.exception_history = deque(maxlen=1000)
        self.stats = {}
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.rules.append(rule)
            logger.info(f"添加告警规则: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        with self._lock:
            self.rules = [r for r in self.rules if r.name != rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def configure_channel(self, channel: AlertChannel, config: Dict[str, Any]):
        """配置告警渠道"""
        with self._lock:
            self.channels[channel] = AlertChannelConfig(channel, config)
            logger.info(f"配置告警渠道: {channel.value}")

    def report_exception(self, exception_context: Dict[str, Any]):
        """报告异常"""
        with self._lock:
            # 添加到历史记录
            self.exception_history.append({
                'context': exception_context,
                'timestamp': time.time()
            })

            # 更新统计信息
            exception_type = exception_context.get('type', 'unknown')
            self.stats[exception_type] = self.stats.get(exception_type, 0) + 1

            # 检查告警规则
            self._check_alert_rules(exception_context)

    def _check_alert_rules(self, exception_context: Dict[str, Any]):
        """检查告警规则"""
        for rule in self.rules:
            if rule.should_trigger(exception_context):
                rule.trigger(exception_context)

                # 创建告警消息
                alert = AlertMessage(
                    title=f"异常告警: {exception_context.get('type', 'unknown')}",
                    message=exception_context.get('message', '异常发生'),
                    level=rule.level,
                    context=exception_context,
                    rule_name=rule.name
                )

                # 发送告警
                self._send_alert(alert, rule.channels)

    def _send_alert(self, alert: AlertMessage, channels: List[AlertChannel]):
        """发送告警"""
        for channel in channels:
            if channel in self.channels:
                success = self.channels[channel].send_alert(alert)
                if success:
                    logger.debug(f"告警发送成功: {channel.value}")
                else:
                    logger.error(f"告警发送失败: {channel.value}")

    def start_monitoring(self):
        """启动监控"""
        if self._monitoring_thread is None:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="ExceptionMonitor"
            )
            self._monitoring_thread.start()
            logger.info("异常监控已启动")

    def _monitoring_loop(self):
        """监控循环"""
        while not self._shutdown_event.is_set():
            try:
                self._perform_health_check()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(30)

    def _perform_health_check(self):
        """执行健康检查"""
        # 检查异常频率
        current_time = time.time()
        recent_exceptions = [
            exc for exc in self.exception_history
            if current_time - exc['timestamp'] < 300  # 最近5分钟
        ]

        if len(recent_exceptions) > 10:  # 每5分钟超过10个异常
            alert_context = {
                'type': 'high_exception_rate',
                'message': f'异常发生频率过高: {len(recent_exceptions)} 个/5分钟',
                'exception_count': len(recent_exceptions),
                'time_window': 300
            }
            self.report_exception(alert_context)

    def shutdown(self):
        """关闭监控"""
        self._shutdown_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        logger.info("异常监控已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'total_exceptions': len(self.exception_history),
                'exception_types': self.stats.copy(),
                'active_rules': len([r for r in self.rules if r.enabled]),
                'configured_channels': list(self.channels.keys())
            }

    def get_recent_exceptions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的异常"""
        return list(self.exception_history)[-limit:]

# 预定义告警规则


def create_high_frequency_rule(name: str = "high_exception_frequency",
                               threshold: int = 10,
                               time_window: int = 300) -> AlertRule:
    """创建高频异常告警规则"""
    def condition(context: Dict[str, Any]) -> bool:
        exception_type = context.get('type', '')
        # 这里可以根据具体逻辑判断是否触发告警
        return 'high_frequency' in exception_type.lower()

    return AlertRule(
        name=name,
        condition=condition,
        level=AlertLevel.WARNING,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
        cooldown=600,  # 10分钟冷却
        metadata={'threshold': threshold, 'time_window': time_window}
    )


def create_critical_exception_rule(name: str = "critical_exceptions") -> AlertRule:
    """创建严重异常告警规则"""
    def condition(context: Dict[str, Any]) -> bool:
        severity = context.get('severity', '').lower()
        return severity in ['critical', 'error']

    return AlertRule(
        name=name,
        condition=condition,
        level=AlertLevel.CRITICAL,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.EMAIL],
        cooldown=300,  # 5分钟冷却
        metadata={'critical_severities': ['critical', 'error']}
    )


def create_database_exception_rule(name: str = "database_exceptions") -> AlertRule:
    """创建数据库异常告警规则"""
    def condition(context: Dict[str, Any]) -> bool:
        exception_type = context.get('type', '').lower()
        return 'database' in exception_type or 'connection' in exception_type

    return AlertRule(
        name=name,
        condition=condition,
        level=AlertLevel.ERROR,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
        cooldown=180,  # 3分钟冷却
        metadata={'target_exceptions': ['database', 'connection']}
    )
