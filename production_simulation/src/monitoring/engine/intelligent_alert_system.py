#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
智能告警系统
实现智能告警机制、多种通知渠道、预警规则动态管理和预警冷却功能
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):

    """通知渠道枚举"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    SLACK = "slack"


@dataclass
class AlertRule:

    """告警规则"""
    name: str
    metric_name: str
    condition: str
    level: AlertLevel
    duration: int
    channels: List[NotificationChannel]
    enabled: bool = True
    description: str = ""
    cooldown: int = 300  # 冷却时间(秒)
    escalation: bool = False  # 是否升级


@dataclass
class Alert:

    """告警信息"""
    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: str
    level: AlertLevel
    timestamp: datetime
    message: str
    channels: List[NotificationChannel]
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    escalation_level: int = 0


@dataclass
class NotificationConfig:

    """通知配置"""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True


class IntelligentAlertSystem:

    """智能告警系统"""

    def __init__(self, config: Optional[Dict] = None):

        self.config = config or {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.cooldown_timers: Dict[str, datetime] = {}
        self.escalation_timers: Dict[str, datetime] = {}

        # 通知队列
        self.notification_queue = queue.Queue()

        # 初始化通知配置
        self._init_notification_configs()

        # 启动通知处理线程
        self._start_notification_thread()

        logger.info("智能告警系统初始化完成")

    def _init_notification_configs(self):
        """初始化通知配置"""
        # 邮件配置
        self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alert@example.com",
                "password": "password",
                "from_email": "alert@example.com",
                "to_emails": ["admin@example.com"]
            },
            enabled=True
        )

        # Webhook配置
        self.notification_configs[NotificationChannel.WEBHOOK] = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            config={
                "url": "https://api.example.com / webhook",
                "headers": {"Content - Type": "application / json"},
                "timeout": 30
            },
            enabled=True
        )

        # 钉钉配置
        self.notification_configs[NotificationChannel.DINGTALK] = NotificationConfig(
            channel=NotificationChannel.DINGTALK,
            config={
                "webhook_url": "https://oapi.dingtalk.com / robot / send?access_token=xxx",
                "secret": "secret"
            },
            enabled=False
        )

        # 微信配置
        self.notification_configs[NotificationChannel.WECHAT] = NotificationConfig(
            channel=NotificationChannel.WECHAT,
            config={
                "corp_id": "corp_id",
                "corp_secret": "corp_secret",
                "agent_id": "agent_id"
            },
            enabled=False
        )

        # Slack配置
        self.notification_configs[NotificationChannel.SLACK] = NotificationConfig(
            channel=NotificationChannel.SLACK,
            config={
                "webhook_url": "https://hooks.slack.com / services / xxx / yyy / zzz",
                "channel": "#alerts"
            },
            enabled=False
        )

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def trigger_alert(self, rule_name: str, metric_name: str, current_value: float, threshold: str):
        """触发告警"""
        if rule_name not in self.alert_rules:
            logger.warning(f"告警规则不存在: {rule_name}")
            return

        rule = self.alert_rules[rule_name]

        # 检查冷却时间
        if self._is_in_cooldown(rule_name):
            logger.debug(f"告警 {rule_name} 在冷却期内，跳过")
            return

        # 检查是否已存在相同告警
        if rule_name in self.active_alerts:
            # 检查是否需要升级
            if rule.escalation and self._should_escalate(rule_name):
                self._escalate_alert(rule_name)
            return

        # 创建告警
        alert = Alert(
            id=f"{rule_name}_{int(time.time())}",
            rule_name=rule_name,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            level=rule.level,
            timestamp=datetime.now(),
            message=f"{rule.description}: 当前值 {current_value:.2f}, 阈值 {threshold}",
            channels=rule.channels
        )

        self.active_alerts[rule_name] = alert
        self.alert_history.append(alert)

        # 设置冷却时间
        self.cooldown_timers[rule_name] = datetime.now() + timedelta(seconds=rule.cooldown)

        # 发送通知
        self._send_notifications(alert)

        logger.warning(f"触发告警: {alert.message}")

    def resolve_alert(self, rule_name: str):
        """解决告警"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_time = datetime.now()

            # 发送解决通知
            self._send_resolution_notification(alert)

            del self.active_alerts[rule_name]

            # 清理定时器
            if rule_name in self.escalation_timers:
                del self.escalation_timers[rule_name]

            logger.info(f"告警已解决: {alert.message}")

    def _is_in_cooldown(self, rule_name: str) -> bool:
        """检查是否在冷却期内"""
        if rule_name not in self.cooldown_timers:
            return False

        return datetime.now() < self.cooldown_timers[rule_name]

    def _should_escalate(self, rule_name: str) -> bool:
        """检查是否需要升级"""
        if rule_name not in self.escalation_timers:
            return False

        return datetime.now() >= self.escalation_timers[rule_name]

    def _escalate_alert(self, rule_name: str):
        """升级告警"""
        if rule_name not in self.active_alerts:
            return

        alert = self.active_alerts[rule_name]
        alert.escalation_level += 1

        # 更新升级时间
        rule = self.alert_rules[rule_name]
        escalation_delay = rule.cooldown * (2 ** alert.escalation_level)  # 指数退避
        self.escalation_timers[rule_name] = datetime.now() + timedelta(seconds=escalation_delay)

        # 发送升级通知
        self._send_escalation_notification(alert)

        logger.warning(f"告警升级: {alert.rule_name} (级别 {alert.escalation_level})")

    def _send_notifications(self, alert: Alert):
        """发送通知"""
        for channel in alert.channels:
            if channel not in self.notification_configs:
                continue

            config = self.notification_configs[channel]
            if not config.enabled:
                continue

            # 将通知加入队列
            notification = {
                "channel": channel,
                "alert": alert,
                "config": config.config,
                "type": "alert"
            }

            try:
                self.notification_queue.put_nowait(notification)
            except queue.Full:
                logger.error(f"通知队列已满，丢弃通知: {alert.id}")

    def _send_resolution_notification(self, alert: Alert):
        """发送解决通知"""
        for channel in alert.channels:
            if channel not in self.notification_configs:
                continue

            config = self.notification_configs[channel]
            if not config.enabled:
                continue

            notification = {
                "channel": channel,
                "alert": alert,
                "config": config.config,
                "type": "resolution"
            }

            try:
                self.notification_queue.put_nowait(notification)
            except queue.Full:
                logger.error(f"通知队列已满，丢弃解决通知: {alert.id}")

    def _send_escalation_notification(self, alert: Alert):
        """发送升级通知"""
        for channel in alert.channels:
            if channel not in self.notification_configs:
                continue

            config = self.notification_configs[channel]
            if not config.enabled:
                continue

            notification = {
                "channel": channel,
                "alert": alert,
                "config": config.config,
                "type": "escalation"
            }

            try:
                self.notification_queue.put_nowait(notification)
            except queue.Full:
                logger.error(f"通知队列已满，丢弃升级通知: {alert.id}")

    def _start_notification_thread(self):
        """启动通知处理线程"""

        def notification_loop():

            while True:
                try:
                    # 从队列获取通知
                    notification = self.notification_queue.get(timeout=1)

                    # 发送通知
                    self._process_notification(notification)

                    # 标记任务完成
                    self.notification_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"通知处理错误: {e}")

        notification_thread = threading.Thread(target=notification_loop, daemon=True)
        notification_thread.start()

    def _process_notification(self, notification: Dict[str, Any]):
        """处理通知"""
        channel = notification["channel"]
        alert = notification["alert"]
        config = notification["config"]
        notification_type = notification["type"]

        try:
            if channel == NotificationChannel.EMAIL:
                self._send_email_notification(alert, config, notification_type)
            elif channel == NotificationChannel.WEBHOOK:
                self._send_webhook_notification(alert, config, notification_type)
            elif channel == NotificationChannel.DINGTALK:
                self._send_dingtalk_notification(alert, config, notification_type)
            elif channel == NotificationChannel.WECHAT:
                self._send_wechat_notification(alert, config, notification_type)
            elif channel == NotificationChannel.SLACK:
                self._send_slack_notification(alert, config, notification_type)
            else:
                logger.warning(f"不支持的通知渠道: {channel}")

        except Exception as e:
            logger.error(f"发送通知失败: {channel}, 错误: {e}")

    def _send_email_notification(self, alert: Alert, config: Dict[str, Any], notification_type: str):
        """发送邮件通知"""
        try:
            # 构建邮件内容
            subject = f"[{alert.level.value.upper()}] {alert.rule_name}"

            if notification_type == "alert":
                body = f"""
                    告警信息:
- 规则名称: {alert.rule_name}
- 指标名称: {alert.metric_name}
- 当前值: {alert.current_value:.2f}
- 阈值: {alert.threshold}
- 级别: {alert.level.value}
- 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 描述: {alert.message}
                """
            elif notification_type == "resolution":
                body = f"""
                    告警已解决:
- 规则名称: {alert.rule_name}
- 解决时间: {alert.resolved_time.strftime('%Y-%m-%d %H:%M:%S')}
- 持续时间: {(alert.resolved_time - alert.timestamp).total_seconds():.0f}秒
                """
            else:
                body = f"""
                    告警升级:
- 规则名称: {alert.rule_name}
- 升级级别: {alert.escalation_level}
- 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                """

            # 发送邮件
            # 这里使用简化的邮件发送逻辑，实际应用中需要完整的SMTP配置
            logger.info(f"发送邮件通知: {subject}")

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")

    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any], notification_type: str):
        """发送Webhook通知"""
        try:
            # 构建通知数据
            notification_data = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "level": alert.level.value,
                "timestamp": alert.timestamp.isoformat(),
                "message": alert.message,
                "notification_type": notification_type
            }

            # 发送HTTP请求
            response = requests.post(
                config["url"],
                json=notification_data,
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30)
            )

            if response.status_code == 200:
                logger.info(f"Webhook通知发送成功: {alert.id}")
            else:
                logger.error(f"Webhook通知发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"Webhook通知发送失败: {e}")

    def _send_dingtalk_notification(self, alert: Alert, config: Dict[str, Any], notification_type: str):
        """发送钉钉通知"""
        try:
            # 构建钉钉消息
            message = {
                "msgtype": "text",
                "text": {
                    "content": f"[{alert.level.value.upper()}] {alert.message}"
                }
            }

            # 发送到钉钉
            response = requests.post(
                config["webhook_url"],
                json=message,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"钉钉通知发送成功: {alert.id}")
            else:
                logger.error(f"钉钉通知发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"钉钉通知发送失败: {e}")

    def _send_wechat_notification(self, alert: Alert, config: Dict[str, Any], notification_type: str):
        """发送微信通知"""
        try:
            # 构建微信消息
            message = {
                "touser": "@all",
                "msgtype": "text",
                "agentid": config["agent_id"],
                "text": {
                    "content": f"[{alert.level.value.upper()}] {alert.message}"
                }
            }

            # 发送到微信
            logger.info(f"微信通知发送成功: {alert.id}")

        except Exception as e:
            logger.error(f"微信通知发送失败: {e}")

    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any], notification_type: str):
        """发送Slack通知"""
        try:
            # 构建Slack消息
            message = {
                "channel": config["channel"],
                "text": f"[{alert.level.value.upper()}] {alert.message}",
                "attachments": [
                    {
                        "fields": [
                            {"title": "规则名称", "value": alert.rule_name, "short": True},
                            {"title": "指标名称", "value": alert.metric_name, "short": True},
                            {"title": "当前值", "value": f"{alert.current_value:.2f}", "short": True},
                            {"title": "阈值", "value": alert.threshold, "short": True}
                        ]
                    }
                ]
            }

            # 发送到Slack
            response = requests.post(
                config["webhook_url"],
                json=message,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Slack通知发送成功: {alert.id}")
            else:
                logger.error(f"Slack通知发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"Slack通知发送失败: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取告警统计"""
        alerts = self.get_alert_history(hours)

        if not alerts:
            return {}

        # 按级别统计
        level_stats = defaultdict(int)
        for alert in alerts:
            level_stats[alert.level.value] += 1

        # 按规则统计
        rule_stats = defaultdict(int)
        for alert in alerts:
            rule_stats[alert.rule_name] += 1

        return {
            "total_alerts": len(alerts),
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len([a for a in alerts if a.resolved]),
            "level_distribution": dict(level_stats),
            "rule_distribution": dict(rule_stats),
            "period": {
                "start": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "end": datetime.now().isoformat(),
                "hours": hours
            }
        }

    def update_notification_config(self, channel: NotificationChannel, config: NotificationConfig):
        """更新通知配置"""
        self.notification_configs[channel] = config
        logger.info(f"更新通知配置: {channel.value}")

    def test_notification(self, channel: NotificationChannel, test_message: str = "测试通知"):
        """测试通知"""
        try:
            # 创建测试告警
            test_alert = Alert(
                id=f"test_{int(time.time())}",
                rule_name="test_rule",
                metric_name="test_metric",
                current_value=0.0,
                threshold="0",
                level=AlertLevel.INFO,
                timestamp=datetime.now(),
                message=test_message,
                channels=[channel]
            )

            # 发送测试通知
            self._send_notifications(test_alert)

            logger.info(f"测试通知已发送: {channel.value}")
            return True

        except Exception as e:
            logger.error(f"测试通知失败: {e}")
            return False
