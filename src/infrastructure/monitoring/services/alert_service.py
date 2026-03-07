"""
alert_system 模块

提供 alert_system 相关功能和接口。
"""

import re
import json
import logging
import requests

import queue
import smtplib
import threading
import time

from .core.constants import (
    NOTIFICATION_MAX_RETRIES, NOTIFICATION_RETRY_DELAY,
    ALERT_LEVEL_INFO, ALERT_LEVEL_WARNING, ALERT_LEVEL_ERROR, ALERT_LEVEL_CRITICAL
)
from .core.exceptions import (
    MonitoringException, NotificationError, handle_monitoring_exception
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
#!/usr/bin/env python3
"""
智能告警系统模块

提供完整的监控告警规则配置、触发和通知功能
    创建时间: 2024年12月
"""

logger = logging.getLogger(__name__)

# 导入组件
try:
    from ..components.alert_rule_manager import AlertRuleManager
    from ..components.alert_condition_evaluator import AlertConditionEvaluator
    from ..components.alert_processor import AlertProcessor
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入告警系统组件: {e}")
    COMPONENTS_AVAILABLE = False

class AlertJSONEncoder(json.JSONEncoder):

    """自定义JSON编码器，支持枚举和datetime序列化"""

    def default(self, obj):

        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class AlertLevel(Enum):

    """告警级别枚举"""
    INFO = "info"        # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"      # 错误
    CRITICAL = "critical"  # 严重

class AlertChannel(Enum):

    """告警渠道枚举"""
    EMAIL = "email"        # 邮件
    SMS = "sms"           # 短信
    WEBHOOK = "webhook"    # Webhook
    SLACK = "slack"       # Slack
    WECHAT = "wechat"     # 微信
    CONSOLE = "console"   # 控制台

class AlertStatus(Enum):

    """告警状态枚举"""
    ACTIVE = "active"      # 激活
    RESOLVED = "resolved"  # 已解决
    ACKNOWLEDGED = "acknowledged"  # 已确认
    SUPPRESSED = "suppressed"  # 抑制

@ dataclass

class AlertRule:

    """告警规则"""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]  # 触发条件
    level: AlertLevel
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown: int = 300  # 冷却时间(秒)
    metadata: Dict[str, Any] = None  # 扩展元数据
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

        # 确保channels包含枚举值
        if self.channels and isinstance(self.channels[0], str):
            self.channels = [AlertChannel(channel) for channel in self.channels]

@ dataclass

class Alert:

    """告警实例"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    level: AlertLevel
    status: AlertStatus = AlertStatus.ACTIVE  # 告警状态
    data: Dict[str, Any] = None  # 告警数据
    created_at: datetime = None
    source: str = "system"  # 告警来源
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.data is None:
            self.data = {}

@ dataclass

class AlertNotification:

    """告警通知"""
    notification_id: str
    alert_id: str
    channel: AlertChannel
    recipient: str
    status: str  # sent, failed, pending
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None

class IAlertNotifier(ABC):

    """告警通知器接口"""

    @ abstractmethod

    def send_notification(self, alert: Alert, recipient: str) -> bool:

        """发送通知"""
        pass

class EmailNotifier(IAlertNotifier):

    """邮件通知器"""

    def __init__(self, smtp_config: Dict[str, Any]):

        if not smtp_config or not all(key in smtp_config for key in ['host', 'port', 'username', 'password']):
            raise ValueError("SMTP配置不完整，必须包含host、port、username和password")
        self.smtp_config = smtp_config
        self.server = None
        # 设置默认发件人
        self.from_email = smtp_config.get('from_email', smtp_config['username'])

    def send_notification(self, alert: Alert, recipient: str) -> bool:

        """发送邮件通知"""
        try:
            if not self.server:
                self.server = smtplib.SMTP(
                    self.smtp_config['host'],
                    self.smtp_config['port']
                )
            if self.smtp_config.get('use_tls'):
                self.server.starttls()
            self.server.login(
                self.smtp_config['username'],
                self.smtp_config['password']
            )

            subject = f"[{alert.level.value.upper()}] {alert.title}"
            body = f"""
                告警详情：
标题：{alert.title}
级别：{alert.level.value}
时间：{alert.created_at}
消息：{alert.message}

告警数据：
{json.dumps(alert.data, ensure_ascii=False, indent=2)}
    """

            message = f"Subject: {subject}\n\n{body}"

            self.server.sendmail(
                self.from_email,
                recipient,
                message
            )

            logger.info(f"邮件告警发送成功: {recipient}")
            return True

        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")
            return False

class WebhookNotifier(IAlertNotifier):

    """Webhook通知器"""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]]=None):

        self.webhook_url = webhook_url
        self.headers = headers or {'Content - Type': 'application / json'}

    def send_notification(self, alert: Alert, recipient: str) -> bool:

        """发送Webhook通知"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'title': alert.title,
                'message': alert.message,
                'level': alert.level.value,
                'status': alert.status.value,
                'data': alert.data,
                'created_at': alert.created_at.isoformat(),
                'recipient': recipient
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Webhook告警发送成功: {recipient}")
                return True
            else:
                logger.error(f"Webhook告警发送失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Webhook告警发送失败: {e}")
            return False

class SlackNotifier(IAlertNotifier):

    """Slack通知器"""

    def __init__(self, webhook_url: str):

        self.webhook_url = webhook_url

    @ handle_monitoring_exception("slack_notification")
    def send_notification(self, alert: Alert, recipient: str) -> bool:
        """
        发送Slack通知

        Args:
            alert: 告警对象
            recipient: 接收者（Slack频道）

        Returns:
            bool: 发送是否成功

        Raises:
            NotificationError: 当通知发送失败时抛出
        """
        try:
            payload = self._build_slack_payload(alert, recipient)
            return self._send_slack_request(payload)

        except NotificationError:
            raise
        except Exception as e:
            error_msg = f"Slack告警发送失败: {str(e)}"
            raise NotificationError(error_msg, notification_type="slack",
                                  recipient=recipient,
                                  details={"original_error": str(e)}) from e

    def _build_slack_payload(self, alert: Alert, recipient: str) -> Dict[str, Any]:
        """构建Slack通知的payload"""
        color = self._get_alert_color(alert.level)
        fields = self._build_alert_fields(alert)

        return {
            "channel": recipient,
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": fields,
                "footer": "智能告警系统",
                "ts": int(alert.created_at.timestamp())
            }]
        }

    def _get_alert_color(self, alert_level: AlertLevel) -> str:
        """获取告警级别的颜色"""
        color_map = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ffcc02",
            AlertLevel.ERROR: "#ff6b6b",
            AlertLevel.CRITICAL: "#dc3545"
        }
        return color_map.get(alert_level, "#36a64f")

    def _build_alert_fields(self, alert: Alert) -> List[Dict[str, Any]]:
        """构建告警字段"""
        return [
            {
                "title": "级别",
                "value": alert.level.value.upper(),
                "short": True
            },
            {
                "title": "时间",
                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "short": True
            }
        ]

    def _send_slack_request(self, payload: Dict[str, Any]) -> bool:
        """发送Slack HTTP请求"""
        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            logger.info(f"Slack告警发送成功")
            return True
        else:
            logger.error(f"Slack告警发送失败: HTTP {response.status_code}")
            return False

class ConsoleNotifier(IAlertNotifier):

    """控制台通知器"""

    def send_notification(self, alert: Alert, recipient: str) -> bool:

        """发送控制台通知"""
        try:
            print(f"\n{'='*60}")
            print(f"🔔 告警通知 [{alert.level.value.upper()}]")
            print(f"标题: {alert.title}")
            print(f"消息: {alert.message}")
            print(f"时间: {alert.created_at}")
            print(f"数据: {json.dumps(alert.data, ensure_ascii=False, indent=2, cls=AlertJSONEncoder)}")
            print(f"{'='*60}\n")

            logger.info(f"控制台告警显示成功")
            return True

        except Exception as e:
            logger.error(f"控制台告警显示失败: {e}")
            return False

class IntelligentAlertSystem:

    """智能告警系统"""

    def __init__(self):
        # 初始化组件
        self._init_components()

        # 向后兼容的属性
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notifications: List[AlertNotification] = {}
        self.notifiers: Dict[AlertChannel, IAlertNotifier] = {}
        self.alert_queue = queue.Queue()

        # 告警状态跟踪
        self.rule_last_triggered: Dict[str, datetime] = {}
        self.alert_counter = 0

        # 启动告警处理线程
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_alerts)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        logger.info("智能告警系统初始化完成")
    
    def _init_components(self) -> None:
        """初始化子组件"""
        if COMPONENTS_AVAILABLE:
            self._rule_manager = AlertRuleManager()
            self._condition_evaluator = AlertConditionEvaluator()
            self._alert_processor = AlertProcessor()
        else:
            self._rule_manager = None
            self._condition_evaluator = None
            self._alert_processor = None

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        # 使用规则管理器组件
        if self._rule_manager:
            self._rule_manager.add_rule(rule)
        
        # 向后兼容
        self.rules[rule.rule_id] = rule
        logger.info(f"告警规则添加成功: {rule.rule_id} - {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        # 使用规则管理器组件
        if self._rule_manager:
            self._rule_manager.remove_rule(rule_id)
        
        # 向后兼容
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"告警规则移除成功: {rule_id}")

    def register_notifier(self, channel: AlertChannel, notifier: IAlertNotifier):

        """注册通知器"""
        # 确保channel是枚举值
        if isinstance(channel, str):
            channel = AlertChannel(channel)
        self.notifiers[channel] = notifier
        logger.info(f"通知器注册成功: {channel.value}")

    def evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估触发条件"""
        # 使用条件评估器组件
        if self._condition_evaluator:
            return self._condition_evaluator.evaluate(condition, data)
        
        # 回退到原有方法
        try:
            operator = condition.get('operator', 'eq')
            field = condition.get('field')
            expected_value = condition.get('value')

            if field not in data:
                return False

            actual_value = data[field]

            # 支持的运算符
            if operator == 'eq':
                return actual_value == expected_value
            elif operator == 'ne':
                return actual_value != expected_value
            elif operator == 'gt':
                return float(actual_value) > float(expected_value)
            elif operator == 'gte':
                return float(actual_value) >= float(expected_value)
            elif operator == 'lt':
                return float(actual_value) < float(expected_value)
            elif operator == 'lte':
                return float(actual_value) <= float(expected_value)
            elif operator == 'contains':
                return str(expected_value) in str(actual_value)
            elif operator == 'regex':
                return bool(re.search(str(expected_value), str(actual_value)))

            return False

        except Exception as e:
            logger.error(f"条件评估失败: {e}")
            return False

    def check_alerts(self, data: Dict[str, Any], source: str = "system") -> bool:

        """检查告警触发"""
        current_time = datetime.now()
        alert_triggered = False

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # 检查冷却时间
            last_triggered = self.rule_last_triggered.get(rule.rule_id)
            if last_triggered and (current_time - last_triggered).seconds < rule.cooldown:
                continue

            # 评估条件
            if self.evaluate_condition(rule.condition, data):
                # 创建告警
                alert = self._create_alert(rule, data, source)
                if alert:
                    self.alert_queue.put(alert)
                    self.rule_last_triggered[rule.rule_id] = current_time
                    alert_triggered = True

                    logger.warning(f"告警触发: {rule.name} - {alert.title}")

        return alert_triggered

    def _create_alert(self, rule: AlertRule, data: Dict[str, Any], source: str) -> Optional[Alert]:

        """创建告警"""
        try:
            self.alert_counter += 1
            alert_id = f"alert_{self.alert_counter:06d}"

            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                title=rule.name,
                message=rule.description,
                level=rule.level,
                status=AlertStatus.ACTIVE,
                data=data,
                created_at=datetime.now()
            )

            self.alerts[alert_id] = alert
            return alert

        except Exception as e:
            logger.error(f"告警创建失败: {e}")
            return None

    def _process_alerts(self):

        """处理告警队列"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)

                # 发送通知
                self._send_notifications(alert)

                # 标记队列任务完成
                self.alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"告警处理失败: {e}")

    def _send_notifications(self, alert: Alert):

        """发送告警通知"""
        rule = self.rules.get(alert.rule_id)
        if not rule:
            return

        for channel in rule.channels:
            notifier = self.notifiers.get(channel)
            if not notifier:
                logger.warning(f"未找到通知器: {channel.value}")
                continue

            # 获取收件人（这里简化处理，实际应从配置获取）
            recipients = self._get_recipients(channel, alert.level)

            for recipient in recipients:
                success = notifier.send_notification(alert, recipient)

                notification = AlertNotification(
                    notification_id=f"notif_{alert.alert_id}_{channel.value}",
                    alert_id=alert.alert_id,
                    channel=channel,
                    recipient=recipient,
                    status="sent" if success else "failed",
                    sent_at=datetime.now(),
                    error_message=None if success else "发送失败"
                )

                self.notifications[notification.notification_id] = notification

    def _get_recipients(self, channel: AlertChannel, level: AlertLevel) -> List[str]:

        """获取收件人列表"""
        # 这里简化处理，实际应从配置系统获取
        recipients_map = {
            AlertChannel.EMAIL: ["admin@example.com", "ops@example.com"],
            AlertChannel.SLACK: ["#alerts", "#trading"],
            AlertChannel.WEBHOOK: ["https://api.example.com / webhook"],
            AlertChannel.CONSOLE: ["console"]
        }

        return recipients_map.get(channel, [])

    def acknowledge_alert(self, alert_id: str, user: str):

        """确认告警"""
        alert = self.alerts.get(alert_id)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user
            logger.info(f"告警已确认: {alert_id} by {user}")

    def resolve_alert(self, alert_id: str):

        """解决告警"""
        alert = self.alerts.get(alert_id)
        if alert and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            logger.info(f"告警已解决: {alert_id}")

    def get_active_alerts(self) -> List[Alert]:

        """获取活跃告警"""
        return [alert for alert in self.alerts.values()
                if alert.status == AlertStatus.ACTIVE]

    def get_alert_history(self, start_time: Optional[datetime] = None,

                          end_time: Optional[datetime] = None,
                          level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取告警历史"""
        alerts = list(self.alerts.values())

        if start_time:
            alerts = [a for a in alerts if a.created_at >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.created_at <= end_time]

        if level:
            alerts = [a for a in alerts if a.level == level]

        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    @handle_monitoring_exception("create_default_rules")
    def create_default_rules(self):
        """
        创建默认告警规则

        Raises:
            MonitoringException: 当创建规则失败时抛出
        """
        rules = []
        rules.extend(self._create_system_rules())
        rules.extend(self._create_business_rules())
        rules.extend(self._create_security_rules())

        self._add_rules_to_system(rules)
        logger.info(f"默认告警规则创建完成，共 {len(rules)} 条")

    def _create_system_rules(self) -> List[AlertRule]:
        """创建系统监控规则"""
        return [
            AlertRule(
                rule_id="cpu_usage_high",
                name="CPU使用率过高",
                description="系统CPU使用率超过阈值",
                condition={
                    "operator": "gt",
                    "field": "cpu_usage",
                    "value": 80
                },
                level=AlertLevel.WARNING,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL]
            ),
            AlertRule(
                rule_id="memory_usage_high",
                name="内存使用率过高",
                description="系统内存使用率超过阈值",
                condition={
                    "operator": "gt",
                    "field": "memory_usage",
                    "value": 85
                },
                level=AlertLevel.ERROR,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL]
            )
        ]

    def _create_business_rules(self) -> List[AlertRule]:
        """创建业务监控规则"""
        return [
            AlertRule(
                rule_id="trading_error",
                name="交易错误",
                description="检测到交易系统错误",
                condition={
                    "operator": "eq",
                    "field": "error_type",
                    "value": "trading_error"
                },
                level=AlertLevel.CRITICAL,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
            ),
            AlertRule(
                rule_id="data_loss",
                name="数据丢失",
                description="检测到数据丢失或损坏",
                condition={
                    "operator": "eq",
                    "field": "data_integrity",
                    "value": "failed"
                },
                level=AlertLevel.CRITICAL,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
            )
        ]

    def _create_security_rules(self) -> List[AlertRule]:
        """创建安全监控规则"""
        return [
            AlertRule(
                rule_id="login_failed",
                name="登录失败",
                description="检测到多次登录失败",
                condition={
                    "operator": "gte",
                    "field": "failed_attempts",
                    "value": 5
                },
                level=AlertLevel.WARNING,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL]
            )
        ]

    def _add_rules_to_system(self, rules: List[AlertRule]) -> None:
        """将规则添加到告警系统中"""
        for rule in rules:
            self.add_alert_rule(rule)

    def export_alert_report(self, start_time: datetime, end_time: datetime,

                            file_path: str = "alert_report.json"):
        """导出告警报告"""
        alerts = self.get_alert_history(start_time, end_time)

        report_data = {
            "export_time": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_alerts": len(alerts),
            "alerts_by_level": {},
            "alerts_by_status": {},
            "alert_details": [asdict(alert) for alert in alerts]
        }

        # 统计数据
        for alert in alerts:
            level_key = alert.level.value
            status_key = alert.status.value

            report_data["alerts_by_level"][level_key] = report_data["alerts_by_level"].get(level_key, 0) + 1
            report_data["alerts_by_status"][status_key] = report_data["alerts_by_status"].get(status_key, 0) + 1

        with open(file_path, 'w', encoding='utf - 8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, cls=AlertJSONEncoder)

        logger.info(f"告警报告导出完成: {file_path}")

    def shutdown(self):

        """关闭告警系统"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        logger.info("智能告警系统已关闭")

    def get_stats(self) -> Dict[str, Any]:

        """获取告警统计信息"""
        total_alerts = len(self.alerts)
        alerts_by_level = {}
        alerts_by_status = {}
        alerts_by_rule = {}

        for alert in self.alerts.values():
            # 按级别统计
            level_key = alert.level.value
            alerts_by_level[level_key] = alerts_by_level.get(level_key, 0) + 1

            # 按状态统计
            status_key = alert.status.value
            alerts_by_status[status_key] = alerts_by_status.get(status_key, 0) + 1

            # 按规则统计
            rule_key = alert.rule_id
            alerts_by_rule[rule_key] = alerts_by_rule.get(rule_key, 0) + 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]),
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_notifications": len(self.notifications),
            "alerts_by_level": alerts_by_level,
            "alerts_by_status": alerts_by_status,
            "alerts_by_rule": alerts_by_rule,
            "registered_channels": [channel.value for channel in self.notifiers.keys()]
        }

# 告警规则配置器

class AlertRuleConfigurator:

    """告警规则配置器"""

    def __init__(self, alert_system: IntelligentAlertSystem):

        self.alert_system = alert_system

    def create_rule_from_template(self, template_name: str, config: Dict[str, Any]) -> AlertRule:
        """从模板创建规则"""
        templates = self._get_alert_templates(config)
        template = self._validate_and_get_template(template_name, templates)
        rule_id = self._generate_rule_id(template_name, config)
        
        return self._build_alert_rule(rule_id, template, config)
    
    def _get_alert_templates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取告警模板"""
        return {
            "performance_threshold": self._create_performance_template(config),
            "error_rate_monitor": self._create_error_rate_template(config),
            "security_alert": self._create_security_template(config)
        }
    
    def _create_performance_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建性能阈值模板"""
        return {
            "name": "性能阈值告警",
            "description": "监控系统性能指标阈值",
            "condition": {
                "operator": "gt",
                "field": config.get('metric', 'cpu_usage'),
                "value": config.get('threshold', 80)
            },
            "level": AlertLevel(config.get('level', 'warning')),
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL]
        }
    
    def _create_error_rate_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建错误率监控模板"""
        return {
            "name": "错误率监控",
            "description": "监控系统错误率",
            "condition": {
                "operator": "gt",
                "field": "error_rate",
                "value": config.get('threshold', 5)
            },
            "level": AlertLevel.ERROR,
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        }
    
    def _create_security_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建安全告警模板"""
        return {
            "name": "安全告警",
            "description": "监控安全相关事件",
            "condition": {
                "operator": "eq",
                "field": "event_type",
                "value": config.get('event_type', 'unauthorized_access')
            },
            "level": AlertLevel.CRITICAL,
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        }
    
    def _validate_and_get_template(self, template_name: str, templates: Dict[str, Any]) -> Dict[str, Any]:
        """验证并获取模板"""
        template = templates.get(template_name)
        if not template:
            raise ValueError(f"模板不存在: {template_name}")
        return template
    
    def _generate_rule_id(self, template_name: str, config: Dict[str, Any]) -> str:
        """生成规则ID"""
        return config.get('rule_id', f"{template_name}_{int(time.time())}")
    
    def _build_alert_rule(self, rule_id: str, template: Dict[str, Any], config: Dict[str, Any]) -> AlertRule:
        """构建告警规则对象"""
        return AlertRule(
            rule_id=rule_id,
            name=config.get('name', template['name']),
            description=config.get('description', template['description']),
            condition=template['condition'],
            level=template['level'],
            channels=config.get('channels', template['channels']),
            cooldown=config.get('cooldown', 300)
        )

# 使用示例
if __name__ == "__main__":
    # 初始化告警系统
    alert_system = IntelligentAlertSystem()

    # 创建默认规则
    alert_system.create_default_rules()

    # 注册控制台通知器
    console_notifier = ConsoleNotifier()
    alert_system.register_notifier(AlertChannel.CONSOLE, console_notifier)

    # 模拟系统监控数据
    monitoring_data = {
        "cpu_usage": 85,
        "memory_usage": 90,
        "error_rate": 3,
        "failed_attempts": 2
    }

    print("开始监控数据...")

    # 检查告警
    alert_system.check_alerts(monitoring_data, "system_monitor")

    # 等待告警处理
    time.sleep(2)

    # 获取活跃告警
    active_alerts = alert_system.get_active_alerts()
    print(f"\n活跃告警数量: {len(active_alerts)}")

    for alert in active_alerts:
        print(f"- {alert.title}: {alert.message}")

    # 模拟更多监控数据
    critical_data = {
        "cpu_usage": 95,
        "memory_usage": 98,
        "error_type": "trading_error",
        "failed_attempts": 6
    }

    print("\n处理严重告警...")
    alert_system.check_alerts(critical_data, "trading_system")

    # 等待处理
    time.sleep(2)

    # 获取告警历史
    all_alerts = alert_system.get_alert_history()
    print(f"\n总告警数量: {len(all_alerts)}")

    # 导出报告
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    alert_system.export_alert_report(start_time, end_time, "sample_alert_report.json")

    # 关闭系统
    alert_system.shutdown()

    print("\n智能告警系统演示完成")
