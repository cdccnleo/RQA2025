#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测层告警系统

实现短期优化目标：添加更多监控指标、实现自动告警机制、完善日志系统
"""

import time
import threading
import logging
import smtplib
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:

    """告警规则"""
    name: str
    metric: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    severity: str  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True
    cooldown_seconds: int = 300  # 5分钟冷却时间
    last_triggered: Optional[datetime] = None


@dataclass
class AlertEvent:

    """告警事件"""
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


class AlertManager:

    """告警管理器"""

    def __init__(self):

        self.rules: List[AlertRule] = []
        self.events: List[AlertEvent] = []
        self.handlers: Dict[str, List[Callable]] = {
            'info': [],
            'warning': [],
            'error': [],
            'critical': []
        }
        self.monitoring = False
        self.monitor_thread = None

        # 初始化默认告警规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认告警规则"""

        default_rules = [
            AlertRule('high_memory_usage', 'memory_percent', 80.0, '>', 'warning'),
            AlertRule('critical_memory_usage', 'memory_percent', 90.0, '>', 'critical'),
            AlertRule('high_cpu_usage', 'cpu_percent', 80.0, '>', 'warning'),
            AlertRule('critical_cpu_usage', 'cpu_percent', 95.0, '>', 'critical'),
            AlertRule('low_cache_hit_rate', 'cache_hit_rate', 0.7, '<', 'warning'),
            AlertRule('high_error_rate', 'error_rate', 0.05, '>', 'error'),
            AlertRule('low_throughput', 'throughput_per_second', 100.0, '<', 'warning'),
            AlertRule('high_thread_count', 'thread_count', 100, '>', 'warning'),
            AlertRule('critical_thread_count', 'thread_count', 200, '>', 'critical'),
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"移除告警规则: {rule_name}")

    def add_handler(self, severity: str, handler: Callable):
        """添加告警处理器"""
        if severity in self.handlers:
            self.handlers[severity].append(handler)
            logger.info(f"添加{severity}级别告警处理器")

    def start_monitoring(self, interval_seconds: int = 30):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("告警监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("告警监控已停止")

    def _monitoring_loop(self, interval_seconds: int):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()

                # 检查告警规则
                self._check_alert_rules(metrics)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"告警监控错误: {e}")
                time.sleep(interval_seconds)

    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=0.1)

        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / 1024 / 1024,
            'cpu_percent': cpu_usage,
            'thread_count': threading.active_count(),
            'cache_hit_rate': 0.85,  # 模拟值，实际应从缓存管理器获取
            'error_rate': 0.02,  # 模拟值，实际应从错误统计获取
            'throughput_per_second': 1500.0,  # 模拟值，实际应从数据处理器获取
            'disk_usage_percent': psutil.disk_usage('/').percent
        }

    def _check_alert_rules(self, metrics: Dict[str, float]):
        """检查告警规则"""
        for rule in self.rules:
            if not rule.enabled:
                continue

            # 检查冷却时间
            if rule.last_triggered and \
               (datetime.now() - rule.last_triggered).seconds < rule.cooldown_seconds:
                continue

            # 获取指标值
            if rule.metric not in metrics:
                continue

            current_value = metrics[rule.metric]

            # 检查阈值
            if self._evaluate_condition(current_value, rule.operator, rule.threshold):
                # 触发告警
                self._trigger_alert(rule, current_value, metrics)

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """评估条件"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        return False

    def _trigger_alert(self, rule: AlertRule, current_value: float, context: Dict[str, Any]):
        """触发告警"""
        # 更新最后触发时间
        rule.last_triggered = datetime.now()

        # 创建告警事件
        event = AlertEvent(
            rule_name=rule.name,
            metric=rule.metric,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            timestamp=datetime.now(),
            message=f"告警: {rule.name} - {rule.metric} = {current_value} {rule.operator} {rule.threshold}",
            context=context
        )

        # 添加到事件列表
        self.events.append(event)

        # 调用处理器
        self._call_handlers(event)

        logger.warning(f"触发告警: {event.message}")

    def _call_handlers(self, event: AlertEvent):
        """调用告警处理器"""
        handlers = self.handlers.get(event.severity, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"告警处理器错误: {e}")

    def get_alert_history(self, hours: int = 24) -> List[AlertEvent]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [event for event in self.events if event.timestamp > cutoff_time]

    def get_active_alerts(self) -> List[AlertEvent]:
        """获取活跃告警"""
        # 返回最近1小时内的告警
        return self.get_alert_history(hours=1)

    def clear_alert_history(self):
        """清空告警历史"""
        self.events.clear()
        logger.info("告警历史已清空")


class EmailAlertHandler:

    """邮件告警处理器"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def __call__(self, event: AlertEvent):
        """处理告警事件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = self.username  # 发送给自己
            msg['Subject'] = f"回测系统告警 - {event.severity.upper()}"

            body = f"""
            告警详情:
            - 规则名称: {event.rule_name}
            - 指标: {event.metric}
            - 当前值: {event.current_value}
            - 阈值: {event.threshold}
            - 严重程度: {event.severity}
            - 时间: {event.timestamp}
            - 消息: {event.message}
            """

            msg.attach(MIMEText(body, 'plain'))

            # 发送邮件
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            logger.info(f"告警邮件已发送: {event.rule_name}")

        except Exception as e:
            logger.error(f"发送告警邮件失败: {e}")


class LogAlertHandler:

    """日志告警处理器"""

    def __init__(self, log_file: str = "logs / alerts.log"):

        self.log_file = log_file

    def __call__(self, event: AlertEvent):
        """处理告警事件"""
        try:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'rule_name': event.rule_name,
                'metric': event.metric,
                'current_value': event.current_value,
                'threshold': event.threshold,
                'severity': event.severity,
                'message': event.message
            }

            with open(self.log_file, 'a', encoding='utf - 8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            logger.info(f"告警日志已记录: {event.rule_name}")

        except Exception as e:
            logger.error(f"记录告警日志失败: {e}")


class WebhookAlertHandler:

    """Webhook告警处理器"""

    def __init__(self, webhook_url: str):

        self.webhook_url = webhook_url

    def __call__(self, event: AlertEvent):
        """处理告警事件"""
        try:
            import requests

            payload = {
                'timestamp': event.timestamp.isoformat(),
                'rule_name': event.rule_name,
                'metric': event.metric,
                'current_value': event.current_value,
                'threshold': event.threshold,
                'severity': event.severity,
                'message': event.message
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Webhook告警已发送: {event.rule_name}")

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")


class AlertSystem:

    """综合告警系统"""

    def __init__(self):

        self.alert_manager = AlertManager()
        self.setup_default_handlers()

    def setup_default_handlers(self):
        """设置默认处理器"""
        # 添加日志处理器
        log_handler = LogAlertHandler()
        self.alert_manager.add_handler('warning', log_handler)
        self.alert_manager.add_handler('error', log_handler)
        self.alert_manager.add_handler('critical', log_handler)

        # 添加邮件处理器（需要配置SMTP）
        # email_handler = EmailAlertHandler('smtp.gmail.com', 587, 'user@gmail.com', 'password')
        # self.alert_manager.add_handler('critical', email_handler)

        # 添加Webhook处理器（需要配置Webhook URL）
        # webhook_handler = WebhookAlertHandler('https://hooks.slack.com / services / xxx')
        # self.alert_manager.add_handler('error', webhook_handler)
        # self.alert_manager.add_handler('critical', webhook_handler)

    def start(self):
        """启动告警系统"""
        self.alert_manager.start_monitoring()

    def stop(self):
        """停止告警系统"""
        self.alert_manager.stop_monitoring()

    def add_custom_rule(self, name: str, metric: str, threshold: float,


                        operator: str = '>', severity: str = 'warning'):
        """添加自定义告警规则"""
        rule = AlertRule(name, metric, threshold, operator, severity)
        self.alert_manager.add_rule(rule)

    def get_alert_status(self) -> Dict[str, Any]:
        """获取告警状态"""
        return {
            'monitoring': self.alert_manager.monitoring,
            'active_rules': len(self.alert_manager.rules),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'total_events': len(self.alert_manager.events),
            'recent_alerts': [
                {
                    'rule_name': event.rule_name,
                    'severity': event.severity,
                    'timestamp': event.timestamp.isoformat(),
                    'message': event.message
                }
                for event in self.alert_manager.get_alert_history(hours=1)
            ]
        }


# 全局告警系统实例
alert_system = AlertSystem()


def start_alert_system():
    """启动告警系统"""
    alert_system.start()


def stop_alert_system():
    """停止告警系统"""
    alert_system.stop()


def add_alert_rule(name: str, metric: str, threshold: float,


                   operator: str = '>', severity: str = 'warning'):
    """添加告警规则"""
    alert_system.add_custom_rule(name, metric, threshold, operator, severity)


def get_alert_status() -> Dict[str, Any]:
    """获取告警状态"""
    return alert_system.get_alert_status()


def get_alert_history(hours: int = 24) -> List[Dict[str, Any]]:
    """获取告警历史"""
    events = alert_system.alert_manager.get_alert_history(hours)
    return [
        {
            'rule_name': event.rule_name,
            'metric': event.metric,
            'current_value': event.current_value,
            'threshold': event.threshold,
            'severity': event.severity,
            'timestamp': event.timestamp.isoformat(),
            'message': event.message
        }
        for event in events
    ]
