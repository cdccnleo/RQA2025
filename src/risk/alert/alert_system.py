#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能风险预警系统

提供多级预警、自动干预、预警通知等功能"""

import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import queue

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """预警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):

    """预警类型"""
    RISK_THRESHOLD = "risk_threshold"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_ALERT = "volatility_alert"
    LIQUIDITY_ALERT = "liquidity_alert"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    MARKET_ANOMALY = "market_anomaly"


class AlertStatus(Enum):

    """预警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class AlertRule:

    """预警规则"""
    rule_id: str
    rule_name: str
    alert_type: AlertType
    alert_level: AlertLevel
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool = True
    cooldown_minutes: int = 30
    created_time: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:

    """预警"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    title: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_time: Optional[datetime] = None
    notification_sent: bool = False


@dataclass
class NotificationConfig:

    """通知配置"""
    email_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    sms_recipients: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)
    notification_cooldown_minutes: int = 15


class AlertSystem:

    """智能风险预警系统"""

    def __init__(self, config: Optional[Dict] = None):

        self.config = config or {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_config = NotificationConfig()
        self.lock = threading.RLock()

        # 通知队列
        self.notification_queue = queue.Queue()
        self.notification_thread = None

        # 停止标志
        self._stop_event = threading.Event()

        # 预警处理器
        self.alert_handlers = defaultdict(list)

        # 初始化默认规则
        self._init_default_rules()

        # 启动通知线程
        self._start_notification_thread()

        logger.info("智能风险预警系统初始化完成")

    def _init_default_rules(self):
        """初始化默认预警规则"""

        default_rules = [
            AlertRule(
                rule_id="risk_threshold_001",
                rule_name="风险阈值预警",
                alert_type=AlertType.RISK_THRESHOLD,
                alert_level=AlertLevel.WARNING,
                conditions={"risk_score": 0.7},
                actions=["email", "log"],
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="position_limit_001",
                rule_name="仓位限制预警",
                alert_type=AlertType.POSITION_LIMIT,
                alert_level=AlertLevel.ERROR,
                conditions={"position_ratio": 0.8},
                actions=["email", "sms", "log"],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="volatility_alert_001",
                rule_name="波动率预警",
                alert_type=AlertType.VOLATILITY_ALERT,
                alert_level=AlertLevel.WARNING,
                conditions={"volatility": 0.3},
                actions=["email", "log"],
                cooldown_minutes=60
            ),
            AlertRule(
                rule_id="liquidity_alert_001",
                rule_name="流动性预警",
                alert_type=AlertType.LIQUIDITY_ALERT,
                alert_level=AlertLevel.ERROR,
                conditions={"liquidity_ratio": 0.1},
                actions=["email", "sms", "log"],
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="system_error_001",
                rule_name="系统错误预警",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.CRITICAL,
                conditions={"error_rate": 0.05},
                actions=["email", "sms", "webhook", "log"],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="performance_degradation_001",
                rule_name="性能退化预警",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                alert_level=AlertLevel.WARNING,
                conditions={"response_time": 5.0},
                actions=["email", "log"],
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="compliance_violation_001",
                rule_name="合规违规预警",
                alert_type=AlertType.COMPLIANCE_VIOLATION,
                alert_level=AlertLevel.CRITICAL,
                conditions={"violation_count": 1},
                actions=["email", "sms", "webhook", "log"],
                cooldown_minutes=5
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def add_alert_rule(self, rule: AlertRule):
        """添加预警规则"""
        with self.lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"添加预警规则: {rule.rule_name}")

    def remove_alert_rule(self, rule_id: str):
        """移除预警规则"""
        with self.lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"移除预警规则: {rule_id}")

    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]):
        """更新预警规则"""
        with self.lock:
            if rule_id in self.alert_rules:
                rule = self.alert_rules[rule_id]
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"更新预警规则: {rule_id}")

    def check_alerts(self, data: Dict[str, Any]) -> List[Alert]:
        """检查预警条件"""
        alerts = []
        current_time = datetime.now()

        with self.lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue

                # 检查冷却时间
                if self._is_in_cooldown(rule.rule_id, current_time):
                    continue

                # 检查预警条件
                if self._evaluate_conditions(rule.conditions, data):
                    alert = self._create_alert(rule, data)
                    if alert:
                        alerts.append(alert)
                        self.active_alerts[alert.alert_id] = alert
                        self.alert_history.append(alert)

                        # 触发预警处理器
                        self._trigger_alert_handlers(alert)

                        # 发送通知
                        self._queue_notification(alert)

        return alerts

    def _is_in_cooldown(self, rule_id: str, current_time: datetime) -> bool:
        """检查是否在冷却时间内"""
        # 查找最近的相同规则预警
        for alert in reversed(self.alert_history):
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                rule = self.alert_rules.get(rule_id)
                if rule:
                    cooldown_time = alert.timestamp + timedelta(minutes=rule.cooldown_minutes)
                    if current_time < cooldown_time:
                        return True
        return False

    def _evaluate_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估预警条件"""
        for key, threshold in conditions.items():
            value = data.get(key)
            if value is None:
                continue

            # 数值比较
            if isinstance(threshold, (int, float)):
                if value > threshold:
                    return True
            # 字符串比较
            elif isinstance(threshold, str):
                if value == threshold:
                    return True
            # 列表比较
            elif isinstance(threshold, list):
                if value in threshold:
                    return True

        return False

    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Optional[Alert]:
        """创建预警"""
        alert_id = f"{rule.rule_id}_{int(time.time() * 1000)}"

        # 生成预警标题和消息
        title = f"{rule.alert_level.value.upper()}: {rule.rule_name}"
        message = self._generate_alert_message(rule, data)

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            title=title,
            message=message,
            details=data,
            timestamp=datetime.now()
        )

        logger.warning(f"触发预警: {title} - {message}")
        return alert

    def _generate_alert_message(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """生成预警消息"""
        if rule.alert_type == AlertType.RISK_THRESHOLD:
            risk_score = data.get("risk_score", 0)
            return f"风险分数 {risk_score:.2f} 超过阈值 {rule.conditions.get('risk_score', 0)}"

        elif rule.alert_type == AlertType.POSITION_LIMIT:
            position_ratio = data.get("position_ratio", 0)
            return f"仓位比例 {position_ratio:.2%} 超过限制 {rule.conditions.get('position_ratio', 0):.2%}"

        elif rule.alert_type == AlertType.VOLATILITY_ALERT:
            volatility = data.get("volatility", 0)
            return f"波动率 {volatility:.2%} 超过阈值 {rule.conditions.get('volatility', 0):.2%}"

        elif rule.alert_type == AlertType.LIQUIDITY_ALERT:
            liquidity_ratio = data.get("liquidity_ratio", 0)
            return f"流动性比例 {liquidity_ratio:.2%} 低于阈值 {rule.conditions.get('liquidity_ratio', 0):.2%}"

        elif rule.alert_type == AlertType.SYSTEM_ERROR:
            error_rate = data.get("error_rate", 0)
            return f"系统错误率 {error_rate:.2%} 超过阈值 {rule.conditions.get('error_rate', 0):.2%}"

        elif rule.alert_type == AlertType.PERFORMANCE_DEGRADATION:
            response_time = data.get("response_time", 0)
            return f"响应时间 {response_time:.2f}s 超过阈值 {rule.conditions.get('response_time', 0)}s"

        elif rule.alert_type == AlertType.COMPLIANCE_VIOLATION:
            violation_count = data.get("violation_count", 0)
            return f"合规违规次数 {violation_count} 超过阈值 {rule.conditions.get('violation_count', 0)}"

        else:
            return f"触发预警: {rule.rule_name}"

    def _trigger_alert_handlers(self, alert: Alert):
        """触发预警处理器"""
        handlers = self.alert_handlers.get(alert.alert_type, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"预警处理器异常 {e}")

    def _queue_notification(self, alert: Alert):
        """将通知加入队列"""
        try:
            self.notification_queue.put(alert, timeout=1)
        except queue.Full:
            logger.warning("通知队列已满，丢弃通知")

    def _start_notification_thread(self):
        """启动通知线程"""
        self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.notification_thread.start()
        logger.info("通知线程已启动")

    def _notification_worker(self):
        """通知工作线程"""
        while not self._stop_event.is_set():
            try:
                # 使用stop_event.wait来支持中断
                if self._stop_event.wait(timeout=1):
                    break  # 收到停止信号，退出循环
                alert = self.notification_queue.get(timeout=0.1)
                self._send_notifications(alert)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"通知工作线程异常: {e}")
            if self._stop_event.is_set():
                break  # 如果收到停止信号，退出循环

    def _send_notifications(self, alert: Alert):
        """发送通知"""
        if alert.notification_sent:
            return

        try:
            # 发送邮件通知
            if self.notification_config.email_enabled and self.notification_config.email_recipients:
                self._send_email_notification(alert)

            # 发送短信通知
            if self.notification_config.sms_enabled and self.notification_config.sms_recipients:
                self._send_sms_notification(alert)

            # 发送Webhook通知
            if self.notification_config.webhook_enabled and self.notification_config.webhook_urls:
                self._send_webhook_notification(alert)

            alert.notification_sent = True
            logger.info(f"通知已发送 {alert.alert_id}")

        except Exception as e:
            logger.error(f"发送通知失败: {e}")

    def stop(self):
        """停止预警系统"""
        logger.info("正在停止智能风险预警系统...")

        # 设置停止标志
        self._stop_event.set()

        # 等待通知线程结束
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)
        if self.notification_thread.is_alive():
            logger.warning("通知线程未能5秒内停止")

        # 清空通知队列
        while not self.notification_queue.empty():
            try:
                self.notification_queue.get_nowait()
                self.notification_queue.task_done()
            except queue.Empty:
                break

    logger.info("智能风险预警系统已停止")

    def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        try:
            # 这里应该实现实际的邮件发送逻辑
            # 示例实现
            subject = f"[{alert.alert_level.value.upper()}] {alert.title}"

            # 实际实现中应该使用SMTP发送邮件
            logger.info(f"邮件通知: {subject}")

        except Exception as e:
            logger.error(f"发送邮件通知失败: {e}")

    def _send_sms_notification(self, alert: Alert):
        """发送短信通知"""
        try:
            # 这里应该实现实际的短信发送逻辑
            message = f"[{alert.alert_level.value.upper()}] {alert.title}: {alert.message}"
            logger.info(f"短信通知: {message}")
        except Exception as e:
            logger.error(f"发送短信通知失败: {e}")

    def _send_webhook_notification(self, alert: Alert):
        """发送Webhook通知"""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "alert_level": alert.alert_level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details
            }

            for webhook_url in self.notification_config.webhook_urls:
                response = requests.post(webhook_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Webhook通知发送成功 {webhook_url}")
                else:
                    logger.warning(f"Webhook通知发送失败 {webhook_url}, 状态码: {response.status_code}")

        except Exception as e:
            logger.error(f"发送Webhook通知失败: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认预警"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_time = datetime.now()
                logger.info(f"预警已确认 {alert_id} by {acknowledged_by}")

    def resolve_alert(self, alert_id: str, resolved_by: str):
        """解决预警"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_by = resolved_by
                alert.resolved_time = datetime.now()
                logger.info(f"预警已解决 {alert_id} by {resolved_by}")

    def get_active_alerts(self, alert_type: Optional[AlertType] = None,

                          alert_level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取活跃预警"""
        with self.lock:
            alerts = [alert for alert in self.active_alerts.values()
                      if alert.status == AlertStatus.ACTIVE]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if alert_level:
            alerts = [a for a in alerts if a.alert_level == alert_level]

        return alerts

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取预警历史"""
        with self.lock:
            return list(self.alert_history)[-limit:]

    def add_alert_handler(self, alert_type: AlertType, handler: Callable[[Alert], None]):
        """添加预警处理器"""
        self.alert_handlers[alert_type].append(handler)
        logger.info(f"添加预警处理器 {alert_type.value}")

    def remove_alert_handler(self, alert_type: AlertType, handler: Callable[[Alert], None]):
        """移除预警处理器"""
        if alert_type in self.alert_handlers:
            self.alert_handlers[alert_type].remove(handler)
            logger.info(f"移除预警处理器 {alert_type.value}")

    def get_alert_summary(self) -> Dict[str, Any]:
        """获取预警摘要"""
        with self.lock:
            active_alerts = [a for a in self.active_alerts.values()
                             if a.status == AlertStatus.ACTIVE]

        summary = {
            "total_active_alerts": len(active_alerts),
            "alerts_by_level": defaultdict(int),
            "alerts_by_type": defaultdict(int),
            "recent_alerts": []
        }

        # 按级别统计
        for alert in active_alerts:
            summary["alerts_by_level"][alert.alert_level.value] += 1
            summary["alerts_by_type"][alert.alert_type.value] += 1

        # 最近预警
        summary["recent_alerts"] = [
            {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "level": alert.alert_level.value,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
        ]

        return summary

    def cleanup_expired_alerts(self, max_age_hours: int = 24):
        """清理过期预警"""
        with self.lock:
            current_time = datetime.now()
            expired_alerts = []

        for alert_id, alert in self.active_alerts.items():
            if current_time - alert.timestamp > timedelta(hours=max_age_hours):
                expired_alerts.append(alert_id)

        for alert_id in expired_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.EXPIRED
            logger.info(f"预警已过期 {alert_id}")

        logger.info(f"清理了{len(expired_alerts)} 个过期预警")
