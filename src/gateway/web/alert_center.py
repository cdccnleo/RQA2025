"""
告警通知中心模块
提供多渠道告警通知、告警生命周期管理、告警抑制和升级等功能
符合量化交易系统合规要求
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"      # 待处理
    ACKNOWLEDGED = "acknowledged"  # 已确认
    RESOLVED = "resolved"    # 已解决
    SUPPRESSED = "suppressed"  # 已抑制


class NotificationChannel(Enum):
    """通知渠道"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"


@dataclass
class Alert:
    """告警对象"""
    alert_id: str
    strategy_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    source: str  # 告警来源（risk_control, anomaly_detection等）
    created_at: float
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    source: str  # 告警来源
    severity: AlertSeverity
    channels: List[NotificationChannel]
    suppress_duration: int = 300  # 抑制持续时间（秒）
    escalation_enabled: bool = False
    escalation_delay: int = 600  # 升级延迟（秒）
    enabled: bool = True


class AlertCenter:
    """
    告警通知中心
    
    职责：
    1. 管理告警规则
    2. 多渠道通知（WebSocket、邮件、短信）
    3. 告警生命周期管理
    4. 告警抑制和升级
    
    参考架构：风险控制层 AlertSystem
    """
    
    def __init__(self):
        # 告警规则
        self._alert_rules: Dict[str, AlertRule] = {}
        
        # 告警存储
        self._alerts: Dict[str, Alert] = {}
        self._strategy_alerts: Dict[str, List[str]] = {}  # strategy_id -> alert_ids
        
        # 抑制记录
        self._suppressed_alerts: Dict[str, float] = {}  # alert_key -> suppress_until
        
        # 通知渠道
        self._notifiers: Dict[NotificationChannel, Callable] = {}
        
        # 回调函数
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 初始化默认规则
        self._init_default_rules()
        
        # 初始化通知渠道
        self._init_notifiers()
        
        logger.info("告警通知中心初始化完成")
        
    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="critical_risk",
                source="risk_control",
                severity=AlertSeverity.CRITICAL,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                suppress_duration=60,
                escalation_enabled=True,
                escalation_delay=300
            ),
            AlertRule(
                rule_id="high_risk",
                source="risk_control",
                severity=AlertSeverity.HIGH,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                suppress_duration=300,
                escalation_enabled=False
            ),
            AlertRule(
                rule_id="anomaly_critical",
                source="anomaly_detection",
                severity=AlertSeverity.CRITICAL,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                suppress_duration=60,
                escalation_enabled=True,
                escalation_delay=300
            ),
            AlertRule(
                rule_id="anomaly_high",
                source="anomaly_detection",
                severity=AlertSeverity.HIGH,
                channels=[NotificationChannel.WEBSOCKET],
                suppress_duration=300,
                escalation_enabled=False
            ),
            AlertRule(
                rule_id="medium_anomaly",
                source="anomaly_detection",
                severity=AlertSeverity.MEDIUM,
                channels=[NotificationChannel.WEBSOCKET],
                suppress_duration=600,
                escalation_enabled=False
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.rule_id] = rule
            
    def _init_notifiers(self):
        """初始化通知渠道"""
        self._notifiers[NotificationChannel.WEBSOCKET] = self._send_websocket_notification
        self._notifiers[NotificationChannel.EMAIL] = self._send_email_notification
        self._notifiers[NotificationChannel.SMS] = self._send_sms_notification
        
    def create_alert(
        self,
        strategy_id: str,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        创建告警
        
        Args:
            strategy_id: 策略ID
            title: 告警标题
            message: 告警消息
            severity: 严重程度
            source: 告警来源
            metadata: 元数据
            
        Returns:
            Alert: 创建的告警对象，如果被抑制则返回None
        """
        # 检查抑制
        alert_key = f"{strategy_id}:{source}:{severity.value}"
        if self._is_suppressed(alert_key):
            logger.info(f"告警被抑制: {alert_key}")
            return None
            
        # 生成告警ID
        alert_id = f"{strategy_id}_{source}_{int(time.time() * 1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            strategy_id=strategy_id,
            title=title,
            message=message,
            severity=severity,
            status=AlertStatus.PENDING,
            source=source,
            created_at=time.time(),
            metadata=metadata or {}
        )
        
        # 保存告警
        self._alerts[alert_id] = alert
        if strategy_id not in self._strategy_alerts:
            self._strategy_alerts[strategy_id] = []
        self._strategy_alerts[strategy_id].append(alert_id)
        
        # 设置抑制
        self._set_suppression(alert_key, severity)
        
        logger.warning(f"告警创建: [{severity.value.upper()}] {title} - {message}")
        
        # 发送通知
        self._send_notifications(alert)
        
        # 触发回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
                
        return alert
        
    def acknowledge_alert(
        self, 
        alert_id: str, 
        acknowledged_by: str,
        note: Optional[str] = None
    ) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人
            note: 备注
            
        Returns:
            bool: 是否成功
        """
        if alert_id not in self._alerts:
            logger.warning(f"告警不存在: {alert_id}")
            return False
            
        alert = self._alerts[alert_id]
        
        if alert.status != AlertStatus.PENDING:
            logger.warning(f"告警状态不正确，无法确认: {alert_id}, 当前状态: {alert.status.value}")
            return False
            
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.acknowledged_by = acknowledged_by
        
        if note:
            alert.metadata["acknowledgment_note"] = note
            
        logger.info(f"告警已确认: {alert_id} by {acknowledged_by}")
        return True
        
    def resolve_alert(
        self, 
        alert_id: str, 
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """
        解决告警
        
        Args:
            alert_id: 告警ID
            resolved_by: 解决人
            resolution_note: 解决备注
            
        Returns:
            bool: 是否成功
        """
        if alert_id not in self._alerts:
            logger.warning(f"告警不存在: {alert_id}")
            return False
            
        alert = self._alerts[alert_id]
        
        if alert.status == AlertStatus.RESOLVED:
            logger.warning(f"告警已解决: {alert_id}")
            return False
            
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()
        alert.resolved_by = resolved_by
        alert.resolution_note = resolution_note
        
        logger.info(f"告警已解决: {alert_id} by {resolved_by}")
        return True
        
    def suppress_alert(self, alert_id: str, duration: int) -> bool:
        """
        手动抑制告警
        
        Args:
            alert_id: 告警ID
            duration: 抑制持续时间（秒）
            
        Returns:
            bool: 是否成功
        """
        if alert_id not in self._alerts:
            return False
            
        alert = self._alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        
        # 设置抑制
        alert_key = f"{alert.strategy_id}:{alert.source}:{alert.severity.value}"
        self._suppressed_alerts[alert_key] = time.time() + duration
        
        logger.info(f"告警已抑制: {alert_id}, 持续时间: {duration}秒")
        return True
        
    def _is_suppressed(self, alert_key: str) -> bool:
        """检查是否被抑制"""
        if alert_key not in self._suppressed_alerts:
            return False
            
        suppress_until = self._suppressed_alerts[alert_key]
        if time.time() > suppress_until:
            # 抑制已过期
            del self._suppressed_alerts[alert_key]
            return False
            
        return True
        
    def _set_suppression(self, alert_key: str, severity: AlertSeverity):
        """设置抑制"""
        # 根据严重程度获取抑制时间
        suppress_duration = self._get_suppress_duration(severity)
        self._suppressed_alerts[alert_key] = time.time() + suppress_duration
        
    def _get_suppress_duration(self, severity: AlertSeverity) -> int:
        """获取抑制持续时间"""
        durations = {
            AlertSeverity.CRITICAL: 60,    # 1分钟
            AlertSeverity.HIGH: 300,       # 5分钟
            AlertSeverity.MEDIUM: 600,     # 10分钟
            AlertSeverity.LOW: 1800        # 30分钟
        }
        return durations.get(severity, 300)
        
    def _send_notifications(self, alert: Alert):
        """发送通知"""
        # 查找匹配的告警规则
        matching_rules = [
            rule for rule in self._alert_rules.values()
            if rule.source == alert.source 
            and rule.severity == alert.severity
            and rule.enabled
        ]
        
        if not matching_rules:
            # 使用默认规则
            channels = [NotificationChannel.WEBSOCKET]
        else:
            # 合并所有匹配规则的渠道
            channels = set()
            for rule in matching_rules:
                channels.update(rule.channels)
                
        # 发送通知
        for channel in channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                try:
                    notifier(alert)
                except Exception as e:
                    logger.error(f"发送{channel.value}通知失败: {e}")
                    
    def _send_websocket_notification(self, alert: Alert):
        """发送WebSocket通知"""
        # TODO: 实现WebSocket通知
        logger.info(f"[WebSocket] 告警通知: [{alert.severity.value}] {alert.title}")
        
    def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        # TODO: 实现邮件通知
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            logger.info(f"[Email] 告警通知: [{alert.severity.value}] {alert.title}")
            
    def _send_sms_notification(self, alert: Alert):
        """发送短信通知"""
        # TODO: 实现短信通知
        if alert.severity == AlertSeverity.CRITICAL:
            logger.info(f"[SMS] 告警通知: [{alert.severity.value}] {alert.title}")
            
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        return self._alerts.get(alert_id)
        
    def get_strategy_alerts(
        self, 
        strategy_id: str,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """获取策略的告警列表"""
        if strategy_id not in self._strategy_alerts:
            return []
            
        alert_ids = self._strategy_alerts[strategy_id]
        alerts = [self._alerts[aid] for aid in alert_ids if aid in self._alerts]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
        
    def get_all_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Alert]:
        """获取所有告警"""
        alerts = list(self._alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
            
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        total = len(self._alerts)
        pending = sum(1 for a in self._alerts.values() if a.status == AlertStatus.PENDING)
        acknowledged = sum(1 for a in self._alerts.values() if a.status == AlertStatus.ACKNOWLEDGED)
        resolved = sum(1 for a in self._alerts.values() if a.status == AlertStatus.RESOLVED)
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1 for a in self._alerts.values() if a.severity == severity
            )
            
        return {
            "total_alerts": total,
            "pending": pending,
            "acknowledged": acknowledged,
            "resolved": resolved,
            "severity_counts": severity_counts,
            "suppressed_keys": len(self._suppressed_alerts)
        }
        
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._alert_rules[rule.rule_id] = rule
        logger.info(f"告警规则已添加: {rule.rule_id}")
        
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"告警规则已移除: {rule_id}")
            
    def enable_alert_rule(self, rule_id: str):
        """启用告警规则"""
        if rule_id in self._alert_rules:
            self._alert_rules[rule_id].enabled = True
            
    def disable_alert_rule(self, rule_id: str):
        """禁用告警规则"""
        if rule_id in self._alert_rules:
            self._alert_rules[rule_id].enabled = False
            
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self._alert_callbacks.append(callback)
        
    def remove_alert_callback(self, callback: Callable[[Alert], None]):
        """移除告警回调"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
            
    def cleanup_old_alerts(self, max_age_days: int = 30):
        """清理旧告警"""
        cutoff = time.time() - (max_age_days * 24 * 3600)
        old_alert_ids = [
            aid for aid, alert in self._alerts.items()
            if alert.created_at < cutoff and alert.status == AlertStatus.RESOLVED
        ]
        
        for alert_id in old_alert_ids:
            alert = self._alerts[alert_id]
            del self._alerts[alert_id]
            
            # 从策略告警列表中移除
            if alert.strategy_id in self._strategy_alerts:
                if alert_id in self._strategy_alerts[alert.strategy_id]:
                    self._strategy_alerts[alert.strategy_id].remove(alert_id)
                    
        logger.info(f"清理了 {len(old_alert_ids)} 条旧告警")
        return len(old_alert_ids)


# 全局告警中心实例
_alert_center: Optional[AlertCenter] = None


def get_alert_center() -> AlertCenter:
    """获取全局告警中心实例（单例模式）"""
    global _alert_center
    if _alert_center is None:
        _alert_center = AlertCenter()
    return _alert_center
