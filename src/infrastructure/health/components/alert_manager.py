"""
告警管理器

负责告警规则的创建、触发、管理和通知等功能。
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    name: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    labels: Dict[str, str] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"alert_{int(self.timestamp)}"
        if isinstance(self.severity, str):
            try:
                self.severity = AlertSeverity(self.severity.lower())
            except ValueError:
                self.severity = AlertSeverity.WARNING
        elif not isinstance(self.severity, AlertSeverity):
            self.severity = AlertSeverity.WARNING


@dataclass
class AlertRule:
    """告警规则"""

    name: str
    query: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    threshold: float = 0.0
    duration: float = 0.0
    enabled: bool = True
    last_triggered: float = 0.0
    condition: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.severity, str):
            try:
                self.severity = AlertSeverity(self.severity.lower())
            except ValueError:
                self.severity = AlertSeverity.WARNING
        if not self.query and self.condition:
            self.query = self.condition


class AlertManager:
    """
    告警管理器
    
    职责：
    - 告警规则的创建和管理
    - 告警触发和状态跟踪
    - 告警通知和回调
    - 告警数据统计
    """
    
    def __init__(self, alert_timeout: float = 300.0):
        """
        初始化告警管理器
        
        Args:
            alert_timeout: 告警超时时间(秒)
        """
        self._alert_timeout = alert_timeout
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
    def add_alert_rule(self,
                      rule_name: Any,
                      query: str = "",
                      severity: Any = AlertSeverity.WARNING,
                      threshold: float = 0.0,
                      duration: float = 0) -> bool:
        """
        添加告警规则
        
        Args:
            rule_name: 规则名称
            query: 查询条件
            severity: 告警级别
            threshold: 阈值
            duration: 持续时间(秒)
            
        Returns:
            是否添加成功
        """
        try:
            if isinstance(rule_name, dict):
                rule_dict = rule_name
                rule_name = rule_dict.get("name") or rule_dict.get("rule_name") or "unnamed_rule"
                query = rule_dict.get("query") or rule_dict.get("condition", "")
                severity = rule_dict.get("severity", severity)
                threshold = rule_dict.get("threshold", threshold)
                duration = rule_dict.get("duration", duration)

            severity_enum = self._coerce_severity(severity)

            with self._lock:
                rule = AlertRule(
                    name=rule_name,
                    query=query,
                    severity=severity_enum,
                    threshold=threshold,
                    duration=duration
                )
                self._alert_rules[rule_name] = rule
                logger.info(f"添加告警规则: {rule_name}")
                return True
                
        except Exception as e:
            logger.error(f"添加告警规则失败 {rule_name}: {e}")
            return False

    @staticmethod
    def _coerce_severity(severity: Any) -> AlertSeverity:
        if isinstance(severity, AlertSeverity):
            return severity
        if isinstance(severity, str):
            try:
                return AlertSeverity(severity.lower())
            except ValueError:
                return AlertSeverity.WARNING
        return AlertSeverity.WARNING
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        移除告警规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否移除成功
        """
        try:
            with self._lock:
                if rule_name in self._alert_rules:
                    del self._alert_rules[rule_name]
                    logger.info(f"移除告警规则: {rule_name}")
                    return True
                else:
                    logger.warning(f"告警规则不存在: {rule_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"移除告警规则失败 {rule_name}: {e}")
            return False
    
    def get_alert_rules(self) -> Dict[str, AlertRule]:
        """获取所有告警规则"""
        with self._lock:
            return self._alert_rules.copy()
    
    def get_alerts(self, 
                  severity: AlertSeverity = None, 
                  resolved: bool = None) -> List[Alert]:
        """
        获取告警信息
        
        Args:
            severity: 告警级别过滤
            resolved: 是否已解决过滤
            
        Returns:
            告警列表
        """
        try:
            with self._lock:
                filtered_alerts = []
                
                for alert in self._alerts:
                    # 级别过滤
                    if severity and alert.severity != severity:
                        continue
                    
                    # 解决状态过滤
                    if resolved is not None and alert.resolved != resolved:
                        continue
                    
                    filtered_alerts.append(alert)
                
                # 按时间戳倒序排列（最新的在前）
                filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
                return filtered_alerts
                
        except Exception as e:
            logger.error(f"获取告警失败: {e}")
            return []
    
    def resolve_alert(self, alert_name: str) -> bool:
        """
        解决告警
        
        Args:
            alert_name: 告警名称
            
        Returns:
            是否解决成功
        """
        try:
            with self._lock:
                for alert in self._alerts:
                    if alert.name == alert_name and not alert.resolved:
                        alert.resolved = True
                        logger.info(f"告警已解决: {alert_name}")
                        return True
                
                logger.warning(f"未找到未解决的告警: {alert_name}")
                return False
                
        except Exception as e:
            logger.error(f"解决告警失败 {alert_name}: {e}")
            return False
    
    def check_alert_rules(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """
        检查告警规则并生成告警
        
        Args:
            metrics_data: 指标数据
            
        Returns:
            新生成的告警列表
        """
        new_alerts = []
        
        try:
            with self._lock:
                current_time = time.time()
                new_alerts = self._process_alert_rules(metrics_data, current_time)
        except Exception as e:
            logger.error(f"检查告警规则失败: {e}")
        
        return new_alerts

    def _process_alert_rules(self, metrics_data: Dict[str, Any], current_time: float) -> List[Alert]:
        """处理所有告警规则"""
        new_alerts = []
        
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            
            if self._should_process_rule(rule, metrics_data, current_time):
                alert = self._create_and_process_alert(rule, current_time)
                new_alerts.append(alert)
                logger.info(f"触发告警: {rule.name}")
        
        return new_alerts

    def _should_process_rule(self, rule: AlertRule, metrics_data: Dict[str, Any], current_time: float) -> bool:
        """判断是否应该处理规则"""
        if not self._should_trigger_alert(rule, metrics_data):
            return False
        
        # 检查是否在冷却期内
        if current_time - rule.last_triggered < rule.duration:
            return False
        
        return True

    def _create_and_process_alert(self, rule: AlertRule, current_time: float) -> Alert:
        """创建告警并处理相关逻辑"""
        # 创建新告警
        alert = self._create_alert_from_rule(rule)
        
        # 添加到告警列表
        self._alerts.append(alert)
        rule.last_triggered = current_time
        
        # 触发回调
        self._trigger_alert_callbacks(alert)
        
        return alert

    def _create_alert_from_rule(self, rule: AlertRule) -> Alert:
        """从规则创建告警对象"""
        return Alert(
            name=rule.name,
            message=f"指标 {rule.query} 达到阈值 {rule.threshold}",
            severity=rule.severity,
            source="alert_rule",
            labels={"rule_name": rule.name, "threshold": str(rule.threshold)}
        )

    def _trigger_alert_callbacks(self, alert: Alert) -> None:
        """触发告警回调"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def _should_trigger_alert(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> bool:
        """
        判断是否应该触发告警
        
        Args:
            rule: 告警规则
            metrics_data: 指标数据
            
        Returns:
            是否触发告警
        """
        try:
            # 这里实现简单的阈值检查
            # 实际应用中需要实现更复杂的查询解析
            
            # 从指标数据中提取值进行比较
            # 这里假设query是指标名称
            if rule.query in metrics_data:
                value = metrics_data[rule.query]
                if isinstance(value, (int, float)):
                    # 简单的数值比较
                    if isinstance(rule.threshold, (int, float)):
                        return value >= rule.threshold
                        
            return False
            
        except Exception as e:
            logger.error(f"检查告警条件失败: {e}")
            return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        添加告警回调函数
        
        Args:
            callback: 回调函数
        """
        if callback not in self._alert_callbacks:
            self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        移除告警回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    def cleanup_old_alerts(self, max_count: int = 1000) -> int:
        """
        清理旧的告警数据
        
        Args:
            max_count: 最大保留告警数量
            
        Returns:
            清理的告警数量
        """
        try:
            with self._lock:
                if len(self._alerts) <= max_count:
                    return 0
                
                # 按时间戳排序，保留最新的
                self._alerts.sort(key=lambda x: x.timestamp, reverse=True)
                old_count = len(self._alerts)
                self._alerts = self._alerts[:max_count]
                
                cleaned_count = old_count - len(self._alerts)
                if cleaned_count > 0:
                    logger.info(f"清理了 {cleaned_count} 个旧告警")
                
                return cleaned_count
                
        except Exception as e:
            logger.error(f"清理旧告警失败: {e}")
            return 0
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """
        获取最近的告警
        
        Args:
            limit: 数量限制
            
        Returns:
            最近的告警列表
        """
        try:
            with self._lock:
                recent_alerts = sorted(self._alerts, key=lambda x: x.timestamp, reverse=True)
                return recent_alerts[:limit]
                
        except Exception as e:
            logger.error(f"获取最近告警失败: {e}")
            return []
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        try:
            with self._lock:
                total_alerts = len(self._alerts)
                resolved_count = sum(1 for alert in self._alerts if alert.resolved)
                unresolved_count = total_alerts - resolved_count
                
                # 按级别统计
                severity_stats = {}
                for severity in AlertSeverity:
                    count = sum(1 for alert in self._alerts if alert.severity == severity)
                    severity_stats[severity.value] = count
                
                return {
                    "total_alerts": total_alerts,
                    "resolved_count": resolved_count,
                    "unresolved_count": unresolved_count,
                    "severity_stats": severity_stats,
                    "total_rules": len(self._alert_rules),
                    "active_rules": sum(1 for rule in self._alert_rules.values() if rule.enabled)
                }
                
        except Exception as e:
            logger.error(f"获取告警统计失败: {e}")
            return {}
