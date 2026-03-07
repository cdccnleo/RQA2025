"""
alert_manager_component 模块

提供 alert_manager_component 相关功能和接口。
"""


import threading
import time

from ..alert_dataclasses import AlertRule, Alert, PerformanceMetrics, TestExecutionInfo
from ..alert_enums import AlertType, AlertLevel
from ..shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
"""
告警管理组件

实现告警规则管理、条件评估和告警触发功能：
- 告警规则配置和管理
- 实时告警条件评估
- 告警处理器注册和调用
- 告警历史记录和状态管理
"""


class MonitoringAlertManager:
    """Alert manager class"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

        # 配置日志和错误处理
        self.logger: ILogger = StandardLogger(f"{self.__class__.__name__}")
        self.error_handler: IErrorHandler = BaseErrorHandler()

        # 应用配置
        if config:
            self._apply_config(config)

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置"""
        # 可以在这里添加配置验证和应用逻辑

    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            # 检查是否已存在相同名称的规则
            if any(r.name == rule.name for r in self.alert_rules):
                self.logger.warning(f"告警规则 '{rule.name}' 已存在，将被替换")
                self.remove_alert_rule(rule.name)
            self.alert_rules.append(rule)
            self.logger.log_info(f"添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        with self._lock:
            original_count = len(self.alert_rules)
            self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
            if len(self.alert_rules) < original_count:
                self.logger.log_info(f"移除告警规则: {rule_name}")

    def register_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Register alert handler"""
        self.alert_handlers[alert_type].append(handler)
        self.logger.log_info(f"注册告警处理器: {alert_type.value}")

    def check_alerts(self, metrics: PerformanceMetrics, test_info: Optional[TestExecutionInfo] = None):
        """Check alert conditions"""
        with self._lock:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue

                # 检查冷却时间
                if (rule.last_triggered and
                        datetime.now() - rule.last_triggered < timedelta(seconds=rule.cooldown)):
                    continue

                # 检查告警条件
                if self._evaluate_condition(rule, metrics, test_info):
                    self._trigger_alert(rule, metrics, test_info)

    def _evaluate_condition(self, rule: AlertRule, metrics: PerformanceMetrics,
                            test_info: Optional[TestExecutionInfo]) -> bool:
        """评估告警条件"""
        try:
            if rule.condition == "cpu_usage > threshold":
                return metrics.cpu_usage > rule.threshold
            elif rule.condition == "memory_usage > threshold":
                return metrics.memory_usage > rule.threshold
            elif rule.condition == "disk_usage > threshold":
                return metrics.disk_usage > rule.threshold
            elif rule.condition == "network_latency > threshold":
                return metrics.network_latency > rule.threshold
            elif rule.condition == "test_execution_time > threshold" and test_info:
                return test_info.execution_time and test_info.execution_time > rule.threshold
            elif rule.condition == "test_success_rate < threshold":
                return metrics.test_success_rate < rule.threshold
            return False
        except Exception as e:
            self.error_handler.handle_error(e, f"告警条件评估错误: {rule.condition}")
            return False

    def _trigger_alert(self, rule: AlertRule, metrics: PerformanceMetrics,
                       test_info: Optional[TestExecutionInfo]):
        """触发告警"""
        alert_id = f"{rule.alert_type.value}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            message=f"触发告警规则: {rule.name}",
            details={
                "rule_name": rule.name,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "current_value": self._get_current_value(rule, metrics, test_info),
                "metrics": {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_usage": metrics.disk_usage,
                    "network_latency": metrics.network_latency
                }
            },
            timestamp=datetime.now(),
            source="performance_monitor"
        )

        with self._lock:
            self.active_alerts[alert_id] = alert
            rule.last_triggered = datetime.now()

        self.logger.warning(f"触发告警: {alert.message} (ID: {alert_id})")

        # 调用告警处理器（异步处理，避免阻塞）
        def call_handlers():
            for handler in self.alert_handlers[rule.alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    self.error_handler.handle_error(e, f"告警处理器执行错误")

        # 在后台线程中调用处理器，避免阻塞主线程
        handler_thread = threading.Thread(target=call_handlers, daemon=True)
        handler_thread.start()

    def _get_current_value(self, rule: AlertRule, metrics: PerformanceMetrics,
                           test_info: Optional[TestExecutionInfo]) -> float:
        """获取当前值"""
        if rule.condition == "cpu_usage > threshold":
            return metrics.cpu_usage
        elif rule.condition == "memory_usage > threshold":
            return metrics.memory_usage
        elif rule.condition == "disk_usage > threshold":
            return metrics.disk_usage
        elif rule.condition == "network_latency > threshold":
            return metrics.network_latency
        elif rule.condition == "test_execution_time > threshold" and test_info:
            return test_info.execution_time or 0.0
        elif rule.condition == "test_success_rate < threshold":
            return metrics.test_success_rate
        return 0.0

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolved_at = datetime.now()
                self.logger.log_info(f"解决告警: {alert_id}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [a for a in self.active_alerts.values() if a.timestamp > cutoff_time]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        with self._lock:
            total_alerts = len(self.active_alerts)
            active_alerts = len([a for a in self.active_alerts.values() if not a.resolved])
            resolved_alerts = total_alerts - active_alerts

            # 按类型统计
            type_stats = {}
            for alert_type in AlertType:
                type_count = len([a for a in self.active_alerts.values()
                                 if a.alert_type == alert_type])
                if type_count > 0:
                    type_stats[alert_type.value] = type_count

            # 按级别统计
            level_stats = {}
            for alert_level in AlertLevel:
                level_count = len([a for a in self.active_alerts.values()
                                  if a.alert_level == alert_level])
                if level_count > 0:
                    level_stats[alert_level.value] = level_count

            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "resolved_alerts": resolved_alerts,
                "type_distribution": type_stats,
                "level_distribution": level_stats
            }
