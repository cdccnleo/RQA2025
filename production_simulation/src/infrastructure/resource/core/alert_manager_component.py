"""
alert_manager_component 模块

提供 alert_manager_component 相关功能和接口。
"""


import threading
import time

from ..models.alert_dataclasses import AlertRule, Alert, PerformanceMetrics, TestExecutionInfo
from ..models.alert_enums import AlertType, AlertLevel
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from ..models.parameter_objects import AlertCreationParameters, AlertCheckParameters
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


class AlertManager:
    """Alert manager class"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

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
                # 直接移除而不调用remove_alert_rule，避免重入锁问题
                self._remove_alert_rule_unsafe(rule.name)
            self.alert_rules.append(rule)
            self.logger.log_info(f"添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        with self._lock:
            self._remove_alert_rule_unsafe(rule_name)

    def _remove_alert_rule_unsafe(self, rule_name: str):
        """Remove alert rule without acquiring lock (internal use only)
        
        Note: This method assumes the caller already holds self._lock
        to avoid deadlock situations when called from within locked methods.
        """
        original_count = len(self.alert_rules)
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        if len(self.alert_rules) < original_count:
            self.logger.log_info(f"移除告警规则: {rule_name}")

    def register_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Register alert handler"""
        self.alert_handlers[alert_type].append(handler)
        self.logger.log_info(f"注册告警处理器: {alert_type.value}")

    def check_alerts(self, metrics_or_params = None, 
                     test_info: Optional[TestExecutionInfo] = None):
        """Check alert conditions using parameter object or legacy parameters"""
        # 支持向后兼容：第一个参数可能是AlertCheckParameters或PerformanceMetrics
        if metrics_or_params is None:
            raise ValueError("必须提供check_params或metrics参数")
        
        # 判断第一个参数的类型
        if hasattr(metrics_or_params, 'include_enabled_check'):
            # 是AlertCheckParameters对象
            check_params = metrics_or_params
        else:
            # 是PerformanceMetrics对象或类似对象，创建AlertCheckParameters
            from ..models.parameter_objects import AlertCheckParameters
            check_params = AlertCheckParameters(metrics=metrics_or_params, test_info=test_info)
        
        # 确保check_params对象已正确初始化
        assert hasattr(check_params, 'include_enabled_check'), f"check_params missing attribute: {type(check_params)}"
        
        with self._lock:
            for rule in self.alert_rules:
                if check_params.include_enabled_check and not rule.enabled:
                    continue

                # 检查冷却时间
                if (check_params.include_cooldown_check and 
                    rule.last_triggered and
                    datetime.now() - rule.last_triggered < timedelta(seconds=rule.cooldown)):
                    continue

                # 检查告警条件
                if self._evaluate_condition(rule, check_params.metrics, check_params.test_info):
                    self._trigger_alert_with_params_unsafe(rule, check_params)
                    
                # 检查最大告警数量限制
                if (check_params.max_alerts and 
                    len(self.active_alerts) >= check_params.max_alerts):
                    break

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

    def _trigger_alert_with_params(self, rule: AlertRule, check_params: AlertCheckParameters):
        """使用参数对象触发告警"""
        alert_params = AlertCreationParameters(
            id=f"{rule.alert_type.value}_{int(time.time())}",
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            message=f"触发告警规则: {rule.name}",
            details=self._build_alert_details(rule, check_params),
            timestamp=datetime.now(),
            source="performance_monitor",
            rule_name=rule.name,
            condition=rule.condition,
            threshold=rule.threshold,
            current_value=self._get_current_value(rule, check_params.metrics, check_params.test_info)
        )
        
        self._create_and_trigger_alert(rule, alert_params)

    def _trigger_alert_with_params_unsafe(self, rule: AlertRule, check_params: AlertCheckParameters):
        """使用参数对象触发告警（无锁版本，假设调用者已持有锁）"""
        alert_params = AlertCreationParameters(
            id=f"{rule.alert_type.value}_{int(time.time())}",
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            message=f"触发告警规则: {rule.name}",
            details=self._build_alert_details(rule, check_params),
            timestamp=datetime.now(),
            source="performance_monitor",
            rule_name=rule.name,
            condition=rule.condition,
            threshold=rule.threshold,
            current_value=self._get_current_value(rule, check_params.metrics, check_params.test_info)
        )
        
        alert = Alert(
            id=alert_params.id,
            alert_type=alert_params.alert_type,
            alert_level=alert_params.alert_level,
            message=alert_params.message,
            details=alert_params.details,
            timestamp=alert_params.timestamp,
            source=alert_params.source
        )

        self._create_and_trigger_alert_unsafe(rule, alert, alert_params)

    def _trigger_alert(self, rule: AlertRule, metrics: PerformanceMetrics,
                       test_info: Optional[TestExecutionInfo]):
        """触发告警（向后兼容方法）"""
        check_params = AlertCheckParameters(metrics=metrics, test_info=test_info)
        self._trigger_alert_with_params(rule, check_params)

    def _build_alert_details(self, rule: AlertRule, check_params: AlertCheckParameters) -> Dict[str, Any]:
        """构建告警详情"""
        return {
            "rule_name": rule.name,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "current_value": self._get_current_value(rule, check_params.metrics, check_params.test_info),
            "metrics": {
                "cpu_usage": check_params.metrics.cpu_usage,
                "memory_usage": check_params.metrics.memory_usage,
                "disk_usage": check_params.metrics.disk_usage,
                "network_latency": check_params.metrics.network_latency
            }
        }

    def _create_and_trigger_alert(self, rule: AlertRule, alert_params: AlertCreationParameters):
        """创建并触发告警"""
        alert = Alert(
            id=alert_params.id,
            alert_type=alert_params.alert_type,
            alert_level=alert_params.alert_level,
            message=alert_params.message,
            details=alert_params.details,
            timestamp=alert_params.timestamp,
            source=alert_params.source
        )

        with self._lock:
            self._create_and_trigger_alert_unsafe(rule, alert, alert_params)

    def _create_and_trigger_alert_unsafe(self, rule: AlertRule, alert: Alert, alert_params: AlertCreationParameters):
        """创建并触发告警（无锁版本，假设调用者已持有锁）"""
        self.active_alerts[alert_params.id] = alert
        rule.last_triggered = datetime.now()

        self.logger.warning(f"触发告警: {alert.message} (ID: {alert_params.id})")

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

    def cleanup(self):
        """清理所有告警规则、活跃告警和处理器"""
        with self._lock:
            self.alert_rules.clear()
            self.active_alerts.clear()
            self.alert_handlers.clear()
            self.logger.log_info("告警管理器已清理")
