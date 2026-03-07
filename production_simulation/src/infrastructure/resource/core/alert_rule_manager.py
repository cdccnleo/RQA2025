
import time

from .alert_dataclasses import AlertRule, Alert, AlertPerformanceMetrics
from .alert_enums import AlertType, AlertLevel
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from ...models.parameter_objects import AlertRuleParameters, AlertCreationParameters, AlertCheckParameters
from typing import Dict, List, Optional, Any
"""
告警规则管理器

负责告警规则的添加、移除、查询和管理，以及告警条件的检查
"""


class AlertRuleManager:
    """告警规则管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        self.config = config or {}
        self.alert_rules: List[AlertRule] = []

        # 初始化默认规则
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认告警规则"""
        try:
            default_rule_params = [
                AlertRuleParameters(
                    name="high_cpu_usage",
                    alert_type=AlertType.PERFORMANCE_DEGRADATION.value,
                    alert_level=AlertLevel.WARNING.value,
                    condition="cpu_usage > 80.0",
                    threshold=80.0
                ),
                AlertRuleParameters(
                    name="high_memory_usage",
                    alert_type=AlertType.PERFORMANCE_DEGRADATION.value,
                    alert_level=AlertLevel.WARNING.value,
                    condition="memory_usage > 85.0",
                    threshold=85.0
                ),
                AlertRuleParameters(
                    name="critical_memory_usage",
                    alert_type=AlertType.PERFORMANCE_DEGRADATION.value,
                    alert_level=AlertLevel.CRITICAL.value,
                    condition="memory_usage > 95.0",
                    threshold=95.0
                ),
                AlertRuleParameters(
                    name="low_disk_space",
                    alert_type=AlertType.RESOURCE_EXHAUSTION.value,
                    alert_level=AlertLevel.WARNING.value,
                    condition="disk_usage > 90.0",
                    threshold=90.0
                )
            ]

            for rule_params in default_rule_params:
                rule = self._create_alert_rule_from_params(rule_params)
                self.add_alert_rule(rule)

            self.logger.log_info(f"已设置 {len(default_rule_params)} 个默认告警规则")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "设置默认告警规则失败"})

    def _create_alert_rule_from_params(self, params: AlertRuleParameters) -> AlertRule:
        """从参数对象创建告警规则"""
        return AlertRule(
            name=params.name,
            alert_type=AlertType(params.alert_type),
            alert_level=AlertLevel(params.alert_level),
            condition=params.condition,
            threshold=params.threshold,
            enabled=params.enabled,
            cooldown=params.cooldown,
            description=params.description
        )

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        try:
            # 检查规则名称是否已存在
            if any(r.name == rule.name for r in self.alert_rules):
                raise ValueError(f"告警规则 '{rule.name}' 已存在")

            self.alert_rules.append(rule)
            self.logger.log_info(f"告警规则 '{rule.name}' 已添加")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "添加告警规则失败", "rule_name": rule.name})

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        try:
            original_count = len(self.alert_rules)
            self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]

            if len(self.alert_rules) < original_count:
                self.logger.log_info(f"告警规则 '{rule_name}' 已移除")
            else:
                raise ValueError(f"告警规则 '{rule_name}' 不存在")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "移除告警规则失败", "rule_name": rule_name})

    def get_alert_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        return self.alert_rules.copy()

    def get_enabled_rules(self) -> List[AlertRule]:
        """获取启用的告警规则"""
        return [rule for rule in self.alert_rules if rule.enabled]

    def enable_rule(self, rule_name: str):
        """启用告警规则"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.log_info(f"告警规则 '{rule_name}' 已启用")
                return
        raise ValueError(f"告警规则 '{rule_name}' 不存在")

    def disable_rule(self, rule_name: str):
        """禁用告警规则"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.log_info(f"告警规则 '{rule_name}' 已禁用")
                return
        raise ValueError(f"告警规则 '{rule_name}' 不存在")

    def check_alerts(self, check_params: AlertCheckParameters = None, 
                     metrics: AlertPerformanceMetrics = None) -> List[Alert]:
        """检查告警条件，返回触发的告警列表"""
        # 支持向后兼容
        if metrics is not None:
            check_params = AlertCheckParameters(metrics=metrics)
        elif check_params is None:
            raise ValueError("必须提供check_params或metrics参数")
        
        alerts = []

        try:
            enabled_rules = self.get_enabled_rules()

            for rule in enabled_rules:
                try:
                    if self._evaluate_condition(rule.condition, check_params.metrics):
                        alert_params = AlertCreationParameters(
                            id=f"{rule.name}_{int(time.time())}",
                            alert_type=rule.alert_type,
                            alert_level=rule.alert_level,
                            message=f"告警规则 '{rule.name}' 被触发 - 阈值: {rule.threshold}",
                            details=self._build_alert_details(rule, check_params.metrics),
                            timestamp=time.time(),
                            source="AlertRuleManager",
                            rule_name=rule.name,
                            threshold=rule.threshold,
                            condition=rule.condition
                        )
                        
                        alert = self._create_alert_from_params(alert_params)
                        alerts.append(alert)
                        
                        # 检查最大告警数量限制
                        if (check_params.max_alerts and 
                            len(alerts) >= check_params.max_alerts):
                            break

                except Exception as e:
                    self.error_handler.handle_error(e, {
                        "context": "检查告警规则异常",
                        "rule_name": rule.name
                    })

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "告警检查过程异常"})

        return alerts

    def _build_alert_details(self, rule: AlertRule, metrics: AlertPerformanceMetrics) -> Dict[str, Any]:
        """构建告警详情"""
        return {
            "rule": rule.name,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "current_value": self._get_metric_value(metrics, rule)
        }

    def _create_alert_from_params(self, alert_params: AlertCreationParameters) -> Alert:
        """从参数对象创建Alert"""
        return Alert(
            alert_id=alert_params.id,
            alert_type=alert_params.alert_type,
            alert_level=alert_params.alert_level,
            message=alert_params.message,
            details=alert_params.details,
            timestamp=alert_params.timestamp,
            source=alert_params.source
        )

    def _evaluate_condition(self, condition: str, metrics: AlertPerformanceMetrics) -> bool:
        """评估告警条件"""
        try:
            # 简单的条件评估，将字符串条件转换为实际比较
            if "cpu_usage >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return metrics.cpu_usage > threshold
            elif "memory_usage >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return metrics.memory_usage > threshold
            elif "disk_usage >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return getattr(metrics, 'disk_usage', 0) > threshold
            else:
                return False
        except Exception:
            return False

    def _get_metric_value(self, metrics: AlertPerformanceMetrics, rule: AlertRule) -> float:
        """获取指标值用于告警信息"""
        if "cpu" in rule.name.lower():
            return metrics.cpu_usage
        elif "memory" in rule.name.lower():
            return metrics.memory_usage
        elif "disk" in rule.name.lower():
            return getattr(metrics, 'disk_usage', 0.0)
        else:
            return 0.0

    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        total_rules = len(self.alert_rules)
        enabled_rules = len(self.get_enabled_rules())

        rule_types = {}
        for rule in self.alert_rules:
            rule_type = rule.alert_type.value
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "rule_types": rule_types
        }
