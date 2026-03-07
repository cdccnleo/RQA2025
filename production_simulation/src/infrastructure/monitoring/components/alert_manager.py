#!/usr/bin/env python3
"""
RQA2025 基础设施层告警管理器

负责监控告警规则的评估、触发和通知处理。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .performance_monitor import monitor_performance

from ..core.parameter_objects import AlertRuleConfig, AlertConditionConfig


logger = logging.getLogger(__name__)


class AlertManager:
    """
    告警管理器

    负责管理和执行告警规则，支持多种告警条件和通知渠道。
    """

    def __init__(
        self,
        pool_name: str = "default_pool",
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        初始化告警管理器

        Args:
            pool_name: 池名称
            alert_thresholds: 告警阈值配置
        """
        self.pool_name = pool_name
        default_thresholds = {
            "hit_rate_low": 0.8,
            "pool_usage_high": 0.9,
            "max_pool_size": 100,
            "memory_high": 100.0,
        }
        incoming_thresholds = dict(alert_thresholds or {})
        self.alert_thresholds = incoming_thresholds
        self._thresholds = {**default_thresholds, **incoming_thresholds}

        # 兼容性配置项
        self.config: Dict[str, Any] = {
            'max_active_alerts': 100,
            'auto_resolve_timeout': 3600,
        }

        # 告警规则
        self.alert_rules: List[AlertRuleConfig] = []
        self._init_default_rules()

        # 告警历史
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # 冷却时间跟踪
        self.last_alert_times: Dict[str, datetime] = {}

    def add_alert_rule(self, rule: AlertRuleConfig):
        """
        添加告警规则

        Args:
            rule: 告警规则配置
        """
        self.alert_rules.append(rule)

    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        移除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功移除
        """
        for i, rule in enumerate(self.alert_rules):
            if rule.rule_id == rule_id:
                self.alert_rules.pop(i)
                return True
        return False

    @monitor_performance("AlertManager", "check_alerts")
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查告警条件

        Args:
            stats: 统计信息

        Returns:
            List[Dict[str, Any]]: 触发的告警列表
        """
        triggered_alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            if self._should_trigger_alert(rule, stats):
                alert = self._create_alert(rule, stats)
                if alert:
                    triggered_alerts.append(alert)
                    self._record_alert(alert)

        return triggered_alerts

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取告警历史

        Args:
            limit: 返回的最大记录数

        Returns:
            List[Dict[str, Any]]: 告警历史列表
        """
        if limit == 0:
            return []

        history = self.alert_history if limit < 0 else self.alert_history[-limit:]
        return list(reversed(history))

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取活跃告警

        Returns:
            List[Dict[str, Any]]: 活跃告警列表
        """
        active_alerts: List[Dict[str, Any]] = []
        for alert in self.alert_history:
            status = alert.get('status', 'active')
            is_active = alert.get('active')
            if is_active is None:
                is_active = status == 'active'
            if is_active and not alert.get('acknowledged', False):
                active_alerts.append(alert)
        max_active = self.config.get('max_active_alerts')
        if isinstance(max_active, int) and max_active > 0:
            active_alerts = active_alerts[-max_active:]
        return active_alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功确认
        """
        for alert in self.alert_history:
            if alert.get('alert_id') == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged'] = True
                alert['active'] = False
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False

    def resolve_alert(self, alert_id: str, resolution: Optional[str] = None) -> bool:
        """
        解决告警。

        Args:
            alert_id: 告警ID
            resolution: 解决说明

        Returns:
            bool: 是否成功标记
        """
        for alert in self.alert_history:
            if alert.get('alert_id') == alert_id:
                alert['status'] = 'resolved'
                alert['active'] = False
                alert['resolved_at'] = datetime.now().isoformat()
                if resolution is not None:
                    alert['resolution'] = resolution
                return True
        return False

    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRuleConfig(
                rule_id="hit_rate_low",
                name="命中率过低",
                description="Logger池命中率低于阈值",
                condition=AlertConditionConfig(
                    operator="lt",
                    field="hit_rate",
                    value=self._thresholds.get('hit_rate_low', 0.8)
                ),
                severity="warning",
                channels=["console"],
                cooldown=300
            ),
            AlertRuleConfig(
                rule_id="pool_usage_high",
                name="池使用率过高",
                description="Logger池使用率过高",
                condition=AlertConditionConfig(
                    operator="gt",
                    field="pool_size",
                    value=self._thresholds.get('pool_usage_high', 0.9),
                    threshold=self._thresholds.get('max_pool_size', 100)
                ),
                severity="warning",
                channels=["console"],
                cooldown=300
            ),
            AlertRuleConfig(
                rule_id="memory_high",
                name="内存使用过高",
                description="内存使用超过阈值",
                condition=AlertConditionConfig(
                    operator="gt",
                    field="memory_usage_mb",
                    value=self._thresholds.get('memory_high', 100.0)
                ),
                severity="error",
                channels=["console"],
                cooldown=600
            )
        ]

        self.alert_rules.extend(default_rules)

    def _should_trigger_alert(self, rule: AlertRuleConfig, stats: Dict[str, Any]) -> bool:
        """
        判断是否应该触发告警

        Args:
            rule: 告警规则
            stats: 统计信息

        Returns:
            bool: 是否触发告警
        """
        try:
            # 检查冷却时间
            cooldown_seconds = getattr(rule, "cooldown", 0) or 0
            if not self._is_cooldown_expired(rule.rule_id, cooldown_seconds):
                return False

            # 评估告警条件
            conditions = list(rule.conditions or [])
            if rule.condition and not conditions:
                conditions = [rule.condition]

            if not conditions:
                logger.warning(f"告警规则 {rule.rule_id} 未配置条件，跳过评估")
                return False

            return all(self._evaluate_condition(condition, stats) for condition in conditions)

        except Exception as e:
            logger.error(f"评估告警规则失败 {rule.rule_id}: {e}")
            return False

    def _evaluate_condition(self, condition: AlertConditionConfig, stats: Dict[str, Any]) -> bool:
        """
        评估告警条件

        Args:
            condition: 告警条件
            stats: 统计信息

        Returns:
            bool: 条件是否满足
        """
        if condition.field not in stats:
            return False

        actual_value = stats[condition.field]

        operator = condition.operator if isinstance(condition.operator, str) else "eq"
        operator = operator.lower()

        comparator_map = {
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
        }

        comparator = comparator_map.get(operator)
        if comparator is None:
            logger.warning(f"不支持的操作符: {condition.operator}")
            return False

        try:
            return comparator(actual_value, condition.value)
        except Exception:
            logger.error("告警条件比较失败", exc_info=True)
            return False

    def _create_alert(self, rule: AlertRuleConfig, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建告警

        Args:
            rule: 告警规则
            stats: 统计信息

        Returns:
            Dict[str, Any]: 告警信息
        """
        message = rule.description or rule.name
        primary_condition = None
        if getattr(rule, "conditions", None):
            primary_condition = rule.conditions[0]
        elif rule.condition is not None:
            primary_condition = rule.condition

        if isinstance(primary_condition, AlertConditionConfig):
            field_name = primary_condition.field
            if field_name in stats:
                message = f"{message} (当前{field_name}={stats[field_name]!r})"

        alert = {
            'alert_id': f"{rule.rule_id}_{int(datetime.now().timestamp())}",
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'description': rule.description,
            'message': message,
            'level': rule.level,
            'severity': getattr(rule, "severity", rule.level),
            'status': 'active',
            'active': True,
            'pool_name': self.pool_name,
            'triggered_at': datetime.now().isoformat(),
            'stats': stats.copy(),
            'channels': rule.channels.copy(),
            'acknowledged': False
        }

        return alert

    def _record_alert(self, alert: Dict[str, Any]):
        """
        记录告警

        Args:
            alert: 告警信息
        """
        self.alert_history.append(alert)

        # 更新最后告警时间
        self.last_alert_times[alert['rule_id']] = datetime.now()

        # 限制历史记录大小
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        # 记录日志
        rule_name = alert.get('rule_name', alert.get('rule_id', 'unknown'))
        description = alert.get('description') or alert.get('message', '')
        logger.warning(f"告警触发: {rule_name} - {description}")
        self._enforce_active_capacity()

    def _enforce_active_capacity(self) -> None:
        """确保活跃告警数量不超过配置的上限"""
        max_active = self.config.get('max_active_alerts')
        if not isinstance(max_active, int) or max_active <= 0:
            return

        active_alerts = [alert for alert in self.alert_history if alert.get('active')]
        overflow = len(active_alerts) - max_active
        if overflow <= 0:
            return

        for alert in active_alerts[:overflow]:
            alert['active'] = False
            if alert.get('status') == 'active':
                alert['status'] = 'archived'

    def _is_cooldown_expired(self, rule_id: str, cooldown_seconds: int) -> bool:
        """
        检查冷却时间是否已过期

        Args:
            rule_id: 规则ID
            cooldown_seconds: 冷却时间（秒）

        Returns:
            bool: 冷却时间是否已过期
        """
        last_alert_time = self.last_alert_times.get(rule_id)
        if not last_alert_time:
            return True

        elapsed = datetime.now() - last_alert_time
        return elapsed.total_seconds() >= cooldown_seconds

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计信息

        Returns:
            Dict[str, Any]: 告警统计
        """
        total_alerts = len(self.alert_history)
        active_alerts = sum(
            1
            for alert in self.alert_history
            if alert.get('active', alert.get('status', 'active') == 'active')
        )
        acknowledged_alerts = sum(1 for alert in self.alert_history if alert.get('acknowledged'))

        level_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        rule_counts: Dict[str, int] = {}

        for alert in self.alert_history:
            level = alert.get('level', 'unknown')
            severity = alert.get('severity', level)
            rule_id = alert.get('rule_id', 'unknown')

            level_counts[level] = level_counts.get(level, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'acknowledged_alerts': acknowledged_alerts,
            'level_distribution': level_counts,
            'severity_breakdown': severity_counts,
            'rule_distribution': rule_counts,
            'generated_at': datetime.now().isoformat()
        }