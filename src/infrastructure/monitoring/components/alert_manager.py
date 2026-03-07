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

        # 测试覆盖率趋势（用于连续监控系统）
        self.test_coverage_trends: List[Dict[str, Any]] = []

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
        now = datetime.now()
        rule_key = alert.get('rule_id') or alert.get('type') or alert.get('alert_id') or 'unknown'
        # 统一字段，避免后续代码依赖 rule_id 时触发 KeyError
        alert.setdefault('rule_id', rule_key)
        self.last_alert_times[rule_key] = now

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

    def analyze_and_alert(
        self,
        coverage_data: Dict[str, Any],
        performance_data: Dict[str, Any],
        resource_data: Dict[str, Any],
        health_data: Dict[str, Any],
        route_health_data: Optional[Dict[str, Any]] = None,
        logger_pool_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        分析监控数据并生成告警（用于连续监控系统）

        Args:
            coverage_data: 测试覆盖率数据
            performance_data: 性能指标数据
            resource_data: 资源使用数据
            health_data: 健康状态数据
            route_health_data: 路由健康检查数据（新增）
            logger_pool_data: Logger池性能数据（新增）

        Returns:
            List[Dict[str, Any]]: 生成的告警列表
        """
        alerts: List[Dict[str, Any]] = []

        # 检查覆盖率告警
        alerts.extend(self._check_coverage_alerts(coverage_data))

        # 检查资源使用告警
        alerts.extend(self._check_resource_alerts(resource_data))

        # 检查健康状态告警
        alerts.extend(self._check_health_alerts(health_data))

        # 检查路由健康告警（新增）
        if route_health_data:
            alerts.extend(self._check_route_health_alerts(route_health_data))

        # 检查Logger池性能告警（新增）
        if logger_pool_data:
            alerts.extend(self._check_logger_pool_alerts(logger_pool_data))

        # 记录告警
        for alert in alerts:
            self._record_alert(alert)

        return alerts

    def _check_route_health_alerts(self, route_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查路由健康告警

        Args:
            route_data: 路由健康检查数据

        Returns:
            List[Dict[str, Any]]: 生成的告警列表
        """
        alerts = []
        
        try:
            health_status = route_data.get('health_status', 'unknown')
            required_missing = route_data.get('required_missing', 0)
            
            # 必需路由缺失：ERROR级别告警
            if required_missing > 0:
                alerts.append({
                    'rule_id': 'route_health_critical',
                    'alert_id': f"route_health_critical_{int(datetime.now().timestamp())}",
                    'type': 'route_health_critical',
                    'severity': 'error',
                    'message': f'发现 {required_missing} 个必需路由缺失',
                    'details': route_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            # 健康状态异常：WARNING级别告警
            if health_status == 'unhealthy' and required_missing == 0:
                alerts.append({
                    'rule_id': 'route_health_warning',
                    'alert_id': f"route_health_warning_{int(datetime.now().timestamp())}",
                    'type': 'route_health_warning',
                    'severity': 'warning',
                    'message': '路由健康检查状态异常（但无必需路由缺失）',
                    'details': route_data,
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.warning(f"检查路由健康告警失败: {e}")
        
        return alerts

    def _check_logger_pool_alerts(self, pool_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查Logger池性能告警

        Args:
            pool_data: Logger池性能数据

        Returns:
            List[Dict[str, Any]]: 生成的告警列表
        """
        alerts = []
        
        try:
            hit_rate = pool_data.get('hit_rate', 0.0)
            status = pool_data.get('status', 'unknown')
            
            # Logger池命中率过低：WARNING级别告警
            if hit_rate < self._thresholds.get('hit_rate_low', 0.8):
                alerts.append({
                    'rule_id': 'logger_pool_performance',
                    'alert_id': f"logger_pool_performance_{int(datetime.now().timestamp())}",
                    'type': 'logger_pool_performance',
                    'severity': 'warning',
                    'message': f'Logger池命中率过低: {hit_rate:.2%} < {self._thresholds.get("hit_rate_low", 0.8):.2%}',
                    'details': pool_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Logger池状态异常：WARNING级别告警
            if status == 'critical':
                alerts.append({
                    'rule_id': 'logger_pool_critical',
                    'alert_id': f"logger_pool_critical_{int(datetime.now().timestamp())}",
                    'type': 'logger_pool_critical',
                    'severity': 'warning',
                    'message': 'Logger池状态异常',
                    'details': pool_data,
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.warning(f"检查Logger池告警失败: {e}")
        
        return alerts

    def update_coverage_trends(self, coverage_data: Dict[str, Any]) -> None:
        """
        更新测试覆盖率趋势（用于连续监控系统）

        Args:
            coverage_data: 测试覆盖率数据
        """
        if not coverage_data:
            return

        trend_entry = {
            'timestamp': datetime.now().isoformat(),
            'coverage_percent': coverage_data.get('coverage_percent') or coverage_data.get('overall_coverage', 0.0),
            'line_coverage': coverage_data.get('line_coverage', 0.0),
            'branch_coverage': coverage_data.get('branch_coverage', 0.0),
            'function_coverage': coverage_data.get('function_coverage', 0.0),
            'class_coverage': coverage_data.get('class_coverage', 0.0),
        }

        self.test_coverage_trends.append(trend_entry)

        # 限制趋势历史记录大小（保留最近1000条）
        max_trends = 1000
        if len(self.test_coverage_trends) > max_trends:
            self.test_coverage_trends = self.test_coverage_trends[-max_trends:]

    def _check_coverage_alerts(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查测试覆盖率告警

        Args:
            coverage_data: 测试覆盖率数据

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        alerts: List[Dict[str, Any]] = []

        if not coverage_data:
            return alerts

        # 获取当前覆盖率
        current_coverage = (
            coverage_data.get('coverage_percent') or
            coverage_data.get('overall_coverage') or
            0.0
        )

        # 检查覆盖率下降
        if self.test_coverage_trends:
            previous_entry = self.test_coverage_trends[-1]
            previous_coverage = previous_entry.get('coverage_percent', current_coverage)
            coverage_drop = previous_coverage - current_coverage

            # 如果覆盖率下降超过阈值（默认5%）
            drop_threshold = self.alert_thresholds.get('coverage_drop', 5.0)
            if coverage_drop >= drop_threshold:
                alerts.append({
                    'type': 'coverage_drop',
                    'severity': 'warning',
                    'level': 'warning',
                    'message': f'测试覆盖率下降了{coverage_drop:.1f}%，从{previous_coverage:.1f}%降至{current_coverage:.1f}%',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'previous': previous_coverage,
                        'current': current_coverage,
                        'drop': coverage_drop
                    }
                })

        # 检查覆盖率过低
        min_coverage_threshold = self.alert_thresholds.get('min_coverage', 70.0)
        if current_coverage < min_coverage_threshold:
            alerts.append({
                'type': 'coverage_low',
                'severity': 'warning',
                'level': 'warning',
                'message': f'测试覆盖率过低: {current_coverage:.1f}% (阈值: {min_coverage_threshold}%)',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'current': current_coverage,
                    'threshold': min_coverage_threshold
                }
            })

        return alerts

    def _check_resource_alerts(self, resource_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查资源使用告警

        Args:
            resource_data: 资源使用数据

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        alerts: List[Dict[str, Any]] = []

        if not resource_data:
            return alerts

        # 检查内存使用
        memory_usage = resource_data.get('memory_detailed', {}).get('usage_percent')
        if memory_usage is None:
            # 尝试其他路径
            memory_usage = resource_data.get('memory', {}).get('usage_percent')
        
        if memory_usage is not None:
            memory_threshold = self.alert_thresholds.get('memory_usage_high', 80.0)
            if memory_usage > memory_threshold:
                alerts.append({
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'level': 'warning',
                    'message': f'内存使用率过高: {memory_usage:.1f}%',
                    'timestamp': datetime.now().isoformat(),
                    'data': {'memory_usage': memory_usage}
                })

        # 检查CPU使用
        cpu_usage = resource_data.get('cpu', {}).get('usage_percent')
        if cpu_usage is None:
            cpu_usage = resource_data.get('process', {}).get('cpu_percent')

        if cpu_usage is not None:
            cpu_threshold = self.alert_thresholds.get('cpu_usage_high', 70.0)
            if cpu_usage > cpu_threshold:
                alerts.append({
                    'type': 'high_cpu_usage',
                    'severity': 'warning',
                    'level': 'warning',
                    'message': f'CPU使用率过高: {cpu_usage:.1f}%',
                    'timestamp': datetime.now().isoformat(),
                    'data': {'cpu_usage': cpu_usage}
                })

        return alerts

    def _check_health_alerts(self, health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查健康状态告警

        Args:
            health_data: 健康状态数据

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        alerts: List[Dict[str, Any]] = []

        if not health_data:
            return alerts

        # 检查整体健康状态
        overall_status = health_data.get('overall_status', 'unknown')
        health_score = health_data.get('health_score', 100)

        if overall_status == 'critical' or health_score < 60:
            alerts.append({
                'type': 'system_health_critical',
                'severity': 'error',
                'level': 'error',
                'message': f'系统健康状态严重: {overall_status} (健康评分: {health_score})',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'status': overall_status,
                    'health_score': health_score
                }
            })
        elif overall_status == 'warning' or (health_score < 80 and health_score >= 60):
            alerts.append({
                'type': 'system_health_warning',
                'severity': 'warning',
                'level': 'warning',
                'message': f'系统健康状态警告: {overall_status} (健康评分: {health_score})',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'status': overall_status,
                    'health_score': health_score
                }
            })

        # 检查组件健康状态
        components = health_data.get('components', {})
        unhealthy_components = []
        for component_name, component_status in components.items():
            if component_status not in ['healthy', 'ok']:
                unhealthy_components.append(component_name)

        if unhealthy_components:
            alerts.append({
                'type': 'component_unhealthy',
                'severity': 'warning',
                'level': 'warning',
                'message': f'以下组件不健康: {", ".join(unhealthy_components)}',
                'timestamp': datetime.now().isoformat(),
                'data': {'unhealthy_components': unhealthy_components}
            })

        return alerts

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