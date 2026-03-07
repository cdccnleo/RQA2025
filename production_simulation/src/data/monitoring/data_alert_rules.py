#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层智能告警规则引擎

实现基于数据层特点的智能告警规则引擎，
支持动态规则配置和智能告警抑制。

设计模式：规则引擎模式 + 策略模式
职责：智能数据层告警检测，动态规则管理和告警抑制
"""

from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from src.core.integration import get_data_adapter
from ...interfaces.standard_interfaces import DataSourceType


class AlertRuleType(Enum):

    """告警规则类型"""
    THRESHOLD = "threshold"      # 阈值规则
    TREND = "trend"             # 趋势规则
    ANOMALY = "anomaly"         # 异常检测规则
    COMPOSITE = "composite"     # 复合规则


class AlertSeverity(Enum):

    """告警严重程度"""
    INFO = "info"           # 信息级
    WARNING = "warning"     # 警告级
    ERROR = "error"         # 错误级
    CRITICAL = "critical"   # 严重级


class AlertConditionType(Enum):

    """告警条件类型"""
    GREATER_THAN = "gt"         # 大于
    LESS_THAN = "lt"           # 小于
    EQUAL = "eq"               # 等于
    NOT_EQUAL = "ne"           # 不等于
    GREATER_EQUAL = "ge"       # 大于等于
    LESS_EQUAL = "le"          # 小于等于
    BETWEEN = "between"        # 在范围内
    OUTSIDE = "outside"        # 超出范围
    CHANGE_RATE = "change_rate"  # 变化率
    CUSTOM = "custom"          # 自定义条件


@dataclass
class AlertCondition:

    """告警条件"""
    type: AlertConditionType
    field: str
    value: Union[float, int, str, List[Union[float, int, str]]]
    custom_func: Optional[Callable] = None

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """
        评估条件

        Args:
            data: 数据字典

        Returns:
            是否满足条件
        """
        if self.field not in data:
            return False

        actual_value = data[self.field]

        try:
            if self.type == AlertConditionType.GREATER_THAN:
                return actual_value > self.value
            elif self.type == AlertConditionType.LESS_THAN:
                return actual_value < self.value
            elif self.type == AlertConditionType.EQUAL:
                return actual_value == self.value
            elif self.type == AlertConditionType.NOT_EQUAL:
                return actual_value != self.value
            elif self.type == AlertConditionType.GREATER_EQUAL:
                return actual_value >= self.value
            elif self.type == AlertConditionType.LESS_EQUAL:
                return actual_value <= self.value
            elif self.type == AlertConditionType.BETWEEN:
                if isinstance(self.value, list) and len(self.value) == 2:
                    return self.value[0] <= actual_value <= self.value[1]
                return False
            elif self.type == AlertConditionType.OUTSIDE:
                if isinstance(self.value, list) and len(self.value) == 2:
                    return actual_value < self.value[0] or actual_value > self.value[1]
                return False
            elif self.type == AlertConditionType.CHANGE_RATE:
                # 计算变化率（需要历史数据支持）
                return self._evaluate_change_rate(actual_value, data)
            elif self.type == AlertConditionType.CUSTOM:
                if self.custom_func:
                    return self.custom_func(data)
                return False
            else:
                return False

        except (TypeError, ValueError):
            return False

    def _evaluate_change_rate(self, current_value: float, data: Dict[str, Any]) -> bool:
        """评估变化率"""
        # 这里需要访问历史数据来计算变化率
        # 简化实现：假设阈值是变化百分比
        if isinstance(self.value, (int, float)):
            # 如果没有历史数据，假设没有变化
            return False

        # TODO: 实现基于历史数据的变化率计算
        return False


@dataclass
class AlertRule:

    """告警规则"""
    rule_id: str
    name: str
    description: str
    rule_type: AlertRuleType
    conditions: List[AlertCondition]
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 5
    data_types: List[DataSourceType] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def evaluate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        评估规则

        Args:
            data: 监控数据
            context: 上下文信息

        Returns:
            告警信息，如果没有触发则返回None
        """
        if not self.enabled:
            return None

        # 检查数据类型过滤
        if self.data_types:
            data_type_str = data.get('data_type')
            if data_type_str:
                try:
                    data_type = DataSourceType(data_type_str)
                    if data_type not in self.data_types:
                        return None
                except ValueError:
                    return None

        # 评估所有条件
        condition_results = []
        for condition in self.conditions:
            try:
                result = condition.evaluate(data)
                condition_results.append(result)
            except Exception as e:
                # 条件评估失败，记录但不影响其他条件
                condition_results.append(False)

        # 根据规则类型判断是否触发
        if self.rule_type == AlertRuleType.THRESHOLD:
            # 阈值规则：所有条件都必须满足
            if all(condition_results):
                return self._create_alert(data, context)
        elif self.rule_type == AlertRuleType.COMPOSITE:
            # 复合规则：至少一个条件满足
            if any(condition_results):
                return self._create_alert(data, context)

        return None

    def _create_alert(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建告警"""
        # 格式化消息
        message = self.message_template.format(**data)

        return {
            'rule_id': self.rule_id,
            'rule_name': self.name,
            'severity': self.severity.value,
            'message': message,
            'data': data,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'cooldown_minutes': self.cooldown_minutes
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type.value,
            'conditions': [
                {
                    'type': c.type.value,
                    'field': c.field,
                    'value': c.value
                }
                for c in self.conditions
            ],
            'severity': self.severity.value,
            'message_template': self.message_template,
            'enabled': self.enabled,
            'cooldown_minutes': self.cooldown_minutes,
            'data_types': [dt.value for dt in self.data_types],
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """从字典创建"""
        conditions = []
        for cond_data in data.get('conditions', []):
            condition = AlertCondition(
                type=AlertConditionType(cond_data['type']),
                field=cond_data['field'],
                value=cond_data['value']
            )
            conditions.append(condition)

        data_types = []
        for dt_str in data.get('data_types', []):
            try:
                data_types.append(DataSourceType(dt_str))
            except ValueError:
                continue

        return cls(
            rule_id=data['rule_id'],
            name=data['name'],
            description=data.get('description', ''),
            rule_type=AlertRuleType(data['rule_type']),
            conditions=conditions,
            severity=AlertSeverity(data['severity']),
            message_template=data['message_template'],
            enabled=data.get('enabled', True),
            cooldown_minutes=data.get('cooldown_minutes', 5),
            data_types=data_types,
            tags=data.get('tags', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )


@dataclass
class AlertSuppression:

    """告警抑制"""
    suppression_id: str
    rule_ids: List[str]  # 被抑制的规则ID列表
    condition: AlertCondition  # 抑制条件
    duration_minutes: int  # 抑制时长
    reason: str  # 抑制原因
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=60))

    def is_active(self) -> bool:
        """检查抑制是否仍然有效"""
        return datetime.now() < self.expires_at

    def matches_alert(self, alert_data: Dict[str, Any]) -> bool:
        """检查告警是否匹配抑制条件"""
        return self.condition.evaluate(alert_data)


class DataAlertRulesEngine:

    """
    数据层智能告警规则引擎

    提供智能的告警规则管理和评估：
    - 动态规则配置和更新
    - 告警抑制和降噪
    - 规则优先级和依赖
    - 告警关联和聚合
    - 学习型规则优化
    """

    def __init__(self):
        """
        初始化告警规则引擎 - 使用统一基础设施集成层
        """
        # 初始化统一基础设施集成层
        try:
            data_adapter = get_data_adapter()
            self.monitoring = data_adapter.get_monitoring()
            self.logger = data_adapter.get_logger()
        except Exception as e:
            import logging
            self.monitoring = None
            self.logger = logging.getLogger(__name__)

        self.rules: Dict[str, AlertRule] = {}
        self.suppressions: Dict[str, AlertSuppression] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.rule_performance: Dict[str, Dict[str, Any]] = {}

        # 初始化标准规则
        self._initialize_standard_rules()

        self._log_operation('initialized', 'DataAlertRulesEngine', 'success')

    def _initialize_standard_rules(self):
        """初始化标准告警规则"""
        standard_rules = [
            # 缓存性能规则
            AlertRule(
                rule_id='cache_hit_rate_low',
                name='缓存命中率过低',
                description='检测缓存命中率是否低于阈值',
                rule_type=AlertRuleType.THRESHOLD,
                conditions=[
                    AlertCondition(
                        type=AlertConditionType.LESS_THAN,
                        field='hit_rate',
                        value=0.8
                    )
                ],
                severity=AlertSeverity.WARNING,
                message_template='缓存命中率过低: {hit_rate:.2%}，低于80 % 阈值',
                cooldown_minutes=5,
                tags={'category': 'cache', 'metric': 'hit_rate'}
            ),

            # 数据质量规则
            AlertRule(
                rule_id='data_quality_degraded',
                name='数据质量下降',
                description='检测数据质量指标是否下降',
                rule_type=AlertRuleType.COMPOSITE,
                conditions=[
                    AlertCondition(
                        type=AlertConditionType.LESS_THAN,
                        field='completeness',
                        value=0.95
                    ),
                    AlertCondition(
                        type=AlertConditionType.LESS_THAN,
                        field='accuracy',
                        value=0.95
                    ),
                    AlertCondition(
                        type=AlertConditionType.LESS_THAN,
                        field='timeliness',
                        value=0.9
                    )
                ],
                severity=AlertSeverity.ERROR,
                message_template='数据质量下降: 完整性={completeness:.2%}, 准确性={accuracy:.2%}, 时效性={timeliness:.2%}',
                cooldown_minutes=10,
                tags={'category': 'quality', 'metric': 'composite'}
            ),

            # 处理性能规则
            AlertRule(
                rule_id='processing_error_rate_high',
                name='处理错误率过高',
                description='检测数据处理错误率是否过高',
                rule_type=AlertRuleType.THRESHOLD,
                conditions=[
                    AlertCondition(
                        type=AlertConditionType.GREATER_THAN,
                        field='error_rate',
                        value=0.05
                    )
                ],
                severity=AlertSeverity.CRITICAL,
                message_template='数据处理错误率过高: {error_rate:.2%}，超过5 % 阈值',
                cooldown_minutes=2,
                tags={'category': 'processing', 'metric': 'error_rate'}
            ),

            # 响应时间规则
            AlertRule(
                rule_id='response_time_high',
                name='响应时间过高',
                description='检测数据响应时间是否过高',
                rule_type=AlertRuleType.THRESHOLD,
                conditions=[
                    AlertCondition(
                        type=AlertConditionType.GREATER_THAN,
                        field='avg_response_time',
                        value=5.0
                    )
                ],
                severity=AlertSeverity.WARNING,
                message_template='数据响应时间过高: {avg_response_time:.2f}秒，超过5秒阈值',
                cooldown_minutes=3,
                tags={'category': 'performance', 'metric': 'response_time'}
            ),

            # 连接失败规则
            AlertRule(
                rule_id='connection_failed',
                name='连接失败',
                description='检测数据源连接是否失败',
                rule_type=AlertRuleType.THRESHOLD,
                conditions=[
                    AlertCondition(
                        type=AlertConditionType.EQUAL,
                        field='connection_status',
                        value='failed'
                    )
                ],
                severity=AlertSeverity.CRITICAL,
                message_template='数据源连接失败: {connection_status}',
                cooldown_minutes=1,
                tags={'category': 'connectivity', 'metric': 'connection_status'}
            )
        ]

        for rule in standard_rules:
            self.add_rule(rule)

    # =========================================================================
    # 规则管理
    # =========================================================================

    def add_rule(self, rule: AlertRule) -> bool:
        """
        添加告警规则

        Args:
            rule: 告警规则

        Returns:
            是否添加成功
        """
        try:
            if rule.rule_id in self.rules:
                raise ValueError(f"规则ID已存在: {rule.rule_id}")

            self.rules[rule.rule_id] = rule
            self.rule_performance[rule.rule_id] = {
                'alerts_triggered': 0,
                'false_positives': 0,
                'last_triggered': None,
                'created_at': datetime.now().isoformat()
            }

            self._log_operation('add_rule', rule.rule_id, 'success')
            return True

        except Exception as e:
            self._log_operation('add_rule', rule.rule_id, f'failed: {e}')
            return False

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新告警规则

        Args:
            rule_id: 规则ID
            updates: 更新内容

        Returns:
            是否更新成功
        """
        try:
            if rule_id not in self.rules:
                raise ValueError(f"规则不存在: {rule_id}")

            rule = self.rules[rule_id]

            # 更新规则属性
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            rule.updated_at = datetime.now()

            self._log_operation('update_rule', rule_id, 'success')
            return True

        except Exception as e:
            self._log_operation('update_rule', rule_id, f'failed: {e}')
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """
        移除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            是否移除成功
        """
        try:
            if rule_id not in self.rules:
                raise ValueError(f"规则不存在: {rule_id}")

            del self.rules[rule_id]
            if rule_id in self.rule_performance:
                del self.rule_performance[rule_id]

            self._log_operation('remove_rule', rule_id, 'success')
            return True

        except Exception as e:
            self._log_operation('remove_rule', rule_id, f'failed: {e}')
            return False

    def enable_rule(self, rule_id: str) -> bool:
        """
        启用规则

        Args:
            rule_id: 规则ID

        Returns:
            是否启用成功
        """
        return self.update_rule(rule_id, {'enabled': True})

    def disable_rule(self, rule_id: str) -> bool:
        """
        禁用规则

        Args:
            rule_id: 规则ID

        Returns:
            是否禁用成功
        """
        return self.update_rule(rule_id, {'enabled': False})

    # =========================================================================
    # 告警评估和抑制
    # =========================================================================

    def evaluate_rules(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        评估所有规则

        Args:
            data: 监控数据
            context: 上下文信息

        Returns:
            触发的告警列表
        """
        alerts = []
        evaluated_rules = []

        try:
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue

                evaluated_rules.append(rule_id)

                # 检查告警抑制
                if self._is_alert_suppressed(rule_id, data):
                    continue

                # 评估规则
                alert = rule.evaluate(data, context)
                if alert:
                    # 检查冷却时间
                    if not self._is_in_cooldown(rule_id, alert['cooldown_minutes']):
                        alerts.append(alert)
                        self._record_alert(rule_id, alert)

            self._log_operation(
                'evaluate_rules', f"{len(evaluated_rules)} rules", f"{len(alerts)} alerts")

        except Exception as e:
            self._log_operation('evaluate_rules', 'batch', f'failed: {e}')

        return alerts

    def _is_alert_suppressed(self, rule_id: str, data: Dict[str, Any]) -> bool:
        """检查告警是否被抑制"""
        for suppression in self.suppressions.values():
            if suppression.is_active() and rule_id in suppression.rule_ids:
                if suppression.matches_alert(data):
                    return True
        return False

    def _is_in_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        """检查是否在冷却时间内"""
        if rule_id not in self.rule_performance:
            return False

        last_triggered = self.rule_performance[rule_id].get('last_triggered')
        if not last_triggered:
            return False

        cooldown_end = datetime.fromisoformat(last_triggered) + timedelta(minutes=cooldown_minutes)
        return datetime.now() < cooldown_end

    def _record_alert(self, rule_id: str, alert: Dict[str, Any]):
        """记录告警"""
        # 更新规则性能统计
        if rule_id in self.rule_performance:
            perf = self.rule_performance[rule_id]
            perf['alerts_triggered'] += 1
            perf['last_triggered'] = datetime.now().isoformat()

        # 添加到历史记录
        alert_record = {
            **alert,
            'recorded_at': datetime.now().isoformat()
        }
        self.alert_history.append(alert_record)

        # 限制历史记录数量
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-5000:]

    # =========================================================================
    # 告警抑制管理
    # =========================================================================

    def add_suppression(self, suppression: AlertSuppression) -> bool:
        """
        添加告警抑制

        Args:
            suppression: 告警抑制

        Returns:
            是否添加成功
        """
        try:
            if suppression.suppression_id in self.suppressions:
                raise ValueError(f"抑制ID已存在: {suppression.suppression_id}")

            self.suppressions[suppression.suppression_id] = suppression
            self._log_operation('add_suppression', suppression.suppression_id, 'success')
            return True

        except Exception as e:
            self._log_operation('add_suppression', suppression.suppression_id, f'failed: {e}')
            return False

    def remove_suppression(self, suppression_id: str) -> bool:
        """
        移除告警抑制

        Args:
            suppression_id: 抑制ID

        Returns:
            是否移除成功
        """
        try:
            if suppression_id not in self.suppressions:
                raise ValueError(f"抑制不存在: {suppression_id}")

            del self.suppressions[suppression_id]
            self._log_operation('remove_suppression', suppression_id, 'success')
            return True

        except Exception as e:
            self._log_operation('remove_suppression', suppression_id, f'failed: {e}')
            return False

    def cleanup_expired_suppressions(self) -> int:
        """
        清理过期的告警抑制

        Returns:
            清理的数量
        """
        expired_ids = [
            sid for sid, suppression in self.suppressions.items()
            if not suppression.is_active()
        ]

        for sid in expired_ids:
            del self.suppressions[sid]

        self._log_operation('cleanup_suppressions', f"{len(expired_ids)} expired", 'success')
        return len(expired_ids)

    # =========================================================================
    # 配置导入导出
    # =========================================================================

    def export_rules(self) -> str:
        """
        导出规则配置

        Returns:
            JSON格式的规则配置
        """
        rules_data = {
            'rules': [rule.to_dict() for rule in self.rules.values()],
            'suppressions': [
                {
                    'suppression_id': s.suppression_id,
                    'rule_ids': s.rule_ids,
                    'condition': {
                        'type': s.condition.type.value,
                        'field': s.condition.field,
                        'value': s.condition.value
                    },
                    'duration_minutes': s.duration_minutes,
                    'reason': s.reason,
                    'created_at': s.created_at.isoformat(),
                    'expires_at': s.expires_at.isoformat()
                }
                for s in self.suppressions.values()
            ],
            'exported_at': datetime.now().isoformat()
        }

        return json.dumps(rules_data, indent=2, ensure_ascii=False)

    def import_rules(self, rules_json: str) -> Dict[str, Any]:
        """
        导入规则配置

        Args:
            rules_json: JSON格式的规则配置

        Returns:
            导入结果
        """
        result = {
            'imported_rules': 0,
            'imported_suppressions': 0,
            'errors': []
        }

        try:
            data = json.loads(rules_json)

            # 导入规则
            for rule_data in data.get('rules', []):
                try:
                    rule = AlertRule.from_dict(rule_data)
                    if self.add_rule(rule):
                        result['imported_rules'] += 1
                    else:
                        result['errors'].append(f"Failed to add rule: {rule_data['rule_id']}")
                except Exception as e:
                    result['errors'].append(f"Failed to parse rule: {e}")

            # 导入抑制
            for supp_data in data.get('suppressions', []):
                try:
                    condition = AlertCondition(
                        type=AlertConditionType(supp_data['condition']['type']),
                        field=supp_data['condition']['field'],
                        value=supp_data['condition']['value']
                    )

                    suppression = AlertSuppression(
                        suppression_id=supp_data['suppression_id'],
                        rule_ids=supp_data['rule_ids'],
                        condition=condition,
                        duration_minutes=supp_data['duration_minutes'],
                        reason=supp_data['reason'],
                        created_at=datetime.fromisoformat(supp_data['created_at']),
                        expires_at=datetime.fromisoformat(supp_data['expires_at'])
                    )

                    if self.add_suppression(suppression):
                        result['imported_suppressions'] += 1
                    else:
                        result['errors'].append(
                            f"Failed to add suppression: {supp_data['suppression_id']}")
                except Exception as e:
                    result['errors'].append(f"Failed to parse suppression: {e}")

            result['success'] = len(result['errors']) == 0

        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Failed to parse JSON: {e}")

        return result

    # =========================================================================
    # 统计和报告
    # =========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            引擎统计信息
        """
        total_alerts = len(self.alert_history)
        recent_alerts = [
            a for a in self.alert_history
            if datetime.fromisoformat(a['recorded_at']) > datetime.now() - timedelta(hours=24)
        ]

        rules_by_severity = {}
        for rule in self.rules.values():
            severity = rule.severity.value
            if severity not in rules_by_severity:
                rules_by_severity[severity] = 0
            rules_by_severity[severity] += 1

        return {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
            'disabled_rules': sum(1 for r in self.rules.values() if not r.enabled),
            'rules_by_severity': rules_by_severity,
            'active_suppressions': len([s for s in self.suppressions.values() if s.is_active()]),
            'total_alerts': total_alerts,
            'recent_alerts_24h': len(recent_alerts),
            'alerts_by_rule': {
                rule_id: perf['alerts_triggered']
                for rule_id, perf in self.rule_performance.items()
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_rule_performance_report(self) -> Dict[str, Any]:
        """
        获取规则性能报告

        Returns:
            规则性能报告
        """
        report = {
            'rule_performance': {},
            'top_triggered_rules': [],
            'inactive_rules': [],
            'timestamp': datetime.now().isoformat()
        }

        # 规则性能详情
        for rule_id, perf in self.rule_performance.items():
            report['rule_performance'][rule_id] = {
                **perf,
                'rule_name': self.rules[rule_id].name if rule_id in self.rules else 'Unknown'
            }

        # 最常触发的规则
        sorted_rules = sorted(
            self.rule_performance.items(),
            key=lambda x: x[1]['alerts_triggered'],
            reverse=True
        )
        report['top_triggered_rules'] = [
            {
                'rule_id': rule_id,
                'rule_name': self.rules[rule_id].name if rule_id in self.rules else 'Unknown',
                'alerts_triggered': perf['alerts_triggered'],
                'last_triggered': perf['last_triggered']
            }
            for rule_id, perf in sorted_rules[:10]
        ]

        # 不活跃的规则
        report['inactive_rules'] = [
            rule_id for rule_id, perf in self.rule_performance.items()
            if perf['alerts_triggered'] == 0
        ]

        return report

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _log_operation(self, operation: str, target: str, status: str) -> None:
        """
        记录操作日志

        Args:
            operation: 操作类型
            target: 操作目标
            status: 操作状态
        """
        try:
            message = f"告警规则引擎 - {operation}: {target}, 状态: {status}"
            print(f"[DataAlertRulesEngine] {message}")

            # 通过统一监控服务记录操作
            if self.monitoring:
                try:
                    self.monitoring.record_metric(
                        'alert_engine_operation',
                        1,
                        {'operation': operation, 'status': status},
                        description='告警引擎操作计数'
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"监控记录失败: {e}")

        except Exception:
            print(f"[DataAlertRulesEngine] {operation}: {target} - {status}")

    def clear_history(self) -> None:
        """清除历史记录"""
        self.alert_history.clear()
        self._log_operation('clear_history', 'alert_history', 'success')

    def reset_rule_performance(self) -> None:
        """重置规则性能统计"""
        for perf in self.rule_performance.values():
            perf['alerts_triggered'] = 0
            perf['false_positives'] = 0
            perf['last_triggered'] = None

        self._log_operation('reset_performance', 'rule_performance', 'success')
