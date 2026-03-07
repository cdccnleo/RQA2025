"""
alert_rule_engine 模块

提供 alert_rule_engine 相关功能和接口。
"""

import json
import logging
import re

# -*- coding: utf-8 -*-
import asyncio
import threading
import time

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from typing import TYPE_CHECKING
"""
基础设施层 - 日志系统组件

alert_rule_engine 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr/bin/env python3
"""
alert_rule_engine - 日志系统

职责说明：
负责系统日志记录、日志格式化、日志存储和日志分析

核心职责：
- 日志记录和格式化
- 日志级别管理
- 日志存储和轮转
- 日志分析和监控
- 日志搜索和过滤
- 日志性能优化

相关接口：
- ILoggingComponent
- ILogger
- ILogHandler
""" """
智能告警规则引擎
支持动态规则配置、自动阈值调整和智能告警抑制
"""

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):

    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):

    """告警状态"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:

    """告警规则定义"""
    rule_id: str
    name: str
    condition: str
    severity: AlertSeverity
    description: str = ""
    metric_name: str = ""
    metric_type: str = ""
    threshold: Optional[float] = None
    enabled: bool = True
    duration: Optional[int] = None
    percentile: Optional[float] = None  # 用于histogram和summary类型
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # 自动阈值调整配置
    auto_adjust_threshold: bool = False
    adjustment_strategy: str = "percentile"  # 'percentile', 'sigma'
    adjustment_percentile: Optional[float] = 0.95
    adjustment_sigma: Optional[float] = 2.0
    baseline_period: int = 100  # 基线计算周期

    # 告警抑制配置
    maintenance_windows: List[Dict[str, datetime]] = field(default_factory=list)
    silence_period: Optional[int] = None  # 静默期（分钟）

    # 创建时间
    created_at: datetime = field(default_factory=datetime.now)

    # 运行手册URL
    runbook_url: Optional[str] = None
    updated_at: Optional[datetime] = None


@dataclass
class Alert:

    """告警对象"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    generator_url: str = "alert_rule_engine"
    status: AlertStatus = AlertStatus.FIRING


@dataclass
class AlertInstance:

    """告警实例"""
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    value: float
    threshold: float
    labels: Dict[str, str]
    annotations: Dict[str, str]
    start_time: datetime
    last_update: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class AlertRuleEngine:
    """
    智能告警规则引擎 - 门面类

    协调各个告警处理组件，提供统一的告警规则管理接口
    遵循门面模式和组合优于继承原则
    """

    def __init__(self, prometheus_exporter=None, alert_manager=None, metrics_collector=None):
        """
        初始化告警规则引擎

        Args:
            prometheus_exporter: Prometheus导出器实例
            alert_manager: 告警管理器实例
            metrics_collector: 指标收集器实例
        """
        # 组合各个组件
        self._rule_manager = RuleManager()
        self._rule_evaluator = RuleEvaluator(metrics_collector)
        self._alert_generator = AlertGenerator(alert_manager)
        self._threshold_adjuster = ThresholdAdjuster()
        self._alert_suppressor = AlertSuppressor()

        # 保留兼容性属性
        self.prometheus_exporter = prometheus_exporter
        self.alert_manager = alert_manager

        # 告警规则存储（兼容性）
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = []

        logger.info("Alert rule engine initialized with modular components")

    def add_rule(self, rule: "AlertRule") -> bool:
        """添加告警规则"""
        success = self._rule_manager.add_rule(rule)
        if success:
            self.rules[rule.rule_id] = rule  # 保持兼容性
        return success

    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        success = self._rule_manager.remove_rule(rule_id)
        if success and rule_id in self.rules:
            del self.rules[rule_id]  # 保持兼容性
        return success

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新告警规则"""
        return self._rule_manager.update_rule(rule_id, updates)

    def evaluate_rules(self) -> List[Dict[str, Any]]:
        """评估所有活跃规则"""
        results = []
        active_rules = self._rule_manager.get_active_rules()

        for rule in active_rules:
            # 评估规则
            evaluation_result = self._rule_evaluator.evaluate_rule(rule)

            if evaluation_result.get('success', False):
                # 检查告警抑制
                rule_obj = self._rule_manager.get_rule(rule.rule_id)
                if rule_obj:
                    # 这里需要创建告警对象来检查抑制，暂时简化
                    is_suppressed = False  # 简化逻辑

                    if not is_suppressed and evaluation_result.get('is_triggered', False):
                        # 生成告警
                        alert = self._alert_generator.generate_alert(rule, evaluation_result)
                        if alert:
                            # 发送告警
                            self._alert_generator.send_alert(alert)

                            # 记录活跃告警
                            self.active_alerts[alert.alert_id] = alert
                            self.alert_history.append(alert)

                # 尝试调整阈值（如果启用）
                if hasattr(rule, 'auto_adjust_threshold') and rule.auto_adjust_threshold:
                    # 这里需要获取历史数据来调整阈值，暂时简化
                    pass

            results.append(evaluation_result)

        return results

    def get_active_alerts(self) -> List[AlertInstance]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, rule_id: Optional[str] = None) -> List[AlertInstance]:
        """获取告警历史"""
        history = self.alert_history[-100:]
        if rule_id:
            history = [alert for alert in history if alert.rule_name == rule_id or alert.rule_id == rule_id]
        return history

    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        all_rules = self._rule_manager.get_all_rules()
        active_rules = self._rule_manager.get_active_rules()

        return {
            'total_rules': len(all_rules),
            'active_rules': len(active_rules),
            'inactive_rules': len(all_rules) - len(active_rules),
            'rules_by_severity': self._count_rules_by_severity(all_rules)
        }

    def _count_rules_by_severity(self, rules: List["AlertRule"]) -> Dict[str, int]:
        """按严重程度统计规则数量"""
        severity_count = {}
        for rule in rules:
            severity = rule.severity.value
            severity_count[severity] = severity_count.get(severity, 0) + 1
        return severity_count

    def acknowledge_alert(self, alert_id: str, user: Optional[str] = None) -> bool:
        """确认告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = user or "system"
            alert.acknowledged_at = datetime.now()
            logger.info(f"告警已确认: {alert_id} by {alert.acknowledged_by}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            logger.info(f"告警已解决: {alert_id}")
            return True
        return False

    def suppress_alert(self, alert_id: str, duration: str) -> bool:
        """抑制告警"""
        if alert_id in self.active_alerts:
            # 解析持续时间
            try:
                duration_minutes = self._parse_duration(duration)
                suppression_config = {
                    'rule_id': alert_id,
                    'suppressed_by_labels': [['suppressed']],
                    'duration_minutes': duration_minutes
                }
                self._alert_suppressor.add_suppression_rule(alert_id, suppression_config)
                logger.info(f"告警已抑制: {alert_id} for {duration}")
                return True
            except ValueError as e:
                logger.error(f"无效的持续时间格式: {duration}")
                return False
        return False

    def _parse_duration(self, duration: str) -> int:
        """解析持续时间字符串"""
        # 支持多种格式：s(秒), m(分钟), h(小时), d(天)
        if duration.endswith('s'):
            return int(duration[:-1])
        elif duration.endswith('m'):
            return int(duration[:-1]) * 60
        elif duration.endswith('h'):
            return int(duration[:-1]) * 3600
        elif duration.endswith('d'):
            return int(duration[:-1]) * 86400
        else:
            raise ValueError(f"不支持的持续时间格式: {duration}")

    def stop(self):
        """停止告警规则引擎"""
        logger.info("停止告警规则引擎")
        # 这里可以停止评估线程等清理工作


class RuleManager:
    """
    规则管理器 - 专门负责告警规则的增删改查

    单一职责：管理告警规则的生命周期
    """

    def __init__(self):
        self._rules: Dict[str, "AlertRule"] = {}
        self._rule_lock = threading.Lock()

    def add_rule(self, rule: "AlertRule") -> bool:
        """添加告警规则"""
        with self._rule_lock:
            if rule.rule_id in self._rules:
                logger.warning(f"规则已存在: {rule.rule_id}")
                return False

            self._rules[rule.rule_id] = rule
            logger.info(f"添加告警规则: {rule.rule_id}")
            return True

    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        with self._rule_lock:
            if rule_id not in self._rules:
                logger.warning(f"规则不存在: {rule_id}")
                return False

            del self._rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")
            return True

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新告警规则"""
        with self._rule_lock:
            if rule_id not in self._rules:
                logger.warning(f"规则不存在: {rule_id}")
                return False

            rule = self._rules[rule_id]

            # 更新规则属性
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            rule.updated_at = datetime.now()
            logger.info(f"更新告警规则: {rule_id}")
            return True

    def get_rule(self, rule_id: str) -> Optional["AlertRule"]:
        """获取告警规则"""
        with self._rule_lock:
            return self._rules.get(rule_id)

    def get_all_rules(self) -> List["AlertRule"]:
        """获取所有规则"""
        with self._rule_lock:
            return list(self._rules.values())

    def get_active_rules(self) -> List["AlertRule"]:
        """获取活跃规则"""
        with self._rule_lock:
            return [rule for rule in self._rules.values() if rule.enabled]

    def clear_rules(self) -> int:
        """清空所有规则"""
        with self._rule_lock:
            count = len(self._rules)
            self._rules.clear()
            logger.info(f"清空所有规则，共{count}个")
            return count


class RuleEvaluator:
    """
    规则评估器 - 专门负责规则条件的评估

    单一职责：评估告警规则条件并计算当前值
    """

    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector

    def evaluate_rule(self, rule: "AlertRule") -> Dict[str, Any]:
        """
        评估单个规则

        Args:
            rule: 告警规则

        Returns:
            评估结果字典
        """
        try:
            # 获取当前值
            current_value = self._get_current_value(rule)

            # 检查是否满足触发条件
            is_triggered = self._check_condition(rule, current_value)

            return {
                'rule_id': rule.rule_id,
                'current_value': current_value,
                'threshold': rule.threshold,
                'is_triggered': is_triggered,
                'evaluation_time': datetime.now(),
                'success': True
            }

        except Exception as e:
            logger.error(f"规则评估失败 {rule.rule_id}: {e}")
            return {
                'rule_id': rule.rule_id,
                'error': str(e),
                'evaluation_time': datetime.now(),
                'success': False
            }

    def _get_current_value(self, rule: "AlertRule") -> float:
        """获取规则的当前值"""
        try:
            if rule.metric_type == 'counter':
                # 计数器类型指标
                return self.metrics_collector.get_counter_value(rule.metric_name)
            elif rule.metric_type == 'gauge':
                # 仪表盘类型指标
                return self.metrics_collector.get_gauge_value(rule.metric_name)
            elif rule.metric_type == 'histogram':
                # 直方图类型指标
                return self.metrics_collector.get_histogram_value(rule.metric_name, rule.percentile or 0.95)
            elif rule.metric_type == 'summary':
                # 摘要类型指标
                return self.metrics_collector.get_summary_value(rule.metric_name, rule.percentile or 0.95)
            else:
                logger.warning(f"不支持的指标类型: {rule.metric_type}")
                return 0.0
        except Exception as e:
            logger.error(f"获取指标值失败 {rule.metric_name}: {e}")
            return 0.0

    def _check_condition(self, rule: "AlertRule", current_value: float) -> bool:
        """检查是否满足触发条件"""
        if rule.condition == 'gt':  # 大于
            return current_value > rule.threshold
        elif rule.condition == 'lt':  # 小于
            return current_value < rule.threshold
        elif rule.condition == 'gte':  # 大于等于
            return current_value >= rule.threshold
        elif rule.condition == 'lte':  # 小于等于
            return current_value <= rule.threshold
        elif rule.condition == 'eq':  # 等于
            return abs(current_value - rule.threshold) < 0.001
        elif rule.condition == 'ne':  # 不等于
            return abs(current_value - rule.threshold) >= 0.001
        else:
            logger.warning(f"不支持的条件类型: {rule.condition}")
            return False


class AlertGenerator:
    """
    告警生成器 - 专门负责告警的生成和发送

    单一职责：根据评估结果生成和发送告警
    """

    def __init__(self, alert_manager):
        self.alert_manager = alert_manager

    def generate_alert(self, rule: "AlertRule", evaluation_result: Dict[str, Any]) -> Optional[Alert]:
        """
        生成告警

        Args:
            rule: 告警规则
            evaluation_result: 评估结果

        Returns:
            生成的告警对象或None
        """
        if not evaluation_result.get('is_triggered', False):
            return None

        try:
            # 创建告警对象
            alert = Alert(
                alert_id=f"{rule.rule_id}_{int(time.time())}",
                rule_id=rule.rule_id,
                severity=rule.severity,
                title=rule.name,
                description=self._generate_description(rule, evaluation_result),
                labels=self._generate_labels(rule, evaluation_result),
                annotations=self._generate_annotations(rule, evaluation_result),
                starts_at=datetime.now(),
                generator_url="alert_rule_engine"
            )

            return alert

        except Exception as e:
            logger.error(f"生成告警失败 {rule.rule_id}: {e}")
            return None

    def send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        try:
            if self.alert_manager:
                self.alert_manager.send_alert(alert)
                logger.info(f"告警已发送: {alert.alert_id}")
                return True
            else:
                logger.warning("告警管理器未配置，告警未发送")
                return False
        except Exception as e:
            logger.error(f"发送告警失败 {alert.alert_id}: {e}")
            return False

    def _generate_description(self, rule: "AlertRule", evaluation_result: Dict[str, Any]) -> str:
        """生成告警描述"""
        current_value = evaluation_result.get('current_value', 0)
        threshold = evaluation_result.get('threshold', 0)

        condition_desc = {
            'gt': f'大于 {threshold}',
            'lt': f'小于 {threshold}',
            'gte': f'大于等于 {threshold}',
            'lte': f'小于等于 {threshold}',
            'eq': f'等于 {threshold}',
            'ne': f'不等于 {threshold}'
        }.get(rule.condition, rule.condition)

        return f"{rule.description}\n当前值: {current_value}, 条件: {condition_desc}"

    def _generate_labels(self, rule: "AlertRule", evaluation_result: Dict[str, Any]) -> Dict[str, str]:
        """生成告警标签"""
        labels = {
            'alertname': rule.name,
            'severity': rule.severity.value,
            'rule_id': rule.rule_id,
            'metric_name': rule.metric_name,
            'service': rule.labels.get('service', 'unknown')
        }

        # 添加规则自定义标签
        labels.update(rule.labels)
        return labels

    def _generate_annotations(self, rule: "AlertRule", evaluation_result: Dict[str, Any]) -> Dict[str, str]:
        """生成告警注释"""
        return {
            'summary': rule.name,
            'description': self._generate_description(rule, evaluation_result),
            'value': str(evaluation_result.get('current_value', 0)),
            'threshold': str(evaluation_result.get('threshold', 0)),
            'runbook_url': rule.runbook_url or ""
        }


class ThresholdAdjuster:
    """
    阈值调整器 - 专门负责阈值的自动调整

    单一职责：根据历史数据和趋势自动调整告警阈值
    """

    def __init__(self):
        self._adjustment_history: Dict[str, List[Dict[str, Any]]] = {}
        self._baseline_data: Dict[str, List[float]] = {}

    def adjust_threshold(self, rule: "AlertRule", recent_values: List[float]) -> Optional[float]:
        """
        调整规则阈值

        Args:
            rule: 告警规则
            recent_values: 最近的指标值列表

        Returns:
            新的阈值或None（不需要调整）
        """
        if len(recent_values) < rule.baseline_period:
            return None  # 数据不足，不调整

        try:
            # 计算统计信息
            stats = self._calculate_statistics(recent_values)

            # 判断是否需要调整
            if not self._should_adjust_threshold(rule, stats):
                return None

            # 计算新阈值
            new_threshold = self._calculate_new_threshold(rule, stats)

            # 验证新阈值合理性
            if self._is_threshold_reasonable(rule, new_threshold, stats):
                self._record_adjustment(rule.rule_id, rule.threshold, new_threshold, stats)
                return new_threshold

            return None

        except Exception as e:
            logger.error(f"阈值调整失败 {rule.rule_id}: {e}")
            return None

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """计算统计信息"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

        sorted_values = sorted(values)
        n = len(values)

        return {
            'mean': sum(values) / n,
            'std': (sum((x - sum(values)/n)**2 for x in values) / n) ** 0.5,
            'min': sorted_values[0],
            'max': sorted_values[-1],
            'median': sorted_values[n//2],
            'q25': sorted_values[n//4],
            'q75': sorted_values[3*n//4]
        }

    def _should_adjust_threshold(self, rule: "AlertRule", stats: Dict[str, float]) -> bool:
        """判断是否需要调整阈值"""
        # 检查数据是否有足够的变异性
        if stats['std'] / stats['mean'] < 0.1:  # 变异系数小于10%
            return False

        # 检查是否经常触发告警（需要调高阈值）
        # 或很少触发告警（需要调低阈值）
        # 这里可以根据历史告警频率来判断

        return True  # 简化逻辑，总是允许调整

    def _calculate_new_threshold(self, rule: "AlertRule", stats: Dict[str, float]) -> float:
        """计算新阈值"""
        if rule.adjustment_strategy == 'percentile':
            # 基于百分位数
            percentile = rule.adjustment_percentile or 0.95
            if rule.condition in ['gt', 'gte']:
                # 对于大于条件，使用高百分位数
                return stats['q75'] if percentile >= 0.75 else stats['median']
            else:
                # 对于小于条件，使用低百分位数
                return stats['q25'] if percentile <= 0.25 else stats['median']

        elif rule.adjustment_strategy == 'sigma':
            # 基于标准差
            sigma_factor = rule.adjustment_sigma or 2.0
            if rule.condition in ['gt', 'gte']:
                return stats['mean'] + sigma_factor * stats['std']
            else:
                return stats['mean'] - sigma_factor * stats['std']

        else:
            # 默认策略：保持当前阈值
            return rule.threshold

    def _is_threshold_reasonable(self, rule: "AlertRule", new_threshold: float, stats: Dict[str, float]) -> bool:
        """检查新阈值是否合理"""
        # 阈值不能超出数据范围太多
        data_range = stats['max'] - stats['min']
        if abs(new_threshold - stats['mean']) > 2 * data_range:
            return False

        # 阈值不能为负（对于某些指标）
        if rule.metric_name.endswith('_count') and new_threshold < 0:
            return False

        return True

    def _record_adjustment(self, rule_id: str, old_threshold: float, new_threshold: float, stats: Dict[str, float]):
        """记录调整历史"""
        if rule_id not in self._adjustment_history:
            self._adjustment_history[rule_id] = []

        self._adjustment_history[rule_id].append({
            'timestamp': datetime.now(),
            'old_threshold': old_threshold,
            'new_threshold': new_threshold,
            'stats': stats
        })

        # 限制历史记录数量
        if len(self._adjustment_history[rule_id]) > 100:
            self._adjustment_history[rule_id] = self._adjustment_history[rule_id][-50:]


class AlertSuppressor:
    """
    告警抑制器 - 专门负责告警的抑制逻辑

    单一职责：根据抑制规则决定是否抑制告警
    """

    def __init__(self):
        self._suppression_rules: Dict[str, Dict[str, Any]] = {}

    def add_suppression_rule(self, rule_id: str, suppression_config: Dict[str, Any]):
        """添加抑制规则"""
        self._suppression_rules[rule_id] = suppression_config
        logger.info(f"添加告警抑制规则: {rule_id}")

    def is_suppressed(self, rule: "AlertRule", alert: Alert) -> bool:
        """
        检查告警是否被抑制

        Args:
            rule: 告警规则
            alert: 告警对象

        Returns:
            是否被抑制
        """
        # 检查时间抑制
        if self._is_time_suppressed(rule, alert):
            return True

        # 检查条件抑制
        if self._is_condition_suppressed(rule, alert):
            return True

        return False

    def _is_time_suppressed(self, rule: "AlertRule", alert: Alert) -> bool:
        """检查时间抑制"""
        now = datetime.now()

        # 检查维护时间窗口
        if rule.maintenance_windows:
            for window in rule.maintenance_windows:
                if window['start'] <= now <= window['end']:
                    logger.info(f"告警被维护时间窗口抑制: {rule.rule_id}")
                    return True

        # 检查静默期
        if rule.silence_period and hasattr(rule, '_last_alert_time'):
            time_since_last = now - rule._last_alert_time
            if time_since_last < timedelta(minutes=rule.silence_period):
                logger.info(f"告警被静默期抑制: {rule.rule_id}")
                return True

        return False

    def _is_condition_suppressed(self, rule: "AlertRule", alert: Alert) -> bool:
        """检查条件抑制"""
        # 这里可以实现更复杂的条件抑制逻辑
        # 例如：基于其他告警的状态、系统状态等

        # 简化实现：检查抑制规则
        suppression_config = self._suppression_rules.get(rule.rule_id)
        if not suppression_config:
            return False

        # 检查标签匹配抑制
        if 'suppressed_by_labels' in suppression_config:
            suppressed_labels = suppression_config['suppressed_by_labels']
            alert_labels = set(alert.labels.keys())

            for label_set in suppressed_labels:
                if set(label_set).issubset(alert_labels):
                    logger.info(f"告警被标签抑制: {rule.rule_id}")
                    return True

        return False

