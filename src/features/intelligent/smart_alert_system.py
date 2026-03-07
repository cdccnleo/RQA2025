# src / features / intelligent / smart_alert_system.py
"""
智能告警系统模块
实现智能化的告警功能，包括异常检测、趋势分析、自适应阈值等
"""

import logging
from typing import Optional, List, Dict, Any, Union, Callable
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):

    """告警类型"""
    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    PERFORMANCE = "performance"


@dataclass
class AlertRule:

    """告警规则"""
    name: str
    alert_type: AlertType
    metric: str
    condition: str
    threshold: float
    level: AlertLevel
    enabled: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:

    """告警信息"""
    id: str
    rule_name: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartAlertSystem:

    """智能告警系统"""

    def __init__(


        self,
        config_manager=None,
        alert_history_size: int = 1000,
        enable_adaptive_thresholds: bool = True,
        enable_trend_analysis: bool = True,
        enable_anomaly_detection: bool = True
    ):
        """
        初始化智能告警系统

        Args:
            config_manager: 配置管理器
            alert_history_size: 告警历史记录大小
            enable_adaptive_thresholds: 是否启用自适应阈值
            enable_trend_analysis: 是否启用趋势分析
            enable_anomaly_detection: 是否启用异常检测
        """
        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.MONITORING, self._on_config_change)

        # 系统参数
        self.alert_history_size = alert_history_size
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_anomaly_detection = enable_anomaly_detection

        # 数据存储
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.metric_history: Dict[str, List[float]] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}

        # 回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        logger.info("智能告警系统初始化完成")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.MONITORING:
            if key == "alert_history_size":
                self.alert_history_size = value
                logger.info(f"更新告警历史记录大小: {value}")
            elif key == "enable_adaptive_thresholds":
                self.enable_adaptive_thresholds = value
                logger.info(f"更新自适应阈值状态: {value}")

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def update_rule(self, rule_name: str, **kwargs) -> None:
        """更新告警规则"""
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"更新告警规则: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
        logger.info("添加告警回调函数")

    def check_metric(self, metric: str, value: float, timestamp: Optional[datetime] = None) -> List[Alert]:
        """检查指标值并触发告警"""
        if timestamp is None:
            timestamp = datetime.now()

        # 更新指标历史
        if metric not in self.metric_history:
            self.metric_history[metric] = []
        self.metric_history[metric].append(value)

        # 保持历史记录大小
        if len(self.metric_history[metric]) > self.alert_history_size:
            self.metric_history[metric] = self.metric_history[metric][-self.alert_history_size:]

        # 检查所有相关规则
        triggered_alerts = []

        for rule_name, rule in self.rules.items():
            if not rule.enabled or rule.metric != metric:
                continue

            # 检查冷却时间
            if rule.last_triggered and (timestamp - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60:
                continue

            # 根据规则类型检查
            if rule.alert_type == AlertType.THRESHOLD:
                alert = self._check_threshold_rule(rule, value, timestamp)
            elif rule.alert_type == AlertType.TREND:
                alert = self._check_trend_rule(rule, metric, value, timestamp)
            elif rule.alert_type == AlertType.ANOMALY:
                alert = self._check_anomaly_rule(rule, metric, value, timestamp)
            elif rule.alert_type == AlertType.PATTERN:
                alert = self._check_pattern_rule(rule, metric, value, timestamp)
            elif rule.alert_type == AlertType.PERFORMANCE:
                alert = self._check_performance_rule(rule, metric, value, timestamp)
            else:
                continue

            if alert:
                triggered_alerts.append(alert)
                rule.last_triggered = timestamp

        # 触发回调函数
        for alert in triggered_alerts:
            # 将告警添加到历史记录中
            self.alerts.append(alert)

            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调函数执行失败: {e}")

        return triggered_alerts

    def _check_threshold_rule(self, rule: AlertRule, value: float, timestamp: datetime) -> Optional[Alert]:
        """检查阈值规则"""
        threshold = rule.threshold

        # 自适应阈值
        if self.enable_adaptive_thresholds:
            adaptive_threshold = self._get_adaptive_threshold(rule.metric)
            if adaptive_threshold:
                threshold = adaptive_threshold

        # 检查条件
        triggered = False
        if rule.condition == ">":
            triggered = value > threshold
        elif rule.condition == ">=":
            triggered = value >= threshold
        elif rule.condition == "<":
            triggered = value < threshold
        elif rule.condition == "<=":
            triggered = value <= threshold
        elif rule.condition == "==":
            triggered = value == threshold
        elif rule.condition == "!=":
            triggered = value != threshold

        if triggered:
            alert = Alert(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                rule_name=rule.name,
                alert_type=rule.alert_type,
                level=rule.level,
                message=f"{rule.metric} {rule.condition} {threshold} (当前值: {value})",
                metric=rule.metric,
                value=value,
                threshold=threshold,
                timestamp=timestamp,
                metadata={
                    "description": rule.description,
                    "tags": rule.tags,
                    "adaptive_threshold": self.enable_adaptive_thresholds
                }
            )
            return alert

        return None

    def _check_trend_rule(self, rule: AlertRule, metric: str, value: float, timestamp: datetime) -> Optional[Alert]:
        """检查趋势规则"""
        if not self.enable_trend_analysis:
            return None

        history = self.metric_history.get(metric, [])
        if len(history) < 10:  # 需要足够的历史数据
            return None

        # 计算趋势
        recent_values = history[-10:]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        # 检查趋势条件
        triggered = False
        if rule.condition == "increasing" and trend > 0.1:
            triggered = True
        elif rule.condition == "decreasing" and trend < -0.1:
            triggered = True
        elif rule.condition == "stable" and abs(trend) < 0.05:
            triggered = True

        if triggered:
            alert = Alert(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                rule_name=rule.name,
                alert_type=rule.alert_type,
                level=rule.level,
                message=f"{metric} 趋势: {rule.condition} (斜率: {trend:.3f})",
                metric=metric,
                value=value,
                threshold=trend,
                timestamp=timestamp,
                metadata={
                    "description": rule.description,
                    "tags": rule.tags,
                    "trend": trend,
                    "recent_values": recent_values
                }
            )
            return alert

        return None

    def _check_anomaly_rule(self, rule: AlertRule, metric: str, value: float, timestamp: datetime) -> Optional[Alert]:
        """检查异常规则"""
        if not self.enable_anomaly_detection:
            return None

        history = self.metric_history.get(metric, [])
        if len(history) < 20:  # 需要足够的历史数据
            return None

        # 计算统计量
        mean_val = np.mean(history)
        std_val = np.std(history)

        if std_val == 0:
            return None

        # 计算Z - score
        z_score = abs((value - mean_val) / std_val)

        # 检查是否为异常值
        threshold = rule.threshold if rule.threshold > 0 else 3.0  # 默认3个标准差
        triggered = z_score > threshold

        if triggered:
            alert = Alert(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                rule_name=rule.name,
                alert_type=rule.alert_type,
                level=rule.level,
                message=f"{metric} 异常检测: Z - score={z_score:.2f} > {threshold}",
                metric=metric,
                value=value,
                threshold=threshold,
                timestamp=timestamp,
                metadata={
                    "description": rule.description,
                    "tags": rule.tags,
                    "z_score": z_score,
                    "mean": mean_val,
                    "std": std_val
                }
            )
            return alert

        return None

    def _check_pattern_rule(self, rule: AlertRule, metric: str, value: float, timestamp: datetime) -> Optional[Alert]:
        """检查模式规则"""
        history = self.metric_history.get(metric, [])
        if len(history) < 10:
            return None

        # 简单的模式检测：检查是否连续上升或下降
        recent_values = history[-5:]

        # 检查连续上升
        if rule.condition == "consecutive_increase":
            triggered = all(recent_values[i] < recent_values[i + 1]
                            for i in range(len(recent_values) - 1))
        # 检查连续下降
        elif rule.condition == "consecutive_decrease":
            triggered = all(recent_values[i] > recent_values[i + 1]
                            for i in range(len(recent_values) - 1))
        # 检查震荡
        elif rule.condition == "oscillating":
            diffs = [recent_values[i + 1] - recent_values[i] for i in range(len(recent_values) - 1)]
            triggered = any(d > 0 for d in diffs) and any(d < 0 for d in diffs)
        else:
            return None

        if triggered:
            alert = Alert(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                rule_name=rule.name,
                alert_type=rule.alert_type,
                level=rule.level,
                message=f"{metric} 模式: {rule.condition}",
                metric=metric,
                value=value,
                threshold=0,
                timestamp=timestamp,
                metadata={
                    "description": rule.description,
                    "tags": rule.tags,
                    "recent_values": recent_values,
                    "pattern": rule.condition
                }
            )
            return alert

        return None

    def _check_performance_rule(self, rule: AlertRule, metric: str, value: float, timestamp: datetime) -> Optional[Alert]:
        """检查性能规则"""
        # 性能相关的特殊检查
        if "response_time" in metric.lower():
            # 响应时间检查
            if value > rule.threshold:
                alert = Alert(
                    id=f"{rule.name}_{timestamp.strftime('%Y % m % d_ % H % M % S')}",
                    rule_name=rule.name,
                    alert_type=rule.alert_type,
                    level=rule.level,
                    message=f"响应时间过长: {value}ms > {rule.threshold}ms",
                    metric=metric,
                    value=value,
                    threshold=rule.threshold,
                    timestamp=timestamp,
                    metadata={
                        "description": rule.description,
                        "tags": rule.tags,
                        "performance_issue": "high_response_time"
                    }
                )
                return alert

        elif "error_rate" in metric.lower():
            # 错误率检查
            if value > rule.threshold:
                alert = Alert(
                    id=f"{rule.name}_{timestamp.strftime('%Y % m % d_ % H % M % S')}",
                    rule_name=rule.name,
                    alert_type=rule.alert_type,
                    level=rule.level,
                    message=f"错误率过高: {value:.2%} > {rule.threshold:.2%}",
                    metric=metric,
                    value=value,
                    threshold=rule.threshold,
                    timestamp=timestamp,
                    metadata={
                        "description": rule.description,
                        "tags": rule.tags,
                        "performance_issue": "high_error_rate"
                    }
                )
                return alert

        return None

    def _get_adaptive_threshold(self, metric: str) -> Optional[float]:
        """获取自适应阈值"""
        if metric not in self.adaptive_thresholds:
            return None

        thresholds = self.adaptive_thresholds[metric]
        history = self.metric_history.get(metric, [])

        if len(history) < 20:
            return None

        # 基于历史数据的自适应阈值
        mean_val = np.mean(history)
        std_val = np.std(history)

        # 动态调整阈值
        adaptive_threshold = mean_val + 2 * std_val

        return adaptive_threshold

    def get_alerts(


        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[AlertLevel] = None,
        rule_name: Optional[str] = None
    ) -> List[Alert]:
        """获取告警历史"""
        filtered_alerts = self.alerts

        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]

        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]

        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]

        if rule_name:
            filtered_alerts = [a for a in filtered_alerts if a.rule_name == rule_name]

        return filtered_alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        if not self.alerts:
            return {}

        # 按级别统计
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = len([a for a in self.alerts if a.level == level])

        # 按规则统计
        rule_counts = {}
        for alert in self.alerts:
            rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1

        # 按时间统计（最近24小时）
        now = datetime.now()
        recent_alerts = [a for a in self.alerts if (now - a.timestamp).total_seconds() < 86400]

        return {
            "total_alerts": len(self.alerts),
            "recent_alerts_24h": len(recent_alerts),
            "level_counts": level_counts,
            "rule_counts": rule_counts,
            "active_rules": len([r for r in self.rules.values() if r.enabled])
        }

    def save_rules(self, filepath: Union[str, Path]) -> None:
        """保存告警规则"""
        rules_data = []
        for rule in self.rules.values():
            rule_dict = {
                "name": rule.name,
                "alert_type": rule.alert_type.value,
                "metric": rule.metric,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "level": rule.level.value,
                "enabled": rule.enabled,
                "description": rule.description,
                "tags": rule.tags,
                "cooldown_minutes": rule.cooldown_minutes
            }
            rules_data.append(rule_dict)

        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        logger.info(f"告警规则已保存到: {filepath}")

    def load_rules(self, filepath: Union[str, Path]) -> None:
        """加载告警规则"""
        with open(filepath, 'r', encoding='utf - 8') as f:
            rules_data = json.load(f)

        self.rules.clear()
        for rule_dict in rules_data:
            rule = AlertRule(
                name=rule_dict["name"],
                alert_type=AlertType(rule_dict["alert_type"]),
                metric=rule_dict["metric"],
                condition=rule_dict["condition"],
                threshold=rule_dict["threshold"],
                level=AlertLevel(rule_dict["level"]),
                enabled=rule_dict["enabled"],
                description=rule_dict["description"],
                tags=rule_dict["tags"],
                cooldown_minutes=rule_dict["cooldown_minutes"]
            )
            self.rules[rule.name] = rule

        logger.info(f"告警规则已从 {filepath} 加载")

    def create_default_rules(self) -> None:
        """创建默认告警规则"""

        default_rules = [
            AlertRule(
                name="high_error_rate",
                alert_type=AlertType.PERFORMANCE,
                metric="error_rate",
                condition=">",
                threshold=0.05,
                level=AlertLevel.WARNING,
                description="错误率过高告警",
                tags=["performance", "error"]
            ),
            AlertRule(
                name="high_response_time",
                alert_type=AlertType.PERFORMANCE,
                metric="response_time",
                condition=">",
                threshold=1000,
                level=AlertLevel.WARNING,
                description="响应时间过长告警",
                tags=["performance", "response_time"]
            ),
            AlertRule(
                name="memory_usage_high",
                alert_type=AlertType.THRESHOLD,
                metric="memory_usage",
                condition=">",
                threshold=0.8,
                level=AlertLevel.WARNING,
                description="内存使用率过高告警",
                tags=["system", "memory"]
            ),
            AlertRule(
                name="cpu_usage_high",
                alert_type=AlertType.THRESHOLD,
                metric="cpu_usage",
                condition=">",
                threshold=0.9,
                level=AlertLevel.CRITICAL,
                description="CPU使用率过高告警",
                tags=["system", "cpu"]
            ),
            AlertRule(
                name="anomaly_detection",
                alert_type=AlertType.ANOMALY,
                metric="feature_generation_time",
                condition=">",
                threshold=3.0,
                level=AlertLevel.WARNING,
                description="特征生成时间异常告警",
                tags=["anomaly", "performance"]
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

        logger.info("已创建默认告警规则")
