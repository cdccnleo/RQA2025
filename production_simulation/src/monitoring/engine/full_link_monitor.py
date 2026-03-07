#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
全链路监控系统
实现全链路监控、智能告警机制、性能分析工具和自动化运维功能
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import psutil

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitorType(Enum):

    """监控类型枚举"""
    SYSTEM = "system"           # 系统监控
    APPLICATION = "application"  # 应用监控
    BUSINESS = "business"       # 业务监控
    PERFORMANCE = "performance"  # 性能监控
    CUSTOM = "custom"          # 自定义监控


@dataclass
class MetricData:

    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    monitor_type: MonitorType
    source: str


@dataclass
class AlertRule:

    """告警规则"""
    name: str
    metric_name: str
    condition: str  # 例如: "> 80", "< 10", "== 0"
    level: AlertLevel
    duration: int  # 持续时间(秒)
    enabled: bool = True
    description: str = ""


@dataclass
class Alert:

    """告警信息"""
    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: str
    level: AlertLevel
    timestamp: datetime
    message: str
    resolved: bool = False
    resolved_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:

    """性能指标"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime


class FullLinkMonitor:

    """全链路监控系统"""

    def __init__(self, config: Optional[Dict] = None):

        self.config = config or {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.performance_metrics: deque = deque(maxlen=1000)

        # 监控回调函数
        self.metric_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        # 初始化告警规则
        self._init_default_alert_rules()

        # 启动监控线程
        self._start_monitoring_threads()

        logger.info("全链路监控系统初始化完成")

    def _init_default_alert_rules(self):
        """初始化默认告警规则"""

        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage",
                condition="> 80",
                level=AlertLevel.WARNING,
                duration=300,
                description="CPU使用率超过80%"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage",
                condition="> 85",
                level=AlertLevel.WARNING,
                duration=300,
                description="内存使用率超过85%"
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="disk_usage",
                condition="> 90",
                level=AlertLevel.ERROR,
                duration=300,
                description="磁盘使用率超过90%"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="> 5",
                level=AlertLevel.ERROR,
                duration=60,
                description="错误率超过5%"
            ),
            AlertRule(
                name="high_response_time",
                metric_name="response_time",
                condition="> 1000",
                level=AlertLevel.WARNING,
                duration=300,
                description="响应时间超过1000ms"
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def add_metric(self, metric: MetricData):
        """
        添加指标数据

        Args:
            metric: 指标数据
        """
        # 存储指标
        self.metrics_history[metric.name].append(metric)

        # 检查告警规则
        self._check_alert_rules(metric)

        # 触发回调
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")

    def _check_alert_rules(self, metric: MetricData):
        """检查告警规则"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name != metric.name:
                continue

            # 检查条件
            if self._evaluate_condition(metric.value, rule.condition):
                # 检查持续时间
                if self._check_duration(metric.name, rule):
                    self._trigger_alert(rule, metric)
            else:
                # 清除告警
                self._clear_alert(rule_name)

        # 立即检查告警规则 - 修复：确保告警检查被调用
        self._check_alert_duration()

    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """评估条件"""
        try:
            # 解析条件
            if condition.startswith(">"):
                threshold = float(condition[2:])
                return value > threshold
            elif condition.startswith("<"):
                threshold = float(condition[2:])
                return value < threshold
            elif condition.startswith(">="):
                threshold = float(condition[3:])
                return value >= threshold
            elif condition.startswith("<="):
                threshold = float(condition[3:])
                return value <= threshold
            elif condition.startswith("=="):
                threshold = float(condition[3:])
                return abs(value - threshold) < 1e-6
            elif condition.startswith("!="):
                threshold = float(condition[3:])
                return abs(value - threshold) >= 1e-6
            else:
                return False
        except Exception as e:
            logger.error(f"条件评估失败: {condition}, 错误: {e}")
            return False

    def _check_duration(self, metric_name: str, rule: AlertRule) -> bool:
        """检查持续时间"""
        if rule.duration <= 0:
            return True

        # 获取最近的时间窗口内的指标
        cutoff_time = datetime.now() - timedelta(seconds=rule.duration)
        recent_metrics = [
            metric for metric in self.metrics_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]

        if len(recent_metrics) < 2:
            return False

        # 检查是否在持续时间内都满足条件
        for metric in recent_metrics:
            if not self._evaluate_condition(metric.value, rule.condition):
                return False

        return True

    def _trigger_alert(self, rule: AlertRule, metric: MetricData):
        """触发告警"""
        alert_id = f"{rule.name}_{int(time.time())}"

        # 检查是否已存在相同告警
        if rule.name in self.active_alerts:
            return

        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            metric_name=metric.name,
            current_value=metric.value,
            threshold=rule.condition,
            level=rule.level,
            timestamp=datetime.now(),
            message=f"{rule.description}: 当前值 {metric.value:.2f}, 阈值 {rule.condition}"
        )

        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)

        # 触发告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

        logger.warning(f"触发告警: {alert.message}")

    def _clear_alert(self, rule_name: str):
        """清除告警"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_time = datetime.now()
            del self.active_alerts[rule_name]

            logger.info(f"告警已清除: {alert.message}")

    def collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)

        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100

        # 网络IO
        network_io = psutil.net_io_counters()
        network_data = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }

        # 应用指标 (模拟)
        try:
            response_time = np.random.normal(100, 20)  # 模拟响应时间
            throughput = np.random.normal(1000, 100)   # 模拟吞吐量
            error_rate = max(0, min(1, np.random.normal(0.01, 0.005)))  # 模拟错误率，确保在0 - 1范围内
        except Exception:
            # 如果随机数生成失败，使用固定值
            response_time = 100.0
            throughput = 1000.0
            error_rate = 0.01

        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_data,
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=datetime.now()
        )

        self.performance_metrics.append(metrics)

        # 添加指标数据
        self.add_metric(MetricData(
            name="cpu_usage",
            value=cpu_usage,
            timestamp=datetime.now(),
            tags={"type": "system"},
            monitor_type=MonitorType.SYSTEM,
            source="system"
        ))

        self.add_metric(MetricData(
            name="memory_usage",
            value=memory_usage,
            timestamp=datetime.now(),
            tags={"type": "system"},
            monitor_type=MonitorType.SYSTEM,
            source="system"
        ))

        self.add_metric(MetricData(
            name="disk_usage",
            value=disk_usage,
            timestamp=datetime.now(),
            tags={"type": "system"},
            monitor_type=MonitorType.SYSTEM,
            source="system"
        ))

        self.add_metric(MetricData(
            name="response_time",
            value=response_time,
            timestamp=datetime.now(),
            tags={"type": "application"},
            monitor_type=MonitorType.APPLICATION,
            source="application"
        ))

        self.add_metric(MetricData(
            name="throughput",
            value=throughput,
            timestamp=datetime.now(),
            tags={"type": "application"},
            monitor_type=MonitorType.APPLICATION,
            source="application"
        ))

        self.add_metric(MetricData(
            name="error_rate",
            value=error_rate,
            timestamp=datetime.now(),
            tags={"type": "application"},
            monitor_type=MonitorType.APPLICATION,
            source="application"
        ))

        return metrics

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能报告"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 过滤最近的数据
        recent_metrics = [
            metrics for metrics in self.performance_metrics
            if metrics.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {
                "summary": {
                    "period": {
                        "start": cutoff_time.isoformat(),
                        "end": datetime.now().isoformat(),
                        "hours": hours
                    },
                    "total_metrics": 0,
                    "total_alerts": len(self.alert_history)
                }
            }

        # 计算统计信息
        report = {
            "summary": {
                "period": {
                    "start": cutoff_time.isoformat(),
                    "end": datetime.now().isoformat(),
                    "hours": hours
                },
                "total_metrics": len(recent_metrics),
                "total_alerts": len(self.alert_history)
            },
            "system_metrics": {
                "cpu_usage": {
                    "avg": np.mean([m.cpu_usage for m in recent_metrics]),
                    "max": np.max([m.cpu_usage for m in recent_metrics]),
                    "min": np.min([m.cpu_usage for m in recent_metrics])
                },
                "memory_usage": {
                    "avg": np.mean([m.memory_usage for m in recent_metrics]),
                    "max": np.max([m.memory_usage for m in recent_metrics]),
                    "min": np.min([m.memory_usage for m in recent_metrics])
                },
                "disk_usage": {
                    "avg": np.mean([m.disk_usage for m in recent_metrics]),
                    "max": np.max([m.disk_usage for m in recent_metrics]),
                    "min": np.min([m.disk_usage for m in recent_metrics])
                }
            },
            "application_metrics": {
                "response_time": {
                    "avg": np.mean([m.response_time for m in recent_metrics]),
                    "max": np.max([m.response_time for m in recent_metrics]),
                    "min": np.min([m.response_time for m in recent_metrics])
                },
                "throughput": {
                    "avg": np.mean([m.throughput for m in recent_metrics]),
                    "max": np.max([m.throughput for m in recent_metrics]),
                    "min": np.min([m.throughput for m in recent_metrics])
                },
                "error_rate": {
                    "avg": np.mean([m.error_rate for m in recent_metrics]),
                    "max": np.max([m.error_rate for m in recent_metrics]),
                    "min": np.min([m.error_rate for m in recent_metrics])
                }
            },
            "alerts": {
                "active_count": len(self.active_alerts),
                "total_count": len(self.alert_history),
                "resolved_count": len([a for a in self.alert_history if a.resolved])
            }
        }

        return report

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        # 查找告警并解决
        for rule_name, alert in list(self.active_alerts.items()):
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_time = datetime.now()
                # 从活跃告警中移除
                del self.active_alerts[rule_name]
                logger.info(f"告警已解决: {alert_id}")
                return

        logger.warning(f"未找到告警: {alert_id}")

    def add_metric_callback(self, callback: Callable[[MetricData], None]):
        """添加指标回调"""
        self.metric_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def _start_monitoring_threads(self):
        """启动监控线程"""
        # 系统指标收集线程

        def system_monitor_loop():

            while True:
                try:
                    self.collect_system_metrics()
                    time.sleep(60)  # 每分钟收集一次
                except Exception as e:
                    logger.error(f"系统监控线程错误: {e}")
                    time.sleep(60)

        system_thread = threading.Thread(target=system_monitor_loop, daemon=True)
        system_thread.start()

        # 告警检查线程

        def alert_check_loop():

            while True:
                try:
                    # 检查告警持续时间
                    self._check_alert_duration()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    logger.error(f"告警检查线程错误: {e}")
                    time.sleep(30)

        alert_thread = threading.Thread(target=alert_check_loop, daemon=True)
        alert_thread.start()

    def _check_alert_duration(self):
        """检查告警持续时间"""
        current_time = datetime.now()

        for rule_name, alert in list(self.active_alerts.items()):
            rule = self.alert_rules.get(rule_name)
            if not rule:
                continue

            # 检查告警是否应该自动清除
            if (current_time - alert.timestamp).total_seconds() > rule.duration * 2:
                self._clear_alert(rule_name)

    def export_metrics(self, file_path: str, hours: int = 24):
        """导出指标数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        export_data = {
            "export_time": datetime.now().isoformat(),
            "period": {
                "start": cutoff_time.isoformat(),
                "end": datetime.now().isoformat(),
                "hours": hours
            },
            "metrics": {},
            "alerts": []
        }

        # 导出指标数据
        for metric_name, metrics in self.metrics_history.items():
            recent_metrics = [
                asdict(metric) for metric in metrics
                if metric.timestamp >= cutoff_time
            ]
            export_data["metrics"][metric_name] = recent_metrics

        # 导出告警数据
        recent_alerts = [
            asdict(alert) for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        export_data["alerts"] = recent_alerts

        # 写入文件
        with open(file_path, 'w', encoding='utf - 8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"指标数据已导出到: {file_path}")

    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        # 获取最新的性能指标
        if not self.performance_metrics:
            return {"status": "unknown", "message": "无性能数据"}

        latest_metrics = self.performance_metrics[-1]

        # 评估健康状态
        health_score = 100

        # CPU使用率检查
        if latest_metrics.cpu_usage > 90:
            health_score -= 30
        elif latest_metrics.cpu_usage > 80:
            health_score -= 15

        # 内存使用率检查
        if latest_metrics.memory_usage > 95:
            health_score -= 30
        elif latest_metrics.memory_usage > 85:
            health_score -= 15

        # 磁盘使用率检查
        if latest_metrics.disk_usage > 95:
            health_score -= 30
        elif latest_metrics.disk_usage > 90:
            health_score -= 15

        # 错误率检查
        if latest_metrics.error_rate > 10:
            health_score -= 20
        elif latest_metrics.error_rate > 5:
            health_score -= 10

        # 响应时间检查
        if latest_metrics.response_time > 2000:
            health_score -= 20
        elif latest_metrics.response_time > 1000:
            health_score -= 10

        # 确定状态
        if health_score >= 80:
            status = "healthy"
            message = "系统运行正常"
        elif health_score >= 60:
            status = "warning"
            message = "系统存在警告"
        elif health_score >= 40:
            status = "error"
            message = "系统存在错误"
        else:
            status = "critical"
            message = "系统严重异常"

        return {
            "status": status,
            "health_score": health_score,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(self.performance_metrics),
            "metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_usage": latest_metrics.disk_usage,
                "error_rate": latest_metrics.error_rate,
                "response_time": latest_metrics.response_time
            }
        }
