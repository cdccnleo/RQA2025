# -*- coding: utf-8 -*-
"""
RQA2025 实时监控系统核心服务

提供实时指标收集、性能监控、告警系统和可视化支持
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    duration: int  # 持续时间(秒)
    severity: str  # 'info', 'warning', 'error', 'critical'
    description: str
    enabled: bool = True


@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = {}
        self.collectors = {}
        self.collection_interval = 5  # 5秒间隔
        self._running = False
        self._thread = None

    def register_collector(self, name: str, collector_func: Callable) -> None:
        """注册指标收集器"""
        self.collectors[name] = collector_func
        logger.info(f"Registered metrics collector: {name}")

    def collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_bytes_sent': psutil.net_io_counters().bytes_sent,
                'network_bytes_recv': psutil.net_io_counters().bytes_recv,
                'load_average_1min': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'num_processes': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    def collect_application_metrics(self) -> Dict[str, float]:
        """收集应用指标"""
        process = psutil.Process()
        try:
            return {
                'app_cpu_percent': process.cpu_percent(),
                'app_memory_rss_mb': process.memory_info().rss / 1024 / 1024,
                'app_memory_vms_mb': process.memory_info().vms / 1024 / 1024,
                'app_num_threads': process.num_threads(),
                'app_num_fds': getattr(process, 'num_fds', lambda: 0)(),
                'app_cpu_times_user': process.cpu_times().user,
                'app_cpu_times_system': process.cpu_times().system
            }
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}

    def collect_business_metrics(self) -> Dict[str, float]:
        """收集业务指标"""
        # 这里可以集成业务特定的指标收集
        # 例如: 请求数、响应时间、错误率等
        return {
            'requests_total': getattr(self, '_request_count', 0),
            'requests_per_second': getattr(self, '_requests_per_second', 0.0),
            'errors_total': getattr(self, '_error_count', 0),
            'error_rate': getattr(self, '_error_rate', 0.0),
            'avg_response_time_ms': getattr(self, '_avg_response_time', 0.0)
        }

    def update_business_metric(self, name: str, value: float) -> None:
        """更新业务指标"""
        if name == 'request':
            self._request_count = getattr(self, '_request_count', 0) + 1
        elif name == 'error':
            self._error_count = getattr(self, '_error_count', 0) + 1
        elif name == 'response_time':
            # 计算平均响应时间
            current_count = getattr(self, '_response_time_count', 0) + 1
            current_avg = getattr(self, '_avg_response_time', 0.0)
            new_avg = (current_avg * (current_count - 1) + value) / current_count

            self._response_time_count = current_count
            self._avg_response_time = new_avg

    def collect_all_metrics(self) -> Dict[str, MetricData]:
        """收集所有指标"""
        all_metrics = {}

        # 系统指标
        system_metrics = self.collect_system_metrics()
        for name, value in system_metrics.items():
            all_metrics[name] = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags={'type': 'system'}
            )

        # 应用指标
        app_metrics = self.collect_application_metrics()
        for name, value in app_metrics.items():
            all_metrics[name] = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags={'type': 'application'}
            )

        # 业务指标
        business_metrics = self.collect_business_metrics()
        for name, value in business_metrics.items():
            all_metrics[name] = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags={'type': 'business'}
            )

        # 自定义收集器
        for collector_name, collector_func in self.collectors.items():
            try:
                custom_metrics = collector_func()
                for name, value in custom_metrics.items():
                    all_metrics[f"{collector_name}_{name}"] = MetricData(
                        name=f"{collector_name}_{name}",
                        value=value,
                        timestamp=datetime.now(),
                        tags={'type': 'custom', 'collector': collector_name}
                    )
            except Exception as e:
                logger.error(f"Failed to collect metrics from {collector_name}: {e}")

        return all_metrics

    def start_collection(self) -> None:
        """启动指标收集"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """停止指标收集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Metrics collection stopped")

    def _collection_loop(self) -> None:
        """指标收集循环"""
        while self._running:
            try:
                metrics = self.collect_all_metrics()
                self.metrics.update(metrics)

                # 保持最近1小时的数据
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics = {
                    name: metric for name, metric in self.metrics.items()
                    if metric.timestamp > cutoff_time
                }

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")

            time.sleep(self.collection_interval)


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def check_alerts(self, metrics: Dict[str, MetricData]) -> List[Alert]:
        """检查告警条件"""
        new_alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if rule.metric_name not in metrics:
                continue

            metric = metrics[rule.metric_name]
            current_value = metric.value

            # 检查条件
            if rule.condition == '>' and current_value > rule.threshold:
                triggered = True
            elif rule.condition == '<' and current_value < rule.threshold:
                triggered = True
            elif rule.condition == '>=' and current_value >= rule.threshold:
                triggered = True
            elif rule.condition == '<=' and current_value <= rule.threshold:
                triggered = True
            elif rule.condition == '==' and current_value == rule.threshold:
                triggered = True
            else:
                triggered = False

            alert_key = f"{rule.name}_{rule.metric_name}"

            if triggered:
                # 检查是否已经存在活跃告警
                if alert_key not in self.active_alerts:
                    alert = Alert(
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        message=f"{rule.description}: {current_value} {rule.condition} {rule.threshold}",
                        timestamp=datetime.now()
                    )

                    self.active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    self.alert_history.append(alert)

                    # 触发回调
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

                    logger.warning(f"Alert triggered: {alert.message}")

            else:
                # 如果之前有活跃告警，现在需要解决它
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    del self.active_alerts[alert_key]

                    logger.info(f"Alert resolved: {alert.message}")

        return new_alerts

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]


class RealTimeMonitor:
    """实时监控系统"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self._running = False
        self._monitor_thread = None
        self._alert_thread = None

        # 初始化默认告警规则
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self) -> None:
        """设置默认告警规则"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_percent",
                condition=">",
                threshold=80.0,
                duration=60,
                severity="warning",
                description="CPU使用率过高"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_percent",
                condition=">",
                threshold=85.0,
                duration=60,
                severity="warning",
                description="内存使用率过高"
            ),
            AlertRule(
                name="low_memory_available",
                metric_name="memory_available_mb",
                condition="<",
                threshold=512.0,  # 512MB
                duration=60,
                severity="error",
                description="可用内存不足"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition=">",
                threshold=0.05,  # 5%
                duration=300,
                severity="error",
                description="错误率过高"
            )
        ]

        for rule in default_rules:
            self.alert_manager.add_rule(rule)

    def start_monitoring(self) -> None:
        """启动监控系统"""
        if self._running:
            return

        self._running = True

        # 启动指标收集
        self.metrics_collector.start_collection()

        # 启动告警检查线程
        self._alert_thread = threading.Thread(target=self._alert_check_loop, daemon=True)
        self._alert_thread.start()

        logger.info("Real-time monitoring system started")

    def stop_monitoring(self) -> None:
        """停止监控系统"""
        self._running = False

        self.metrics_collector.stop_collection()

        if self._alert_thread:
            self._alert_thread.join(timeout=5)

        logger.info("Real-time monitoring system stopped")

    def _alert_check_loop(self) -> None:
        """告警检查循环"""
        while self._running:
            try:
                metrics = self.metrics_collector.metrics
                if metrics:
                    self.alert_manager.check_alerts(metrics)
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")

            time.sleep(10)  # 每10秒检查一次

    def get_current_metrics(self) -> Dict[str, MetricData]:
        """获取当前指标"""
        return self.metrics_collector.metrics.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        metrics = self.get_current_metrics()

        status = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'healthy',
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'metrics_count': len(metrics)
        }

        # 检查关键指标
        if metrics.get('cpu_percent', MetricData('', 0, datetime.now())).value > 90:
            status['system_health'] = 'critical'
        elif metrics.get('memory_percent', MetricData('', 0, datetime.now())).value > 90:
            status['system_health'] = 'warning'

        return status

    def update_business_metric(self, name: str, value: float) -> None:
        """更新业务指标"""
        self.metrics_collector.update_business_metric(name, value)

    def add_custom_collector(self, name: str, collector_func: Callable) -> None:
        """添加自定义指标收集器"""
        self.metrics_collector.register_collector(name, collector_func)

    def add_alert_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.alert_manager.add_rule(rule)

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调"""
        self.alert_manager.add_alert_callback(callback)

    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        active_alerts = self.alert_manager.get_active_alerts()
        recent_alerts = self.alert_manager.get_alert_history(hours=1)

        return {
            'active_count': len(active_alerts),
            'active_alerts': [
                {
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            'recent_count': len(recent_alerts),
            'recent_alerts': [
                {
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'resolved': alert.resolved,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts[-10:]  # 最近10个
            ]
        }


# 全局监控实例
_monitor_instance = None


def get_monitor() -> RealTimeMonitor:
    """获取全局监控实例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeMonitor()
    return _monitor_instance


def start_monitoring() -> None:
    """启动全局监控"""
    monitor = get_monitor()
    monitor.start_monitoring()


def stop_monitoring() -> None:
    """停止全局监控"""
    monitor = get_monitor()
    monitor.stop_monitoring()


def update_business_metric(name: str, value: float) -> None:
    """更新业务指标"""
    monitor = get_monitor()
    monitor.update_business_metric(name, value)
