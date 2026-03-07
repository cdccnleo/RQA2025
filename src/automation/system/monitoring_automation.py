"""
Monitoring Automation Module
监控自动化模块

This module provides automated monitoring capabilities for quantitative trading systems
此模块为量化交易系统提供自动化监控能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import psutil
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):

    """Metric types"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    ERROR = "error"


class AlertSeverity(Enum):

    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):

    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class MetricData:

    """
    Metric data class
    指标数据类
    """
    metric_name: str
    metric_type: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:

    """
    Alert data class
    告警数据类
    """
    alert_id: str
    alert_name: str
    severity: str
    status: str
    message: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        return data


class MetricCollector:

    """
    Metric Collector Class
    指标收集器类

    Collects system and application metrics
    收集系统和应用指标
    """

    def __init__(self):
        """
        Initialize metric collector
        初始化指标收集器
        """
        self.collectors = {}
        self.metrics_buffer = deque(maxlen=10000)
        self.collection_interval = 30  # seconds

    def register_collector(self, name: str, collector_func: Callable) -> None:
        """
        Register a metric collector function
        注册指标收集器函数

        Args:
            name: Collector name
                 收集器名称
            collector_func: Function that returns metric data
                           返回指标数据的函数
        """
        self.collectors[name] = collector_func
        logger.info(f"Registered metric collector: {name}")

    def collect_metrics(self) -> List[MetricData]:
        """
        Collect metrics from all registered collectors
        从所有注册的收集器收集指标

        Returns:
            list: List of collected metrics
                  收集的指标列表
        """
        metrics = []
        timestamp = datetime.now()

        for name, collector_func in self.collectors.items():
            try:
                collector_metrics = collector_func()

                if isinstance(collector_metrics, dict):
                    collector_metrics = [collector_metrics]

                for metric_data in collector_metrics:
                    metric = MetricData(
                        metric_name=metric_data.get('name', name),
                        metric_type=metric_data.get('type', 'system'),
                        value=metric_data.get('value'),
                        timestamp=timestamp,
                        tags=metric_data.get('tags', {}),
                        metadata=metric_data.get('metadata', {})
                    )
                    metrics.append(metric)

            except Exception as e:
                logger.error(f"Failed to collect metrics from {name}: {str(e)}")

        # Add to buffer
        self.metrics_buffer.extend(metrics)

        return metrics

    def get_system_metrics(self) -> List[Dict[str, Any]]:
        """
        Get system metrics
        获取系统指标

        Returns:
            list: System metrics
                  系统指标
        """
        try:
            return [
                {
                    'name': 'cpu_percent',
                    'type': 'system',
                    'value': psutil.cpu_percent(interval=1),
                    'tags': {'resource': 'cpu'},
                    'metadata': {'unit': 'percent'}
                },
                {
                    'name': 'memory_percent',
                    'type': 'system',
                    'value': psutil.virtual_memory().percent,
                    'tags': {'resource': 'memory'},
                    'metadata': {'unit': 'percent'}
                },
                {
                    'name': 'disk_usage_percent',
                    'type': 'system',
                    'value': psutil.disk_usage('/').percent,
                    'tags': {'resource': 'disk', 'mount': '/'},
                    'metadata': {'unit': 'percent'}
                },
                {
                    'name': 'network_bytes_sent',
                    'type': 'system',
                    'value': psutil.net_io_counters().bytes_sent,
                    'tags': {'resource': 'network', 'direction': 'sent'},
                    'metadata': {'unit': 'bytes'}
                },
                {
                    'name': 'network_bytes_recv',
                    'type': 'system',
                    'value': psutil.net_io_counters().bytes_recv,
                    'tags': {'resource': 'network', 'direction': 'recv'},
                    'metadata': {'unit': 'bytes'}
                }
            ]
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return []

    def get_application_metrics(self) -> List[Dict[str, Any]]:
        """
        Get application metrics
        获取应用指标

        Returns:
            list: Application metrics
                  应用指标
        """
        # Placeholder for application - specific metrics
        return [
            {
                'name': 'active_connections',
                'type': 'application',
                'value': 150,
                'tags': {'app': 'quant_trading'},
                'metadata': {'unit': 'count'}
            },
            {
                'name': 'request_rate',
                'type': 'application',
                'value': 25.5,
                'tags': {'app': 'quant_trading', 'endpoint': 'api'},
                'metadata': {'unit': 'requests_per_second'}
            }
        ]

    def get_business_metrics(self) -> List[Dict[str, Any]]:
        """
        Get business metrics
        获取业务指标

        Returns:
            list: Business metrics
                  业务指标
        """
        # Placeholder for business - specific metrics
        return [
            {
                'name': 'daily_pnl',
                'type': 'business',
                'value': 1250.75,
                'tags': {'portfolio': 'main'},
                'metadata': {'unit': 'currency', 'currency': 'USD'}
            },
            {
                'name': 'trade_count',
                'type': 'business',
                'value': 45,
                'tags': {'portfolio': 'main'},
                'metadata': {'unit': 'count'}
            }
        ]


class AlertManager:

    """
    Alert Manager Class
    告警管理器类

    Manages alerts and notifications
    管理告警和通知
    """

    def __init__(self):
        """
        Initialize alert manager
        初始化告警管理器
        """
        self.alerts = {}
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_history = deque(maxlen=1000)

    def define_alert_rule(self,


                          rule_id: str,
                          name: str,
                          metric_name: str,
                          condition: str,
                          threshold: float,
                          severity: AlertSeverity,
                          description: str = "") -> None:
        """
        Define an alert rule
        定义告警规则

        Args:
            rule_id: Unique rule identifier
                    唯一规则标识符
            name: Rule name
                 规则名称
            metric_name: Metric to monitor
                        要监控的指标
            condition: Alert condition ('>', '<', '>=', '<=', '==', '!=')
                      告警条件 ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
                      阈值
            severity: Alert severity
                     告警严重程度
            description: Rule description
                        规则描述
        """
        rule = {
            'rule_id': rule_id,
            'name': name,
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'severity': severity.value,
            'description': description,
            'enabled': True,
            'created_at': datetime.now()
        }

        self.alert_rules[rule_id] = rule
        logger.info(f"Defined alert rule: {name} ({rule_id})")

    def evaluate_alerts(self, metrics: List[MetricData]) -> List[Alert]:
        """
        Evaluate metrics against alert rules
        根据告警规则评估指标

        Args:
            metrics: List of metrics to evaluate
                    要评估的指标列表

        Returns:
            list: List of triggered alerts
                  触发的告警列表
        """
        alerts = []

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.metric_name].append(metric)

        for rule in self.alert_rules.values():
            if not rule['enabled']:
                continue

            rule_id = rule['rule_id']
            metric_name = rule['metric_name']
            condition = rule['condition']
            threshold = rule['threshold']
            severity = rule['severity']

            if metric_name in metrics_by_name:
                latest_metric = metrics_by_name[metric_name][-1]  # Use latest metric

                if self._evaluate_condition(latest_metric.value, condition, threshold):
                    # Check if alert already exists and is active
                    alert_key = f"{rule_id}_{metric_name}"
                    if alert_key not in self.alerts or self.alerts[alert_key].status != AlertStatus.ACTIVE.value:
                        alert = Alert(
                            alert_id=f"alert_{alert_key}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                            alert_name=rule['name'],
                            severity=severity,
                            status=AlertStatus.ACTIVE.value,
                            message=f"{metric_name} {condition} {threshold} (current: {latest_metric.value})",
                            created_at=datetime.now(),
                            metric_name=metric_name,
                            threshold_value=threshold,
                            actual_value=latest_metric.value,
                            tags=latest_metric.tags
                        )

                        self.alerts[alert_key] = alert
                        alerts.append(alert)

                        # Send notifications
                        self._send_alert_notifications(alert)

        return alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """
        Evaluate alert condition
        评估告警条件

        Args:
            value: Metric value
                  指标值
            condition: Condition operator
                      条件运算符
            threshold: Threshold value
                      阈值

        Returns:
            bool: True if condition is met
                  如果满足条件则返回True
        """
        try:
            if condition == '>':
                return value > threshold
            elif condition == '<':
                return value < threshold
            elif condition == '>=':
                return value >= threshold
            elif condition == '<=':
                return value <= threshold
            elif condition == '==':
                return value == threshold
            elif condition == '!=':
                return value != threshold
            else:
                return False
        except Exception:
            return False

    def acknowledge_alert(self, alert_key: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert
        确认告警

        Args:
            alert_key: Alert key
                      告警键
            acknowledged_by: User acknowledging the alert
                           确认告警的用户

        Returns:
            bool: True if acknowledged successfully
                  确认成功返回True
        """
        if alert_key in self.alerts:
            alert = self.alerts[alert_key]
            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            return True
        return False

    def resolve_alert(self, alert_key: str) -> bool:
        """
        Resolve an alert
        解决告警

        Args:
            alert_key: Alert key
                      告警键

        Returns:
            bool: True if resolved successfully
                  解决成功返回True
        """
        if alert_key in self.alerts:
            alert = self.alerts[alert_key]
            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = datetime.now()
            return True
        return False

    def register_notification_channel(self,


                                      channel_name: str,
                                      notification_func: Callable) -> None:
        """
        Register a notification channel
        注册通知渠道

        Args:
            channel_name: Channel name
                         渠道名称
            notification_func: Function to send notifications
                              发送通知的函数
        """
        self.notification_channels[channel_name] = notification_func
        logger.info(f"Registered notification channel: {channel_name}")

    def _send_alert_notifications(self, alert: Alert) -> None:
        """
        Send alert notifications
        发送告警通知

        Args:
            alert: Alert to notify about
                  要通知的告警
        """
        for channel_name, notification_func in self.notification_channels.items():
            try:
                notification_func(alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {str(e)}")


class DashboardManager:

    """
    Dashboard Manager Class
    仪表板管理器类

    Manages monitoring dashboards and visualizations
    管理监控仪表板和可视化
    """

    def __init__(self):
        """
        Initialize dashboard manager
        初始化仪表板管理器
        """
        self.dashboards = {}
        self.widgets = {}

    def create_dashboard(self,


                         dashboard_id: str,
                         name: str,
                         description: str,
                         widgets: List[Dict[str, Any]]) -> str:
        """
        Create a monitoring dashboard
        创建监控仪表板

        Args:
            dashboard_id: Unique dashboard identifier
                         唯一仪表板标识符
            name: Dashboard name
                 仪表板名称
            description: Dashboard description
                        仪表板描述
            widgets: List of dashboard widgets
                    仪表板组件列表

        Returns:
            str: Created dashboard ID
                 创建的仪表板ID
        """
        dashboard = {
            'dashboard_id': dashboard_id,
            'name': name,
            'description': description,
            'widgets': widgets,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }

        self.dashboards[dashboard_id] = dashboard
        logger.info(f"Created dashboard: {name} ({dashboard_id})")
        return dashboard_id

    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dashboard data
        获取仪表板数据

        Args:
            dashboard_id: Dashboard identifier
                         仪表板标识符

        Returns:
            dict: Dashboard data or None
                  仪表板数据或None
        """
        if dashboard_id in self.dashboards:
            return self.dashboards[dashboard_id]
        return None


class MonitoringAutomationEngine:

    """
    Monitoring Automation Engine Class
    监控自动化引擎类

    Core engine for automated monitoring and alerting
    自动化监控和告警的核心引擎
    """

    def __init__(self, engine_name: str = "default_monitoring_engine"):
        """
        Initialize monitoring automation engine
        初始化监控自动化引擎

        Args:
            engine_name: Name of the engine
                        引擎名称
        """
        self.engine_name = engine_name
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Sub - managers
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.dashboard_manager = DashboardManager()

        # Configuration
        self.collection_interval = 30  # seconds
        self.alert_evaluation_interval = 60  # seconds
        self.retention_period = timedelta(days=7)

        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0
        }

        # Initialize default collectors
        self._setup_default_collectors()

        logger.info(f"Monitoring automation engine {engine_name} initialized")

    def _setup_default_collectors(self) -> None:
        """Setup default metric collectors"""
        self.metric_collector.register_collector('system', self.metric_collector.get_system_metrics)
        self.metric_collector.register_collector(
            'application', self.metric_collector.get_application_metrics)
        self.metric_collector.register_collector(
            'business', self.metric_collector.get_business_metrics)

    def start_monitoring(self) -> bool:
        """
        Start the monitoring automation engine
        启动监控自动化引擎

        Returns:
            bool: True if started successfully
                  启动成功返回True
        """
        if self.is_running:
            logger.warning("Monitoring engine is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Monitoring automation engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring engine: {str(e)}")
            self.is_running = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop the monitoring automation engine
        停止监控自动化引擎

        Returns:
            bool: True if stopped successfully
                  停止成功返回True
        """
        if not self.is_running:
            logger.warning("Monitoring engine is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Monitoring automation engine stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop monitoring engine: {str(e)}")
            return False

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info("Monitoring loop started")

        last_alert_evaluation = datetime.now()

        while self.is_running:
            try:
                # Collect metrics
                metrics = self.metric_collector.collect_metrics()
                self.stats['metrics_collected'] += len(metrics)

                # Evaluate alerts periodically
                now = datetime.now()
                if (now - last_alert_evaluation).seconds >= self.alert_evaluation_interval:
                    alerts = self.alert_manager.evaluate_alerts(metrics)
                    self.stats['alerts_triggered'] += len(alerts)

                    # Check for resolved alerts
                    resolved_count = self._check_resolved_alerts(metrics)
                    self.stats['alerts_resolved'] += resolved_count

                    last_alert_evaluation = now

                # Sleep before next collection
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(self.collection_interval)

        logger.info("Monitoring loop stopped")

    def _check_resolved_alerts(self, metrics: List[MetricData]) -> int:
        """
        Check for resolved alerts
        检查已解决的告警

        Args:
            metrics: Current metrics
                    当前指标

        Returns:
            int: Number of resolved alerts
                 已解决告警的数量
        """
        resolved_count = 0

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.metric_name].append(metric)

        for alert_key, alert in self.alert_manager.alerts.items():
            if alert.status != AlertStatus.ACTIVE.value:
                continue

            metric_name = alert.metric_name
            if metric_name in metrics_by_name:
                latest_metric = metrics_by_name[metric_name][-1]

                # Check if alert condition is no longer met
                rule = self.alert_manager.alert_rules.get(alert.alert_name)
                if rule:
                    condition_met = self.alert_manager._evaluate_condition(
                        latest_metric.value, rule['condition'], rule['threshold']
                    )

                    if not condition_met:
                        self.alert_manager.resolve_alert(alert_key)
                        resolved_count += 1

        return resolved_count

    def define_alert_rule(self,


                          rule_id: str,
                          name: str,
                          metric_name: str,
                          condition: str,
                          threshold: float,
                          severity: AlertSeverity,
                          description: str = "") -> None:
        """
        Define an alert rule
        定义告警规则

        Args:
            rule_id: Rule identifier
                    规则标识符
            name: Rule name
                 规则名称
            metric_name: Metric to monitor
                        要监控的指标
            condition: Alert condition
                      告警条件
            threshold: Threshold value
                      阈值
            severity: Alert severity
                     告警严重程度
            description: Rule description
                        规则描述
        """
        self.alert_manager.define_alert_rule(
            rule_id, name, metric_name, condition, threshold, severity, description
        )

    def register_notification_channel(self,


                                      channel_name: str,
                                      notification_func: Callable) -> None:
        """
        Register a notification channel
        注册通知渠道

        Args:
            channel_name: Channel name
                         渠道名称
            notification_func: Notification function
                              通知函数
        """
        self.alert_manager.register_notification_channel(channel_name, notification_func)

    def create_dashboard(self,


                         dashboard_id: str,
                         name: str,
                         description: str,
                         widgets: List[Dict[str, Any]]) -> str:
        """
        Create a monitoring dashboard
        创建监控仪表板

        Args:
            dashboard_id: Dashboard identifier
                         仪表板标识符
            name: Dashboard name
                 仪表板名称
            description: Dashboard description
                        仪表板描述
            widgets: Dashboard widgets
                    仪表板组件

        Returns:
            str: Created dashboard ID
                 创建的仪表板ID
        """
        return self.dashboard_manager.create_dashboard(
            dashboard_id, name, description, widgets
        )

    def get_metrics_history(self,


                            metric_name: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[MetricData]:
        """
        Get metrics history
        获取指标历史

        Args:
            metric_name: Specific metric name (optional)
                        特定指标名称（可选）
            start_time: Start time for history (optional)
                       历史开始时间（可选）
            end_time: End time for history (optional)
                     历史结束时间（可选）

        Returns:
            list: Metrics history
                  指标历史
        """
        metrics = list(self.metric_collector.metrics_buffer)

        # Filter by metric name
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]

        # Filter by time range
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        return metrics

    def get_active_alerts(self) -> List[Alert]:
        """
        Get active alerts
        获取活跃告警

        Returns:
            list: Active alerts
                  活跃告警
        """
        return [
            alert for alert in self.alert_manager.alerts.values()
            if alert.status == AlertStatus.ACTIVE.value
        ]

    def acknowledge_alert(self, alert_key: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert
        确认告警

        Args:
            alert_key: Alert key
                      告警键
            acknowledged_by: User acknowledging the alert
                           确认告警的用户

        Returns:
            bool: True if acknowledged
                  确认成功返回True
        """
        return self.alert_manager.acknowledge_alert(alert_key, acknowledged_by)

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get monitoring engine statistics
        获取监控引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'stats': self.stats,
            'active_alerts': len(self.get_active_alerts()),
            'total_dashboards': len(self.dashboard_manager.dashboards),
            'total_alert_rules': len(self.alert_manager.alert_rules),
            'metrics_buffer_size': len(self.metric_collector.metrics_buffer)
        }


# Global monitoring automation engine instance
# 全局监控自动化引擎实例
monitoring_engine = MonitoringAutomationEngine()

# Default notification channels
# 默认通知渠道


def log_notification(alert: Alert) -> None:
    """Log notification channel"""
    logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")


def email_notification(alert: Alert) -> None:
    """Email notification channel (placeholder)"""
    logger.info(f"Email notification: {alert.message}")


# Register default notification channels
# 注册默认通知渠道
monitoring_engine.register_notification_channel('log', log_notification)
monitoring_engine.register_notification_channel('email', email_notification)

__all__ = [
    'MetricType',
    'AlertSeverity',
    'AlertStatus',
    'MetricData',
    'Alert',
    'MetricCollector',
    'AlertManager',
    'DashboardManager',
    'MonitoringAutomationEngine',
    'monitoring_engine'
]
