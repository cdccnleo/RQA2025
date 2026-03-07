"""
Monitoring Processor Module
监控处理器模块

This module provides monitoring and metrics processing capabilities for async operations
此模块为异步操作提供监控和指标处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import time
import json
import csv

logger = logging.getLogger(__name__)


class MetricType(Enum):

    """Metric type enumeration"""
    COUNTER = "counter"        # Monotonically increasing counter
    GAUGE = "gauge"           # Can go up and down
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Similar to histogram but client - side


class AlertSeverity(Enum):

    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricData:

    """
    Metric Data Class
    指标数据类

    Represents a single metric measurement
    表示单个指标测量
    """

    def __init__(self,


                 name: str,
                 value: Union[int, float],
                 metric_type: MetricType,
                 labels: Optional[Dict[str, str]] = None,
                 timestamp: Optional[datetime] = None):
        """
        Initialize metric data
        初始化指标数据

        Args:
            name: Metric name
                 指标名称
            value: Metric value
                  指标值
            metric_type: Type of metric
                        指标类型
            labels: Metric labels / tags
                  指标标签
            timestamp: Metric timestamp
                      指标时间戳
        """
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.labels = labels or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary
        将指标转换为字典

        Returns:
            dict: Metric data as dictionary
                  指标数据字典
        """
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat()
        }


class AlertRule:

    """
    Alert Rule Class
    警报规则类

    Defines conditions for generating alerts based on metrics
    定义基于指标生成警报的条件
    """

    def __init__(self,


                 name: str,
                 metric_name: str,
                 condition: Callable,
                 severity: AlertSeverity,
                 description: str,
                 labels: Optional[Dict[str, str]] = None):
        """
        Initialize alert rule
        初始化警报规则

        Args:
            name: Alert rule name
                 警报规则名称
            metric_name: Name of metric to monitor
                        要监控的指标名称
            condition: Function that returns True if alert should be triggered
                      如果应触发警报则返回True的函数
            severity: Alert severity level
                     警报严重程度
            description: Human - readable description
                        人类可读的描述
            labels: Additional labels for the alert
                   警报的附加标签
        """
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.description = description
        self.labels = labels or {}

        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0

    def evaluate(self, metric_value: Any) -> Optional[Dict[str, Any]]:
        """
        Evaluate the alert rule against a metric value
        根据指标值评估警报规则

        Args:
            metric_value: Current metric value
                         当前指标值

        Returns:
            dict: Alert data if triggered, None otherwise
                  如果触发则返回警报数据，否则返回None
        """
        try:
            if self.condition(metric_value):
                self.trigger_count += 1
                self.last_triggered = datetime.now()

                return {
                    'rule_name': self.name,
                    'metric_name': self.metric_name,
                    'metric_value': metric_value,
                    'severity': self.severity.value,
                    'description': self.description,
                    'labels': self.labels,
                    'timestamp': self.last_triggered,
                    'trigger_count': self.trigger_count
                }

        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {self.name}: {str(e)}")

        return None


class MonitoringProcessor:

    """
    Monitoring Processor Class
    监控处理器类

    Processes metrics and generates alerts for async operations
    处理异步操作的指标并生成警报
    """

    def __init__(self, processor_name: str = "default_monitoring_processor"):
        """
        Initialize the monitoring processor
        初始化监控处理器

        Args:
            processor_name: Name of this processor
                          此处理器的名称
        """
        self.processor_name = processor_name
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.is_running = False
        self.monitoring_thread = None

        # Configuration
        self.collection_interval = 10  # seconds
        self.alert_check_interval = 30  # seconds
        self.max_alerts_history = 1000

        logger.info(f"Monitoring processor {processor_name} initialized")

    def start_monitoring(self) -> bool:
        """
        Start monitoring and alert processing
        开始监控和警报处理

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.processor_name} is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info(f"Monitoring started for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            self.is_running = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop monitoring and alert processing
        停止监控和警报处理

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.processor_name} is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info(f"Monitoring stopped for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {str(e)}")
            return False

    def record_metric(self, metric: MetricData) -> None:
        """
        Record a metric measurement
        记录指标测量

        Args:
            metric: Metric data to record
                   要记录的指标数据
        """
        self.metrics[metric.name].append({
            'value': metric.value,
            'timestamp': metric.timestamp,
            'labels': metric.labels
        })

        logger.debug(f"Recorded metric: {metric.name} = {metric.value}")

    def add_alert_rule(self, alert_rule: AlertRule) -> None:
        """
        Add an alert rule
        添加警报规则

        Args:
            alert_rule: Alert rule to add
                       要添加的警报规则
        """
        self.alert_rules[alert_rule.name] = alert_rule
        logger.info(f"Added alert rule: {alert_rule.name}")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule
        移除警报规则

        Args:
            rule_name: Name of the alert rule to remove
                      要移除的警报规则名称

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def get_metric_stats(self, metric_name: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get statistics for a metric
        获取指标的统计信息

        Args:
            metric_name: Name of the metric
                        指标名称
            time_window: Time window for statistics (None for all data)
                        统计的时间窗口（None表示所有数据）

        Returns:
            dict: Metric statistics
                  指标统计信息
        """
        if metric_name not in self.metrics:
            return {'error': f'Metric {metric_name} not found'}

        data_points = list(self.metrics[metric_name])

        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            data_points = [dp for dp in data_points if dp['timestamp'] >= cutoff_time]

        if not data_points:
            return {'error': 'No data points in specified time window'}

        values = [dp['value'] for dp in data_points]

        try:
            import statistics
            return {
                'metric_name': metric_name,
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'latest_value': values[-1],
                'time_range': {
                    'start': data_points[0]['timestamp'],
                    'end': data_points[-1]['timestamp']
                }
            }

        except Exception as e:
            return {'error': f'Statistics calculation failed: {str(e)}'}

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get currently active alerts
        获取当前活动的警报

        Returns:
            list: List of active alerts
                  活动警报列表
        """
        # Filter alerts from the last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        return [alert for alert in self.active_alerts
                if alert['timestamp'] > cutoff_time]

    def export_metrics(self, format_type: str = 'json', filepath: Optional[str] = None) -> str:
        """
        Export metrics data
        导出指标数据

        Args:
            format_type: Export format ('json' or 'csv')
                        导出格式 ('json' 或 'csv')
            filepath: File path to save data (None for in - memory export)
                     保存数据的文件路径（None表示内存中导出）

        Returns:
            str: Exported data as string
                  作为字符串导出的数据
        """
        try:
            export_data = {}
            for metric_name, data_points in self.metrics.items():
                export_data[metric_name] = [dp for dp in data_points]

            if format_type.lower() == 'json':
                result = json.dumps(export_data, default=str, indent=2)
            elif format_type.lower() == 'csv':
                # Convert to CSV format
                result = self._convert_to_csv(export_data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            if filepath:
                with open(filepath, 'w', encoding='utf - 8') as f:
                    f.write(result)
                logger.info(f"Metrics exported to {filepath}")
                return f"Data exported to {filepath}"

            return result

        except Exception as e:
            error_msg = f"Failed to export metrics: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """
        Convert data to CSV format
        将数据转换为CSV格式

        Args:
            data: Data to convert
                 要转换的数据

        Returns:
            str: CSV formatted data
                  CSV格式的数据
        """
        if not data:
            return ""

        # Use StringIO for CSV generation
        import io
        output = io.StringIO()

        writer = csv.writer(output)

        # Write header
        writer.writerow(['metric_name', 'timestamp', 'value', 'labels'])

        # Write data
        for metric_name, data_points in data.items():
            for point in data_points:
                labels_str = json.dumps(point.get('labels', {}))
                writer.writerow([
                    metric_name,
                    point['timestamp'].isoformat(),
                    point['value'],
                    labels_str
                ])

        return output.getvalue()

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info(f"Monitoring loop started for {self.processor_name}")

        last_alert_check = datetime.now()

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check for alerts periodically
                if (current_time - last_alert_check).seconds >= self.alert_check_interval:
                    self._check_alerts()
                    last_alert_check = current_time

                # Sleep before next cycle
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(self.collection_interval)

        logger.info(f"Monitoring loop stopped for {self.processor_name}")

    def _check_alerts(self) -> None:
        """
        Check all alert rules against current metrics
        根据当前指标检查所有警报规则
        """
        try:
            for rule_name, rule in self.alert_rules.items():
                if rule.metric_name in self.metrics:
                    # Get latest metric value
                    metric_data = self.metrics[rule.metric_name]
                    if metric_data:
                        latest_value = metric_data[-1]['value']

                        # Evaluate rule
                        alert = rule.evaluate(latest_value)
                        if alert:
                            self.active_alerts.append(alert)

                            # Log alert
                            logger.warning(f"Alert triggered: {alert['description']} "
                                           f"(value: {alert['metric_value']})")

            # Clean up old alerts
            if len(self.active_alerts) > self.max_alerts_history:
                self.active_alerts = self.active_alerts[-self.max_alerts_history:]

        except Exception as e:
            logger.error(f"Alert checking error: {str(e)}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get monitoring status and statistics
        获取监控状态和统计信息

        Returns:
            dict: Monitoring status information
                  监控状态信息
        """
        return {
            'processor_name': self.processor_name,
            'is_running': self.is_running,
            'metrics_count': len(self.metrics),
            'alert_rules_count': len(self.alert_rules),
            'active_alerts_count': len(self.get_active_alerts()),
            'total_alerts_history': len(self.active_alerts),
            'collection_interval': self.collection_interval,
            'alert_check_interval': self.alert_check_interval,
            'metrics_sample': {
                name: len(data) for name, data in self.metrics.items()
            }
        }


# Global monitoring processor instance
# 全局监控处理器实例
monitoring_processor = MonitoringProcessor()

__all__ = [
    'MetricType',
    'AlertSeverity',
    'MetricData',
    'AlertRule',
    'MonitoringProcessor',
    'monitoring_processor'
]
