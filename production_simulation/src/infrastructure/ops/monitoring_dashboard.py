"""
monitoring_dashboard 模块

提供 monitoring_dashboard 相关功能和接口。
"""

import json
import logging

# 更新系统指标
import psutil
import secrets
import threading
import time

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 监控仪表板
提供系统监控和可视化功能
"""

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """告警严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    """指标数据类"""
    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    description: str = ""
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'metric_type': self.metric_type.value,
            'description': self.description,
            'unit': self.unit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """从字典创建"""
        return cls(
            name=data['name'],
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            labels=data.get('labels', {}),
            metric_type=MetricType(data.get('metric_type', 'gauge')),
            description=data.get('description', ''),
            unit=data.get('unit', '')
        )


@dataclass
class Alert:
    """告警数据类"""
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'labels': self.labels,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """从字典创建"""
        return cls(
            title=data['title'],
            message=data['message'],
            severity=AlertSeverity(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data.get('source', ''),
            labels=data.get('labels', {}),
            resolved=data.get('resolved', False),
            resolved_at=datetime.fromisoformat(
                data['resolved_at']) if data.get('resolved_at') else None
        )

    def resolve(self):
        """标记告警为已解决"""
        self.resolved = True
        self.resolved_at = datetime.now()


@dataclass
class DashboardConfig:
    """仪表板配置类"""
    title: str = "RQA2025 监控仪表板"
    refresh_interval: int = 30  # 刷新间隔(秒)
    theme: str = "dark"
    layout: Dict[str, Any] = field(default_factory=dict)
    widgets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'title': self.title,
            'refresh_interval': self.refresh_interval,
            'theme': self.theme,
            'layout': self.layout,
            'widgets': self.widgets
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardConfig':
        """从字典创建"""
        return cls(
            title=data.get('title', 'RQA2025 监控仪表板'),
            refresh_interval=data.get('refresh_interval', 30),
            theme=data.get('theme', 'dark'),
            layout=data.get('layout', {}),
            widgets=data.get('widgets', [])
        )


class MonitoringDashboard:
    """监控仪表板类"""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or DashboardConfig()
        self.metrics: Dict[str, Metric] = {}
        self.alerts: List[Alert] = []
        self._running = False
        self._refresh_thread = None

        # 默认指标
        self._initialize_default_metrics()

    def _initialize_default_metrics(self):
        """初始化默认指标"""
        default_metrics = [
            Metric(
                name="system.cpu.usage",
                value=0.0,
                description="CPU使用率",
                unit="percent",
                metric_type=MetricType.GAUGE
            ),
            Metric(
                name="system.memory.usage",
                value=0.0,
                description="内存使用率",
                unit="percent",
                metric_type=MetricType.GAUGE
            ),
            Metric(
                name="system.disk.usage",
                value=0.0,
                description="磁盘使用率",
                unit="percent",
                metric_type=MetricType.GAUGE
            ),
            Metric(
                name="trading.requests.total",
                value=0,
                description="总交易请求数",
                unit="count",
                metric_type=MetricType.COUNTER
            ),
            Metric(
                name="trading.requests.success",
                value=0,
                description="成功交易请求数",
                unit="count",
                metric_type=MetricType.COUNTER
            ),
            Metric(
                name="trading.requests.error",
                value=0,
                description="错误交易请求数",
                unit="count",
                metric_type=MetricType.COUNTER
            )
        ]

        for metric in default_metrics:
            self.metrics[metric.name] = metric

    def add_metric(self, metric: Metric):
        """添加指标"""
        self.metrics[metric.name] = metric
        self.logger.debug(f"添加指标: {metric.name} = {metric.value}")

    def update_metric(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """更新指标值"""
        if name in self.metrics:
            self.metrics[name].value = value
            if labels:
                self.metrics[name].labels.update(labels)
            self.metrics[name].timestamp = datetime.now()
            self.logger.debug(f"更新指标: {name} = {value}")
        else:
            # 如果指标不存在，创建新指标
            metric = Metric(name=name, value=value, labels=labels or {})
            self.add_metric(metric)

    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        return self.metrics.get(name)

    def get_all_metrics(self) -> List[Metric]:
        """获取所有指标"""
        return list(self.metrics.values())

    def add_alert(self, alert: Alert):
        """添加告警"""
        self.alerts.append(alert)
        self.logger.warning(f"添加告警: {alert.title} - {alert.message}")

    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                     source: str = "", labels: Optional[Dict[str, str]] = None):
        """创建告警"""
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            labels=labels or {}
        )
        self.add_alert(alert)

    def get_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """获取告警"""
        if resolved is None:
            return self.alerts
        return [alert for alert in self.alerts if alert.resolved == resolved]

    def resolve_alert(self, alert_index: int):
        """解决告警"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolve()
            self.logger.info(f"解决告警: {self.alerts[alert_index].title}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return {
            'config': self.config.to_dict(),
            'metrics': [metric.to_dict() for metric in self.get_all_metrics()],
            'alerts': [alert.to_dict() for alert in self.get_alerts()],
            'summary': self._get_summary_stats()
        }

    def _get_summary_stats(self) -> Dict[str, Any]:
        """获取摘要统计"""
        total_alerts = len(self.alerts)
        active_alerts = len([a for a in self.alerts if not a.resolved])
        critical_alerts = len([a for a in self.alerts if a.severity ==
                              AlertSeverity.CRITICAL and not a.resolved])

        return {
            'total_metrics': len(self.metrics),
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'critical_alerts': critical_alerts,
            'last_update': datetime.now().isoformat()
        }

    def export_dashboard(self, filepath: str):
        """导出仪表板配置"""
        data = self.get_dashboard_data()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"仪表板配置已导出到: {filepath}")

    def import_dashboard(self, filepath: str):
        """导入仪表板配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 恢复配置
        if 'config' in data:
            self.config = DashboardConfig.from_dict(data['config'])

        # 恢复指标
        if 'metrics' in data:
            for metric_data in data['metrics']:
                metric = Metric.from_dict(metric_data)
                self.metrics[metric.name] = metric

        # 恢复告警
        if 'alerts' in data:
            for alert_data in data['alerts']:
                alert = Alert.from_dict(alert_data)
                self.alerts.append(alert)

        self.logger.info(f"仪表板配置已从 {filepath} 导入")

    def start_auto_refresh(self):
        """启动自动刷新"""
        if self._running:
            return

        self._running = True
        self._refresh_thread = threading.Thread(target=self._auto_refresh_loop, daemon=True)
        self._refresh_thread.start()
        self.logger.info("启动自动刷新")

    def stop_auto_refresh(self):
        """停止自动刷新"""
        self._running = False
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5.0)
        self.logger.info("停止自动刷新")

    def _auto_refresh_loop(self):
        """自动刷新循环"""
        while self._running:
            try:
                # 模拟更新一些指标
                self.update_metric("system.cpu.usage", psutil.cpu_percent())
                memory = psutil.virtual_memory()
                self.update_metric("system.memory.usage", memory.percent)
                disk = psutil.disk_usage('/')
                self.update_metric("system.disk.usage", disk.percent)

                # 随机更新交易指标
                self.update_metric("trading.requests.total",
                                   self.get_metric("trading.requests.total").value + secrets.randint(0, 5))
                self.update_metric("trading.requests.success",
                                   self.get_metric("trading.requests.success").value + secrets.randint(0, 4))
                self.update_metric("trading.requests.error",
                                   self.get_metric("trading.requests.error").value + secrets.randint(0, 1))

                # 检查是否需要创建告警
                if memory.percent > 90:
                    self.create_alert(
                        "高内存使用率",
                        f"内存使用率过高: {memory.percent}%",
                        AlertSeverity.HIGH,
                        "system_monitor"
                    )

                time.sleep(self.config.refresh_interval)

            except Exception as e:
                self.logger.error(f"自动刷新出错: {e}")
                time.sleep(5)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        metrics = self.get_all_metrics()
        alerts = self.get_alerts(resolved=False)

        # 计算健康分数 (0-100)
        health_score = 100

        # 检查系统指标
        cpu_usage = next((m.value for m in metrics if m.name == "system.cpu.usage"), 0)
        memory_usage = next((m.value for m in metrics if m.name == "system.memory.usage"), 0)
        disk_usage = next((m.value for m in metrics if m.name == "system.disk.usage"), 0)

        # 根据使用率降低健康分数
        if cpu_usage > 90:
            health_score -= 20
        elif cpu_usage > 70:
            health_score -= 10

        if memory_usage > 90:
            health_score -= 25
        elif memory_usage > 80:
            health_score -= 15

        if disk_usage > 90:
            health_score -= 20
        elif disk_usage > 80:
            health_score -= 10

        # 根据告警数量降低健康分数
        critical_count = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
        high_count = len([a for a in alerts if a.severity == AlertSeverity.HIGH])

        health_score -= critical_count * 10
        health_score -= high_count * 5

        health_score = max(0, min(100, health_score))

        return {
            'health_score': health_score,
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'active_alerts': len(alerts),
            'critical_alerts': critical_count,
            'last_check': datetime.now().isoformat()
        }
