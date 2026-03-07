#!/usr/bin/env python3
"""
监控告警组件

提供数据采集流程的全面监控和智能告警：
1. 实时性能监控
2. 异常检测和告警
3. 趋势分析和预测
4. 多渠道告警通知
5. 监控面板和报告
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import statistics

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    metric: str
    condition: str  # "gt", "lt", "eq", "ne", "range"
    threshold: Union[float, List[float]]
    level: AlertLevel
    enabled: bool = True
    cooldown_minutes: int = 5
    notification_channels: List[str] = field(default_factory=lambda: ["log"])


@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    rule_name: str
    level: AlertLevel
    status: AlertStatus
    title: str
    message: str
    metric: str
    value: Any
    threshold: Any
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricData:
    """指标数据"""
    metric_name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""

    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, List[MetricData]] = {}
        self.retention_hours = retention_hours
        self._lock = threading.RLock()

        # 启动清理任务
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self):
        """启动清理任务"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """停止清理任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"指标清理任务异常: {e}")

    def _cleanup_old_metrics(self):
        """清理旧指标数据"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

        with self._lock:
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    data for data in self.metrics[metric_name]
                    if data.timestamp > cutoff_time
                ]

                # 如果没有数据了，删除这个指标
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]

    def record_metric(self, metric_name: str, value: Union[int, float],
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """记录指标"""
        metric_data = MetricData(
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(metric_data)

            # 限制每个指标的最大数据点数
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name] = self.metrics[metric_name][-1000:]

    def get_metric_data(self, metric_name: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       tags: Optional[Dict[str, str]] = None) -> List[MetricData]:
        """获取指标数据"""
        with self._lock:
            if metric_name not in self.metrics:
                return []

            data = self.metrics[metric_name]

            # 时间过滤
            if start_time:
                data = [d for d in data if d.timestamp >= start_time]
            if end_time:
                data = [d for d in data if d.timestamp <= end_time]

            # 标签过滤
            if tags:
                data = [d for d in data if all(d.tags.get(k) == v for k, v in tags.items())]

            return data

    def get_metric_stats(self, metric_name: str,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取指标统计信息"""
        data = self.get_metric_data(metric_name, start_time, end_time)

        if not data:
            return {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "latest": None,
                "trend": "stable"
            }

        values = [d.value for d in data]
        latest_values = values[-10:] if len(values) >= 10 else values

        # 计算趋势
        if len(latest_values) >= 2:
            if latest_values[-1] > latest_values[0] * 1.1:
                trend = "increasing"
            elif latest_values[-1] < latest_values[0] * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "count": len(values),
            "avg": statistics.mean(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "latest": values[-1] if values else None,
            "trend": trend
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标的统计信息"""
        result = {}
        with self._lock:
            for metric_name in self.metrics.keys():
                result[metric_name] = self.get_metric_stats(metric_name)
        return result

    def collect_test_coverage(self) -> Dict[str, Any]:
        """
        收集测试覆盖率数据（兼容方法，避免AttributeError）
        
        注意：此方法返回空数据，因为monitoring_alerts.py中的MetricsCollector
        主要用于数据采集监控，不负责测试覆盖率收集。
        如果需要测试覆盖率功能，应使用infrastructure/monitoring中的MetricsCollector。
        
        Returns:
            Dict[str, Any]: 测试覆盖率数据（空数据）
        """
        return {
            'timestamp': datetime.now(),
            'success': False,
            'coverage_percent': 0.0,
            'note': 'test_coverage_not_supported_in_data_collection_monitor'
        }


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Callable] = {}

        self._lock = threading.RLock()

        # 默认通知渠道
        self.register_notification_channel("log", self._log_notification)
        self.register_notification_channel("console", self._console_notification)

    def register_rule(self, rule: AlertRule):
        """注册告警规则"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"告警规则已注册: {rule.name} ({rule.rule_id})")

    def unregister_rule(self, rule_id: str):
        """注销告警规则"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"告警规则已注销: {rule_id}")

    def register_notification_channel(self, channel_name: str, handler: Callable):
        """注册通知渠道"""
        self.notification_channels[channel_name] = handler

    def evaluate_rules(self, metrics_collector: MetricsCollector):
        """评估告警规则"""
        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                try:
                    self._evaluate_rule(rule, metrics_collector)
                except Exception as e:
                    logger.error(f"评估告警规则失败 {rule.rule_id}: {e}")

    def _evaluate_rule(self, rule: AlertRule, metrics_collector: MetricsCollector):
        """评估单个告警规则"""
        # 获取指标数据
        metric_stats = metrics_collector.get_metric_stats(rule.metric)

        if not metric_stats or metric_stats["count"] == 0:
            return

        latest_value = metric_stats["latest"]
        if latest_value is None:
            return

        # 检查是否满足告警条件
        alert_triggered = self._check_condition(latest_value, rule.condition, rule.threshold)

        if alert_triggered:
            self._trigger_alert(rule, latest_value, metric_stats)
        else:
            # 检查是否需要解除告警
            self._resolve_alert_if_exists(rule.rule_id)

    def _check_condition(self, value: Union[int, float],
                        condition: str,
                        threshold: Union[float, List[float]]) -> bool:
        """检查是否满足告警条件"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "ne":
            return value != threshold
        elif condition == "range" and isinstance(threshold, list) and len(threshold) == 2:
            return threshold[0] <= value <= threshold[1]
        else:
            logger.warning(f"不支持的告警条件: {condition}")
            return False

    def _trigger_alert(self, rule: AlertRule, current_value: Any, metric_stats: Dict[str, Any]):
        """触发告警"""
        alert_key = f"{rule.rule_id}_{rule.metric}"

        # 检查是否在冷却期内
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            cooldown_end = existing_alert.created_at + timedelta(minutes=rule.cooldown_minutes)

            if datetime.now() < cooldown_end:
                # 在冷却期内，不重复告警
                return

        # 创建新告警
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{rule.rule_id}",
            rule_id=rule.rule_id,
            rule_name=rule.name,
            level=rule.level,
            status=AlertStatus.ACTIVE,
            title=f"{rule.name} - 阈值告警",
            message=self._generate_alert_message(rule, current_value, metric_stats),
            metric=rule.metric,
            value=current_value,
            threshold=rule.threshold,
            source="data_collection_monitor",
            metadata={
                "metric_stats": metric_stats,
                "rule_description": rule.description
            }
        )

        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # 发送通知
        self._send_notifications(alert, rule.notification_channels)

        logger.warning(f"告警触发: {alert.title} - 值: {current_value}, 阈值: {rule.threshold}")

    def _resolve_alert_if_exists(self, rule_id: str):
        """解除告警（如果存在）"""
        alert_key = f"{rule_id}_"

        # 查找相关的活跃告警
        alerts_to_resolve = []
        for key, alert in self.active_alerts.items():
            if key.startswith(alert_key):
                alerts_to_resolve.append((key, alert))

        for key, alert in alerts_to_resolve:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()

            # 发送恢复通知
            self._send_notifications(alert, ["log"], is_resolution=True)

            logger.info(f"告警已恢复: {alert.title}")

    def _generate_alert_message(self, rule: AlertRule, current_value: Any, metric_stats: Dict[str, Any]) -> str:
        """生成告警消息"""
        condition_desc = {
            "gt": "大于",
            "lt": "小于",
            "eq": "等于",
            "ne": "不等于",
            "range": "在范围内"
        }.get(rule.condition, rule.condition)

        return (f"指标 {rule.metric} 当前值 {current_value} {condition_desc} 阈值 {rule.threshold}。"
                f"指标统计: 平均值 {metric_stats.get('avg', 0):.2f}, "
                f"最小值 {metric_stats.get('min', 0)}, "
                f"最大值 {metric_stats.get('max', 0)}")

    def _send_notifications(self, alert: Alert, channels: List[str], is_resolution: bool = False):
        """发送通知"""
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](alert, is_resolution)
                except Exception as e:
                    logger.error(f"发送通知失败 {channel}: {e}")

    def _log_notification(self, alert: Alert, is_resolution: bool = False):
        """日志通知"""
        level_name = "恢复" if is_resolution else alert.level.value.upper()
        logger.log(
            getattr(logging, level_name if level_name in ["INFO", "WARNING", "ERROR", "CRITICAL"] else "WARNING"),
            f"[{level_name}] {alert.title}: {alert.message}"
        )

    def _console_notification(self, alert: Alert, is_resolution: bool = False):
        """控制台通知"""
        status = "恢复" if is_resolution else "触发"
        print(f"[{status}] {alert.level.value.upper()}: {alert.title}")
        print(f"  消息: {alert.message}")
        print(f"  时间: {alert.created_at}")
        print()

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """确认告警"""
        with self._lock:
            for alert in self.active_alerts.values():
                if alert.alert_id == alert_id and alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = acknowledged_by
                    logger.info(f"告警已确认: {alert.title} by {acknowledged_by}")
                    break

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            return [alert for alert in self.alert_history if alert.created_at >= cutoff_time]

    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        with self._lock:
            active_alerts = self.get_active_alerts()
            recent_history = self.get_alert_history(24)

            return {
                "active_alerts": len(active_alerts),
                "alerts_by_level": {
                    level.value: len([a for a in active_alerts if a.level == level])
                    for level in AlertLevel
                },
                "alerts_last_24h": len(recent_history),
                "resolved_alerts": len([a for a in recent_history if a.status == AlertStatus.RESOLVED])
            }

    async def send_alert(self, title: str, message: str, level: AlertLevel, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        发送告警（异步方法，兼容现有代码调用）
        
        Args:
            title: 告警标题
            message: 告警消息
            level: 告警级别（AlertLevel枚举）
            metadata: 可选元数据
        
        Returns:
            str: 告警ID
        """
        try:
            import time
            # 创建告警对象
            alert = Alert(
                alert_id=f"alert_{int(time.time())}_{abs(hash(title))}",
                rule_id="manual_alert",
                rule_name=title,
                level=level,
                status=AlertStatus.ACTIVE,
                title=title,
                message=message,
                metric="manual",
                value=0,
                threshold=0,
                source="data_collection_orchestrator",
                metadata=metadata or {}
            )
            
            # 添加到活跃告警和历史记录
            with self._lock:
                alert_key = f"manual_{alert.alert_id}"
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
            
            # 发送通知（根据级别选择通知渠道）
            channels = ["log", "console"]
            if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                channels.append("console")
            
            self._send_notifications(alert, channels)
            
            logger.warning(f"告警已发送: {title} - {message}")
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
            return ""


class DataCollectionMonitor:
    """数据采集监控器"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()

        # 注册默认告警规则
        self._register_default_alert_rules()

        # 监控状态
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None

        self._lock = threading.RLock()

    def _register_default_alert_rules(self):
        """注册默认告警规则"""

        # 数据采集失败率告警
        self.alert_manager.register_rule(AlertRule(
            rule_id="collection_failure_rate",
            name="数据采集失败率过高",
            description="数据采集失败率超过阈值",
            metric="collection_failure_rate",
            condition="gt",
            threshold=0.1,  # 10%
            level=AlertLevel.WARNING,
            cooldown_minutes=10,
            notification_channels=["log", "console"]
        ))

        # 数据采集延迟告警
        self.alert_manager.register_rule(AlertRule(
            rule_id="collection_latency_high",
            name="数据采集延迟过高",
            description="数据采集平均延迟超过阈值",
            metric="collection_avg_latency",
            condition="gt",
            threshold=5000,  # 5秒
            level=AlertLevel.WARNING,
            cooldown_minutes=5,
            notification_channels=["log", "console"]
        ))

        # 数据质量评分告警
        self.alert_manager.register_rule(AlertRule(
            rule_id="data_quality_low",
            name="数据质量评分过低",
            description="数据质量评分低于阈值",
            metric="data_quality_score",
            condition="lt",
            threshold=0.8,  # 80%
            level=AlertLevel.ERROR,
            cooldown_minutes=15,
            notification_channels=["log", "console"]
        ))

        # 系统资源使用率告警
        self.alert_manager.register_rule(AlertRule(
            rule_id="system_memory_high",
            name="系统内存使用率过高",
            description="系统内存使用率超过阈值",
            metric="system_memory_usage",
            condition="gt",
            threshold=90,  # 90%
            level=AlertLevel.CRITICAL,
            cooldown_minutes=5,
            notification_channels=["log", "console"]
        ))

    async def start_monitoring(self):
        """启动监控"""
        with self._lock:
            if self.monitoring_active:
                return

            self.monitoring_active = True
            await self.metrics_collector.start_cleanup_task()
            self.monitor_task = asyncio.create_task(self._monitoring_loop())

            logger.info("数据采集监控已启动")

    async def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if not self.monitoring_active:
                return

            self.monitoring_active = False

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            await self.metrics_collector.stop_cleanup_task()

            logger.info("数据采集监控已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                await self._collect_system_metrics()

                # 评估告警规则
                self.alert_manager.evaluate_rules(self.metrics_collector)

                # 等待下一次检查
                await asyncio.sleep(30)  # 每30秒检查一次

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(30)

    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil

            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(
                "system_cpu_usage",
                cpu_percent,
                tags={"type": "system"}
            )

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics_collector.record_metric(
                "system_memory_usage",
                memory_percent,
                tags={"type": "system"}
            )

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.metrics_collector.record_metric(
                "system_disk_usage",
                disk_percent,
                tags={"type": "system"}
            )

        except ImportError:
            # 如果没有psutil，使用模拟数据
            import random
            self.metrics_collector.record_metric("system_cpu_usage", random.uniform(10, 80))
            self.metrics_collector.record_metric("system_memory_usage", random.uniform(20, 85))
            self.metrics_collector.record_metric("system_disk_usage", random.uniform(30, 90))

    def record_collection_metric(self, source_id: str, metric_name: str,
                                value: Union[int, float],
                                metadata: Optional[Dict[str, Any]] = None):
        """记录数据采集指标"""
        tags = {"source_id": source_id, "type": "collection"}
        self.metrics_collector.record_metric(
            f"collection_{metric_name}",
            value,
            tags=tags,
            metadata=metadata
        )

    def record_workflow_metric(self, workflow_id: str, metric_name: str,
                              value: Union[int, float],
                              metadata: Optional[Dict[str, Any]] = None):
        """记录工作流程指标"""
        tags = {"workflow_id": workflow_id, "type": "workflow"}
        self.metrics_collector.record_metric(
            f"workflow_{metric_name}",
            value,
            tags=tags,
            metadata=metadata
        )

    def record_metric(self, name: str, value: Union[int, float], 
                      labels: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录指标（统一接口，兼容现有代码）
        
        Args:
            name: 指标名称
            value: 指标值
            labels: 指标标签（可选，兼容代码中使用的labels参数）
        
        Returns:
            bool: 是否记录成功
        """
        try:
            # 委托给metrics_collector，使用metric_name参数（匹配MetricsCollector.record_metric的参数名）
            return self.metrics_collector.record_metric(
                metric_name=name,  # MetricsCollector.record_metric的参数是metric_name，不是name
                value=value,
                tags=labels or {}
            )
        except Exception as e:
            logger.warning(f"记录指标失败: {name}, 错误: {e}")
            return False

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        with self._lock:
            return {
                "monitoring_active": self.monitoring_active,
                "metrics": self.metrics_collector.get_all_metrics(),
                "alerts": self.alert_manager.get_alert_stats(),
                "active_alerts": [
                    {
                        "id": alert.alert_id,
                        "title": alert.title,
                        "level": alert.level.value,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ]
            }
