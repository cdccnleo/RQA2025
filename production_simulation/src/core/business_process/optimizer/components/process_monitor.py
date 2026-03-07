"""
流程监控器组件

职责:
- 监控流程执行状态和性能
- 收集和分析流程指标
- 触发告警和通知
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Protocol
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from src.core.constants import (
    DEFAULT_BATCH_SIZE, SECONDS_PER_HOUR
)

logger = logging.getLogger(__name__)


# 先定义数据类（在Protocol之前）
class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ProcessMetrics:
    """流程指标"""
    process_id: str
    timestamp: datetime
    stage: str
    execution_time: float
    success_rate: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """告警"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metrics: Optional['ProcessMetrics'] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


# 流程监控相关协议（在数据类之后）
class MetricsCollector(Protocol):
    """指标收集器协议"""
    async def collect_metrics(self, process_id: str) -> ProcessMetrics: ...


class AlertChecker(Protocol):
    """告警检查器协议"""
    async def check_alerts(self, metrics: ProcessMetrics) -> List[Alert]: ...


class AlertTrigger(Protocol):
    """告警触发器协议"""
    async def trigger_alerts(self, alerts: List[Alert]): ...


class ProcessTracker(Protocol):
    """流程跟踪器协议"""
    def register_monitor(self, process_id: str, context: Any): ...
    def unregister_monitor(self, process_id: str): ...
    def get_active_monitors(self) -> Dict[str, Dict[str, Any]]: ...


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ProcessMetrics:
    """流程指标"""
    process_id: str
    timestamp: datetime
    stage: str
    execution_time: float
    success_rate: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """告警"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metrics: Optional[ProcessMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


# 专门组件实现
class MetricsCollectorImpl:
    """指标收集器实现 - 职责：收集流程指标"""

    def __init__(self, config: 'MonitoringConfig', metrics_history: deque, active_monitors: Dict[str, Dict[str, Any]]):
        self.config = config
        self._metrics_history = metrics_history
        self._active_monitors = active_monitors

    async def collect_metrics(self, process_id: str) -> ProcessMetrics:
        """收集流程指标"""
        # 获取监控数据
        monitor_data = self._active_monitors.get(process_id, {})

        # 计算执行时间
        if 'start_time' in monitor_data:
            execution_time = (datetime.now() - monitor_data['start_time']).total_seconds()
        else:
            execution_time = 0.0

        # 创建指标对象
        metrics = ProcessMetrics(
            process_id=process_id,
            timestamp=datetime.now(),
            stage='monitoring',
            execution_time=execution_time,
            success_rate=self._calculate_success_rate(),
            resource_usage=self._collect_resource_usage(),
            performance_score=self._calculate_performance_score(execution_time)
        )

        # 保存到历史
        self._metrics_history.append(metrics)

        return metrics

    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if not self._metrics_history:
            return 1.0
        recent_metrics = list(self._metrics_history)[-20:]
        success_count = sum(1 for m in recent_metrics if m.performance_score > 0.5)
        return success_count / len(recent_metrics) if recent_metrics else 1.0

    def _collect_resource_usage(self) -> Dict[str, Any]:
        """收集资源使用情况"""
        return {
            'cpu_usage': 0.3,
            'memory_usage': 0.5,
            'disk_io': 0.2
        }

    def _calculate_performance_score(self, execution_time: float) -> float:
        """计算性能评分"""
        if execution_time < 1.0:
            return 1.0
        elif execution_time < 5.0:
            return 0.8
        elif execution_time < DEFAULT_BATCH_SIZE:
            return 0.6
        else:
            return 0.4


class AlertCheckerImpl:
    """告警检查器实现 - 职责：检查告警条件"""

    def __init__(self, config: 'MonitoringConfig', alerts: List[Alert]):
        self.config = config
        self._alerts = alerts

    async def check_alerts(self, metrics: ProcessMetrics) -> List[Alert]:
        """检查告警条件"""
        if not self.config.enable_auto_alert:
            return []

        alerts = []

        # 检查执行时间告警
        time_threshold = self.config.alert_threshold.get('execution_time', float('inf'))
        if metrics.execution_time > time_threshold:
            alerts.append(Alert(
                alert_id=f"alert_{len(self._alerts) + 1}",
                level=AlertLevel.WARNING,
                title="执行时间超限",
                message=f"流程{metrics.process_id}执行时间{metrics.execution_time:.2f}秒，超过阈值{time_threshold}秒",
                metrics=metrics
            ))

        # 检查成功率告警
        success_threshold = self.config.alert_threshold.get('success_rate', 0.0)
        if metrics.success_rate < success_threshold:
            alerts.append(Alert(
                alert_id=f"alert_{len(self._alerts) + 2}",
                level=AlertLevel.ERROR,
                title="成功率过低",
                message=f"流程{metrics.process_id}成功率{metrics.success_rate:.2%}，低于阈值{success_threshold:.2%}",
                metrics=metrics
            ))

        return alerts


class AlertTriggerImpl:
    """告警触发器实现 - 职责：触发告警"""

    def __init__(self, config: 'MonitoringConfig', alerts: List[Alert], alert_handlers: List[Callable]):
        self.config = config
        self._alerts = alerts
        self._alert_handlers = alert_handlers

    async def trigger_alerts(self, alerts: List[Alert]):
        """触发告警"""
        self._alerts.extend(alerts)

        for alert in alerts:
            logger.warning(f"告警触发 [{alert.level.value}] {alert.title}: {alert.message}")

            # 调用注册的处理器
            for handler in self._alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"告警处理器执行失败: {e}")


class ProcessTrackerImpl:
    """流程跟踪器实现 - 职责：跟踪流程状态"""

    def __init__(self, config: 'MonitoringConfig'):
        self.config = config
        self._active_monitors: Dict[str, Dict[str, Any]] = {}

    def register_monitor(self, process_id: str, context: Any):
        """注册监控"""
        self._active_monitors[process_id] = {
            'context': context,
            'start_time': datetime.now(),
            'last_check': datetime.now(),
            'metrics': None
        }

    def unregister_monitor(self, process_id: str):
        """注销监控"""
        if process_id in self._active_monitors:
            del self._active_monitors[process_id]

    def get_active_monitors(self) -> Dict[str, Dict[str, Any]]:
        """获取活跃监控列表"""
        return self._active_monitors.copy()


class ProcessMonitor:
    """
    流程监控器组件 - 重构版：组合模式

    实时监控流程执行，收集性能指标
    支持告警通知和异常检测
    """

    def __init__(self, config: 'MonitoringConfig'):
        """
        初始化流程监控器

        Args:
            config: 监控配置对象
        """
        self.config = config
        self._metrics_history: deque = deque(maxlen=config.metrics_retention)
        self._alert_handlers: List[Callable] = []
        self._alerts: List[Alert] = []
        self._monitoring_active = False

        # 初始化专门组件
        self._tracker = ProcessTrackerImpl(config)
        self._metrics_collector = MetricsCollectorImpl(config, self._metrics_history, self._tracker._active_monitors)
        self._alert_checker = AlertCheckerImpl(config, self._alerts)
        self._alert_trigger = AlertTriggerImpl(config, self._alerts, self._alert_handlers)

        logger.info(f"重构后的流程监控器初始化完成 (间隔: {config.monitoring_interval}秒)")

    # 代理方法到专门的组件
    async def monitor_process(self, process_id: str, context: Any) -> ProcessMetrics:
        """
        监控流程执行 - 代理到专门组件

        Args:
            process_id: 流程ID
            context: 流程上下文

        Returns:
            ProcessMetrics: 流程指标
        """
        # 注册监控
        self._tracker.register_monitor(process_id, context)

        # 收集指标
        metrics = await self._metrics_collector.collect_metrics(process_id)

        # 更新监控数据
        if process_id in self._tracker._active_monitors:
            self._tracker._active_monitors[process_id]['metrics'] = metrics
            self._tracker._active_monitors[process_id]['last_check'] = datetime.now()

        # 检查告警
        alerts = await self._alert_checker.check_alerts(metrics)
        if alerts:
            await self._alert_trigger.trigger_alerts(alerts)

        logger.debug(f"流程监控更新: {process_id}, 性能评分: {metrics.performance_score:.3f}")
        return metrics

    async def collect_metrics(self, process_id: str) -> ProcessMetrics:
        """收集流程指标 - 代理到指标收集器"""
        return await self._metrics_collector.collect_metrics(process_id)

    def register_alert_handler(self, handler: Callable):
        """注册告警处理器"""
        self._alert_handlers.append(handler)
        logger.info(f"告警处理器已注册，当前数量: {len(self._alert_handlers)}")

    async def check_alerts(self, metrics: ProcessMetrics) -> List[Alert]:
        """检查告警条件 - 代理到告警检查器"""
        return await self._alert_checker.check_alerts(metrics)

    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        # 统计数据
        total_processes = len(self._metrics_history)
        active_monitors = self._tracker.get_active_monitors()

        if total_processes > 0:
            avg_execution_time = sum(m.execution_time for m in self._metrics_history) / total_processes
            avg_success_rate = sum(m.success_rate for m in self._metrics_history) / total_processes
            avg_performance = sum(m.performance_score for m in self._metrics_history) / total_processes
        else:
            avg_execution_time = 0.0
            avg_success_rate = 0.0
            avg_performance = 0.0

        return {
            'summary': {
                'active_monitors': len(active_monitors),
                'total_processes_monitored': total_processes,
                'total_alerts': len(self._alerts),
                'unacknowledged_alerts': sum(1 for a in self._alerts if not a.acknowledged)
            },
            'averages': {
                'execution_time': avg_execution_time,
                'success_rate': avg_success_rate,
                'performance_score': avg_performance
            },
            'recent_alerts': self._get_recent_alerts(5),
            'monitoring_config': {
                'interval': self.config.monitoring_interval,
                'auto_alert': self.config.enable_auto_alert,
                'anomaly_detection': self.config.enable_anomaly_detection
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        active_monitors = self._tracker.get_active_monitors()
        return {
            'monitoring_active': self._monitoring_active,
            'active_monitors': len(active_monitors),
            'metrics_history_size': len(self._metrics_history),
            'alerts_count': len(self._alerts),
            'handlers_count': len(self._alert_handlers)
        }

    async def start_monitoring(self):
        """启动监控服务"""
        self._monitoring_active = True
        logger.info("流程监控服务已启动")

        # 启动后台监控任务
        asyncio.create_task(self._background_monitoring())

    async def stop_monitoring(self):
        """停止监控服务"""
        self._monitoring_active = False
        logger.info("流程监控服务已停止")

    # 保持向后兼容性
    async def _trigger_alerts(self, alerts: List[Alert]):
        """触发告警（向后兼容）"""
        return await self._alert_trigger.trigger_alerts(alerts)

    def _get_recent_alerts(self, limit: int) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        recent = self._alerts[-limit:] if self._alerts else []
        return [
            {
                'alert_id': alert.alert_id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged
            }
            for alert in recent
        ]

    async def _background_monitoring(self):
        """后台监控任务"""
        while self._monitoring_active:
            try:
                # 检查所有活跃监控
                active_monitors = self._tracker.get_active_monitors()
                for process_id, monitor_data in list(active_monitors.items()):
                    # 更新监控数据
                    elapsed = (datetime.now() - monitor_data['start_time']).total_seconds()

                    # 如果超过一定时间没有更新，清理
                    if elapsed > SECONDS_PER_HOUR:  # 1小时
                        self._tracker.unregister_monitor(process_id)
                        logger.info(f"清理过期监控: {process_id}")

                # 等待下一个监控周期
                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                logger.error(f"后台监控任务异常: {e}")
                await asyncio.sleep(self.config.monitoring_interval)


# 配置类会通过参数传入，无需导入
