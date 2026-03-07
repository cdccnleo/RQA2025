
from typing import Dict, Any, List, Optional
import logging
import threading
from .dashboard_alerts import AlertManager
from .dashboard_collectors import MetricsCollector
from .dashboard_models import (
    MonitoringConfig, PerformanceMetrics, SystemResources,
    AlertSeverity
)
from datetime import datetime, timedelta

"""
监控面板统一管理器

整合所有监控功能的核心管理器
"""
logger = logging.getLogger(__name__)


class UnifiedMonitoringManager:
    """统一监控管理器"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._lock = threading.RLock()

        # 监控组件
        self._metrics_collector: Optional[MetricsCollector] = None
        self._alert_manager: Optional[AlertManager] = None

        # 数据存储
        self._performance_history: List[PerformanceMetrics] = []
        self._system_resources_history: List[SystemResources] = []
        self._max_history_size = 10000

        # 统计信息
        self._stats = {
            'start_time': datetime.now(),
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'alerts_triggered': 0
        }

    def set_metrics_collector(self, collector: MetricsCollector):
        """设置指标收集器"""
        with self._lock:
            self._metrics_collector = collector

    def set_alert_manager(self, manager: AlertManager):
        """设置告警管理器"""
        with self._lock:
            self._alert_manager = manager

    def start_monitoring(self):
        """启动监控"""
        if not self.config.enabled:
            logger.info("监控已禁用")
            return

        with self._lock:
            if self._metrics_collector:
                self._metrics_collector.start_collection()
                logger.info("指标收集已启动")

            if self._alert_manager and self.config.alerting_enabled:
                logger.info("告警管理已启用")

        logger.info("统一监控管理器已启动")

    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if self._metrics_collector:
                self._metrics_collector.stop_collection()
                logger.info("指标收集已停止")

        logger.info("统一监控管理器已停止")

    def record_operation(self, operation_type: str, duration: float,
                         success: bool, error_message: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """记录操作"""
        with self._lock:
            # 创建性能指标
            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_type=operation_type,
                duration=duration,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )

            # 添加到历史记录
            self._performance_history.append(metric)
            if len(self._performance_history) > self._max_history_size:
                self._performance_history.pop(0)

            # 更新统计信息
            self._stats['total_operations'] += 1
            if success:
                self._stats['successful_operations'] += 1
            else:
                self._stats['failed_operations'] += 1

            # 记录到指标收集器
            if self._metrics_collector:
                self._metrics_collector.record_operation(
                    operation_type, duration, success, metadata
                )

            # 检查是否需要触发告警
            self._check_alerts(metric)

    def record_system_resources(self, cpu_percent: float, memory_percent: float,
                                disk_usage: Dict[str, float], network_io: Dict[str, int],
                                load_average: Optional[tuple] = None):
        """记录系统资源使用情况"""
        with self._lock:
            resources = SystemResources(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                load_average=load_average
            )

            self._system_resources_history.append(resources)
            if len(self._system_resources_history) > self._max_history_size:
                self._system_resources_history.pop(0)

            # 检查资源告警
            self._check_resource_alerts(resources)

    def _check_alerts(self, metric: PerformanceMetrics):
        """检查性能告警"""
        if not self._alert_manager or not self.config.alerting_enabled:
            return

        # 检查操作失败率
        if not metric.success:
            self._alert_manager.create_alert(
                name=f"Operation Failed: {metric.operation_type}",
                description=f"操作 {metric.operation_type} 执行失败",
                severity=AlertSeverity.ERROR,
                labels={'operation': metric.operation_type},
                annotations={'error': metric.error_message or 'Unknown error'},
                value=metric.duration
            )

        # 检查操作超时
        if metric.duration > 30.0:  # 30秒超时阈值
            self._alert_manager.create_alert(
                name=f"Operation Timeout: {metric.operation_type}",
                description=f"操作 {metric.operation_type} 执行超时",
                severity=AlertSeverity.WARNING,
                labels={'operation': metric.operation_type},
                value=metric.duration,
                threshold=30.0
            )

    def _check_resource_alerts(self, resources: SystemResources):
        """检查资源告警"""
        if not self._alert_manager or not self.config.alerting_enabled:
            return

        # CPU 使用率告警
        if resources.cpu_percent > 90:
            self._alert_manager.create_alert(
                name="High CPU Usage",
                description="CPU 使用率过高",
                severity=AlertSeverity.WARNING,
                labels={'resource': 'cpu'},
                value=resources.cpu_percent,
                threshold=90.0
            )

        # 内存使用率告警
        if resources.memory_percent > 85:
            self._alert_manager.create_alert(
                name="High Memory Usage",
                description="内存使用率过高",
                severity=AlertSeverity.WARNING,
                labels={'resource': 'memory'},
                value=resources.memory_percent,
                threshold=85.0
            )

        # 磁盘使用率告警
        for mount_point, usage in resources.disk_usage.items():
            if usage > 95:
                self._alert_manager.create_alert(
                    name=f"High Disk Usage: {mount_point}",
                    description=f"磁盘 {mount_point} 使用率过高",
                    severity=AlertSeverity.ERROR,
                    labels={'resource': 'disk', 'mount': mount_point},
                    value=usage,
                    threshold=95.0
                )

    def get_performance_metrics(self, operation_type: Optional[str] = None,
                                time_range: Optional[timedelta] = None) -> List[PerformanceMetrics]:
        """获取性能指标"""
        with self._lock:
            metrics = self._performance_history

            # 按操作类型过滤
            if operation_type:
                metrics = [m for m in metrics if m.operation_type == operation_type]

            # 按时间范围过滤
            if time_range:
                cutoff_time = datetime.now() - time_range
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            return metrics.copy()

    def get_system_resources(self, time_range: Optional[timedelta] = None) -> List[SystemResources]:
        """获取系统资源历史"""
        with self._lock:
            resources = self._system_resources_history

            if time_range:
                cutoff_time = datetime.now() - time_range
                resources = [r for r in resources if r.timestamp >= cutoff_time]

            return resources.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()

            # 计算运行时间
            stats['uptime'] = (datetime.now() - stats['start_time']).total_seconds()

            # 计算成功率
            if stats['total_operations'] > 0:
                stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            else:
                stats['success_rate'] = 0.0

            # 添加告警统计
            if self._alert_manager:
                alert_summary = self._alert_manager.get_alerts_summary()
                stats.update({
                    'alerts_total': alert_summary.get('total', 0),
                    'alerts_active': alert_summary.get('active', 0),
                    'alerts_resolved': alert_summary.get('resolved', 0)
                })

            return stats

    def cleanup_old_data(self, max_age_days: int = 30):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        with self._lock:
            # 清理性能历史
            original_perf_count = len(self._performance_history)
            self._performance_history = [
                m for m in self._performance_history
                if m.timestamp >= cutoff_time
            ]
            cleaned_count += original_perf_count - len(self._performance_history)

            # 清理系统资源历史
            original_sys_count = len(self._system_resources_history)
            self._system_resources_history = [
                r for r in self._system_resources_history
                if r.timestamp >= cutoff_time
            ]
            cleaned_count += original_sys_count - len(self._system_resources_history)

            # 清理过期告警
            if self._alert_manager:
                alert_cleaned = self._alert_manager.clear_resolved_alerts(max_age_days)
                cleaned_count += alert_cleaned

        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 条过期数据")

        return cleaned_count




