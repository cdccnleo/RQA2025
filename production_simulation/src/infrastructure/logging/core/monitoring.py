
import threading
import time

from .interfaces import LogLevel
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
"""
基础设施层 - 日志系统运行时监控

提供日志系统的性能监控、统计信息和健康检查功能。
"""


@dataclass
class LogSystemMetrics:
    """日志系统指标"""
    total_logs_processed: int = 0
    logs_per_second: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    average_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    active_loggers: int = 0
    queued_logs: int = 0
    last_health_check: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """
    指标收集器 - 负责日志系统指标的收集和存储

    单一职责：收集、更新和提供系统运行指标
    """

    def __init__(self):
        self._metrics = LogSystemMetrics()
        self._start_time = time.time()
        self._lock = threading.Lock()

    def record_log_processed(self, level: LogLevel, processing_time: float = 0.0):
        """
        记录日志处理事件

        Args:
            level: 日志级别
            processing_time: 处理时间（秒）
        """
        with self._lock:
            self._metrics.total_logs_processed += 1

            if level == LogLevel.ERROR:
                self._metrics.error_count += 1
            elif level == LogLevel.WARNING:
                self._metrics.warning_count += 1

            if processing_time > 0:
                # 计算移动平均处理时间
                current_avg = self._metrics.average_processing_time
                self._metrics.average_processing_time = (current_avg + processing_time) / 2

    def update_active_loggers(self, count: int):
        """更新活跃日志器数量"""
        with self._lock:
            self._metrics.active_loggers = count

    def update_queue_size(self, size: int):
        """更新队列大小"""
        with self._lock:
            self._metrics.queued_logs = size

    def get_metrics(self) -> LogSystemMetrics:
        """获取当前指标"""
        with self._lock:
            # 更新实时指标
            self._metrics.uptime_seconds = time.time() - self._start_time
            self._metrics.logs_per_second = (
                self._metrics.total_logs_processed / self._metrics.uptime_seconds
                if self._metrics.uptime_seconds > 0 else 0
            )
            return self._metrics

    def get_raw_metrics(self) -> LogSystemMetrics:
        """获取原始指标数据"""
        return self._metrics


class HealthChecker:
    """
    健康检查器 - 负责系统健康状态评估

    单一职责：执行各种健康检查并聚合结果
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

    def check_health(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            健康状态字典
        """
        metrics = self.metrics_collector.get_metrics()

        # 执行各项健康检查
        health_checks = [
            self._check_error_rate(metrics),
            self._check_performance(metrics),
            self._check_queue_backlog(metrics),
            self._check_memory_usage(metrics)
        ]

        # 聚合健康状态
        overall_status, all_issues = self._aggregate_health_status(health_checks)

        # 构建响应
        return self._build_health_response(overall_status, all_issues, metrics)

    def _check_error_rate(self, metrics) -> tuple[str, list[str]]:
        """检查错误率"""
        total_logs = metrics.total_logs_processed
        if total_logs == 0:
            return "healthy", []

        error_rate = (metrics.error_count + metrics.warning_count) / total_logs

        if error_rate > 0.1:  # 10%错误率
            return "critical", [f"错误率过高: {error_rate:.1%}"]
        elif error_rate > 0.05:  # 5%错误率
            return "warning", [f"错误率较高: {error_rate:.1%}"]
        else:
            return "healthy", []

    def _check_performance(self, metrics) -> tuple[str, list[str]]:
        """检查性能"""
        avg_time = metrics.average_processing_time
        if avg_time > 1.0:  # 平均处理时间超过1秒
            return "critical", [f"平均处理时间过长: {avg_time:.3f}s"]
        elif avg_time > 0.5:  # 平均处理时间超过0.5秒
            return "warning", [f"平均处理时间较长: {avg_time:.3f}s"]
        else:
            return "healthy", []

    def _check_queue_backlog(self, metrics) -> tuple[str, list[str]]:
        """检查队列积压"""
        queue_size = metrics.queued_logs
        if queue_size > 1000:  # 队列积压超过1000
            return "critical", [f"队列积压严重: {queue_size}个日志"]
        elif queue_size > 500:  # 队列积压超过500
            return "warning", [f"队列积压较大: {queue_size}个日志"]
        else:
            return "healthy", []

    def _check_memory_usage(self, metrics) -> tuple[str, list[str]]:
        """检查内存使用"""
        memory_mb = metrics.memory_usage_mb
        if memory_mb > 500:  # 内存使用超过500MB
            return "critical", [f"内存使用过高: {memory_mb:.1f}MB"]
        elif memory_mb > 200:  # 内存使用超过200MB
            return "warning", [f"内存使用较高: {memory_mb:.1f}MB"]
        else:
            return "healthy", []

    def _aggregate_health_status(self, health_checks: list[tuple[str, list[str]]]) -> tuple[str, list[str]]:
        """聚合健康状态"""
        status_priority = {"healthy": 0, "warning": 1, "critical": 2}
        overall_status = "healthy"
        all_issues = []

        for status, issues in health_checks:
            if status_priority[status] > status_priority[overall_status]:
                overall_status = status
            all_issues.extend(issues)

        return overall_status, all_issues

    def _build_health_response(self, status: str, issues: list[str], metrics) -> Dict[str, Any]:
        """构建健康检查响应"""
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "metrics": {
                "total_logs_processed": metrics.total_logs_processed,
                "logs_per_second": metrics.logs_per_second,
                "error_count": metrics.error_count,
                "warning_count": metrics.warning_count,
                "average_processing_time": metrics.average_processing_time,
                "memory_usage_mb": metrics.memory_usage_mb,
                "uptime_seconds": metrics.uptime_seconds,
                "active_loggers": metrics.active_loggers,
                "queued_logs": metrics.queued_logs
            }
        }


class AlertManager:
    """
    告警管理器 - 负责告警检测和触发

    单一职责：监控指标，检测告警条件，触发告警回调
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._alert_callbacks: List[callable] = []
        self._lock = threading.Lock()

    def add_alert_callback(self, callback: callable):
        """
        添加告警回调函数

        Args:
            callback: 回调函数，接收告警信息字典
        """
        self._alert_callbacks.append(callback)

    def check_alerts(self):
        """检查告警条件"""
        metrics = self.metrics_collector.get_metrics()
        alerts = self._collect_all_alerts(metrics)

        if alerts:
            self._trigger_alert_callbacks(alerts)

    def _collect_all_alerts(self, metrics) -> list:
        """收集所有告警"""
        alerts = []

        # 检查各种告警条件
        alerts.extend(self._check_error_rate_alerts(metrics))
        alerts.extend(self._check_performance_alerts(metrics))
        alerts.extend(self._check_queue_backlog_alerts(metrics))

        return alerts

    def _check_error_rate_alerts(self, metrics) -> list:
        """检查错误率告警"""
        alerts = []
        total_logs = metrics.total_logs_processed

        if total_logs > 0:
            error_rate = (metrics.error_count + metrics.warning_count) / total_logs

            if error_rate > 0.15:  # 15%错误率触发告警
                alerts.append({
                    "type": "error_rate",
                    "level": "critical",
                    "message": f"错误率严重超标: {error_rate:.1%}",
                    "value": error_rate,
                    "threshold": 0.15
                })
            elif error_rate > 0.08:  # 8%错误率触发警告
                alerts.append({
                    "type": "error_rate",
                    "level": "warning",
                    "message": f"错误率偏高: {error_rate:.1%}",
                    "value": error_rate,
                    "threshold": 0.08
                })

        return alerts

    def _check_performance_alerts(self, metrics) -> list:
        """检查性能告警"""
        alerts = []
        avg_time = metrics.average_processing_time

        if avg_time > 2.0:  # 处理时间超过2秒
            alerts.append({
                "type": "performance",
                "level": "critical",
                "message": f"处理性能严重下降: {avg_time:.3f}s",
                "value": avg_time,
                "threshold": 2.0
            })
        elif avg_time > 1.0:  # 处理时间超过1秒
            alerts.append({
                "type": "performance",
                "level": "warning",
                "message": f"处理性能下降: {avg_time:.3f}s",
                "value": avg_time,
                "threshold": 1.0
            })

        return alerts

    def _check_queue_backlog_alerts(self, metrics) -> list:
        """检查队列积压告警"""
        alerts = []
        queue_size = metrics.queued_logs

        if queue_size > 2000:  # 队列积压超过2000
            alerts.append({
                "type": "queue_backlog",
                "level": "critical",
                "message": f"队列积压严重: {queue_size}个日志",
                "value": queue_size,
                "threshold": 2000
            })
        elif queue_size > 1000:  # 队列积压超过1000
            alerts.append({
                "type": "queue_backlog",
                "level": "warning",
                "message": f"队列积压较大: {queue_size}个日志",
                "value": queue_size,
                "threshold": 1000
            })

        return alerts

    def _trigger_alert_callbacks(self, alerts: list) -> None:
        """触发告警回调"""
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    # 避免回调失败影响其他回调
                    print(f"告警回调失败: {e}")


class LogSystemMonitor:
    """
    日志系统监控器 - 门面类

    协调各个监控组件，提供统一的监控接口
    遵循门面模式和组合优于继承原则
    """

    def __init__(self):
        # 组合各个组件
        self._metrics_collector = MetricsCollector()
        self._health_checker = HealthChecker(self._metrics_collector)
        self._alert_manager = AlertManager(self._metrics_collector)

        # 启动监控线程
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

    # 门面方法 - 委托给各个组件

    def record_log_processed(self, level: LogLevel, processing_time: float = 0.0):
        """
        记录日志处理事件

        Args:
            level: 日志级别
            processing_time: 处理时间（秒）
        """
        self._metrics_collector.record_log_processed(level, processing_time)
        # 检查告警
        self._alert_manager.check_alerts()

    def update_active_loggers(self, count: int):
        """更新活跃日志器数量"""
        self._metrics_collector.update_active_loggers(count)

    def update_queue_size(self, size: int):
        """更新队列大小"""
        self._metrics_collector.update_queue_size(size)

    def add_alert_callback(self, callback: callable):
        """
        添加告警回调函数

        Args:
            callback: 回调函数，接收告警信息字典
        """
        self._alert_manager.add_alert_callback(callback)

    def get_metrics(self) -> LogSystemMetrics:
        """获取当前指标"""
        return self._metrics_collector.get_metrics()

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            健康状态字典
        """
        return self._health_checker.check_health()

    def _monitoring_loop(self):
        """监控循环"""
        while self._monitoring_active:
            try:
                # 执行定期健康检查
                health_status = self.get_health_status()

                # 检查告警
                self._alert_manager.check_alerts()

                # 等待下一个检查周期 (60秒)
                time.sleep(60)

            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(60)  # 出错时等待60秒后重试

    def shutdown(self):
        """关闭监控器"""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)


# 全局监控器实例
_global_monitor: Optional[LogSystemMonitor] = None
_monitor_lock = threading.Lock()


def get_log_monitor() -> LogSystemMonitor:
    """
    获取全局日志监控器实例

    Returns:
        LogSystemMonitor: 监控器实例
    """
    global _global_monitor

    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = LogSystemMonitor()
        return _global_monitor


def record_log_event(level: LogLevel, processing_time: float = 0.0):
    """
    记录日志事件（便捷函数）

    Args:
        level: 日志级别
        processing_time: 处理时间
    """
    monitor = get_log_monitor()
    monitor.record_log_processed(level, processing_time)


class LoggingMonitor(LogSystemMonitor):
    """向后兼容的 LoggingMonitor 别名"""
    pass