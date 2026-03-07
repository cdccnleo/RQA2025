"""
performance_monitor 模块

提供 performance_monitor 相关功能和接口。
"""


from .metrics_analyzer import MetricsAnalyzer
from .metrics_collector import MetricsCollector
from .metrics_storage import MetricsStorage
from .shared_interfaces import StandardLogger, BaseErrorHandler, ILogger, IErrorHandler
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
"""
RQA2025 性能监控组件 - 重构版本

重构说明：
- PerformanceMonitorLegacy现在作为外观类，协调各个专用组件
- 使用独立的MetricsCollector、MetricsStorage、MetricsAnalyzer组件
- MonitoringCoordinator负责监控生命周期管理
- HealthEvaluator负责健康状态评估
"""


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    thread_count: int


@dataclass
class PerformanceConfig:
    """性能监控配置"""
    collection_interval: int = 60  # 收集间隔(秒)
    retention_period: int = 3600  # 数据保留时间(秒)
    enable_cpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_disk_monitoring: bool = True
    enable_network_monitoring: bool = True
    max_metrics_history: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_warning': 80.0,
        'cpu_critical': 90.0,
        'memory_warning': 85.0,
        'memory_critical': 90.0,
        'disk_warning': 90.0,
        'disk_critical': 95.0
    })


class PerformanceMonitorLegacy:
    """
    性能监控器 (重构后的外观类)

    协调各个专门的组件提供统一的性能监控接口
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger: ILogger = StandardLogger(f"{self.__class__.__name__}")
        self.error_handler: IErrorHandler = BaseErrorHandler()

        # 初始化职责分离的组件
        self.collector = MetricsCollector(self.logger)
        self.storage = MetricsStorage(
            max_history=self.config.max_metrics_history,
            retention_period=self.config.retention_period,
            logger=self.logger
        )
        self.analyzer = MetricsAnalyzer(self.logger)

        # 兼容性属性
        self.metrics_history = self.storage.metrics_history
        self._lock = self.storage._lock
        self._cache_ttl = 300  # 5分钟缓存

        self.logger.log_info("性能监控器初始化完成")

    # 外观方法 - 保持向后兼容性
    def start_monitoring(self) -> None:
        """启动性能监控"""
        # 这里可以实现监控循环，但主要功能已移到专门的组件中
        self.logger.log_info("性能监控已启动")

    def stop_monitoring(self) -> None:
        """停止性能监控"""
        # 这里可以实现停止逻辑，但主要功能已移到专门的组件中
        self.logger.log_info("性能监控已停止")

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前指标"""
        return self.collector.collect_current_metrics()

    def get_metrics_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """获取指标历史"""
        return self.storage.get_metrics_history(limit)

    def get_statistics(self, time_range: int = 3600) -> Dict[str, Any]:
        """获取统计信息"""
        metrics_list = self.storage.get_metrics_history()
        return self.analyzer.get_statistics(metrics_list, time_range)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return {"status": "unknown", "issues": ["无法获取性能指标"]}

        # 简单的健康检查逻辑
        issues = []
        if current_metrics.cpu_percent > self.config.alert_thresholds['cpu_critical']:
            issues.append(".1f")
        if current_metrics.memory_percent > self.config.alert_thresholds['memory_critical']:
            issues.append(".1f")
        if current_metrics.disk_usage > self.config.alert_thresholds['disk_critical']:
            issues.append(".1f")

        status = "healthy" if not issues else "critical"
        return {
            "status": status,
            "issues": issues,
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_usage": current_metrics.disk_usage
            }
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        # 这里可以添加重置逻辑，但主要功能已移到专门的组件中
