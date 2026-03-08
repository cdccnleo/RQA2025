import logging
"""
特征层统一监控器

提供特征层组件的统一监控接口，包括性能监控、指标收集、告警管理等功能。
"""

import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime
# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
except ImportError:
    # 降级到直接导入
    pass

logger = logging.getLogger(__name__)

from .metrics_collector import MetricsCollector
from .alert_manager import AlertManager
from .performance_analyzer import PerformanceAnalyzer


# logger is already defined above


class MetricType(Enum):

    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:

    """指标值数据结构"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class ComponentInfo:

    """组件信息数据结构"""
    name: str
    component_type: str
    status: str = "unknown"
    start_time: Optional[float] = None
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    last_update: Optional[float] = None


class FeaturesMonitor:

    """
    特征层统一监控器

    提供特征层组件的统一监控接口，包括：
    - 组件注册和管理
    - 性能指标收集
    - 实时告警
    - 性能分析
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化监控器

        Args:
            config: 监控配置
        """
        self.config = config or {}

        # 组件注册表
        self.components: Dict[str, ComponentInfo] = {}
        self.component_locks: Dict[str, threading.Lock] = {}

        # 指标收集器
        self.metrics_collector = MetricsCollector()

        # 告警管理器
        self.alert_manager = AlertManager()

        # 性能分析器
        self.performance_analyzer = PerformanceAnalyzer()

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = self.config.get('monitor_interval', 5.0)

        # 指标存储
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 性能阈值
        self.thresholds = self.config.get('thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'response_time': 1.0,
            'error_rate': 0.05
        })

        logger.info("特征层监控器初始化完成")

    def register_component(self, name: str, component_type: str,


                           component: Any = None) -> None:
        """
        注册组件到监控器

        Args:
            name: 组件名称
            component_type: 组件类型
            component: 组件实例
        """
        if name in self.components:
            logger.warning(f"组件 {name} 已存在，将更新注册信息")

        self.components[name] = ComponentInfo(
            name=name,
            component_type=component_type,
            status="registered",
            start_time=time.time()
        )
        self.component_locks[name] = threading.Lock()

        logger.info(f"组件 {name} ({component_type}) 注册成功")

    def unregister_component(self, name: str) -> None:
        """
        注销组件

        Args:
            name: 组件名称
        """
        if name in self.components:
            del self.components[name]
            del self.component_locks[name]
            logger.info(f"组件 {name} 注销成功")
        else:
            logger.warning(f"组件 {name} 不存在")

    def update_component_status(self, name: str, status: str,


                                metrics: Optional[Dict] = None) -> None:
        """
        更新组件状态

        Args:
            name: 组件名称
            status: 组件状态
            metrics: 组件指标
        """
        if name not in self.components:
            logger.warning(f"组件 {name} 未注册")
            return

        with self.component_locks[name]:
            self.components[name].status = status
            self.components[name].last_update = time.time()

            if metrics:
                for metric_name, value in metrics.items():
                    metric_value = MetricValue(
                        name=metric_name,
                        value=float(value),
                        timestamp=time.time(),
                        labels={'component': name}
                    )
                    self.components[name].metrics[metric_name] = metric_value

                    # 存储到历史记录
                    self.metrics_history[f"{name}.{metric_name}"].append(metric_value)

        logger.debug(f"组件 {name} 状态更新: {status}")

    def collect_metrics(self, component_name: str,


                        metric_name: str, value: float,
                        metric_type: MetricType = MetricType.GAUGE,
                        labels: Optional[Dict] = None) -> None:
        """
        收集组件指标

        Args:
            component_name: 组件名称
            metric_name: 指标名称
            value: 指标值
            metric_type: 指标类型
            labels: 标签
        """
        if component_name not in self.components:
            logger.warning(f"组件 {component_name} 未注册")
            return

        metric_value = MetricValue(
            name=metric_name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=metric_type
        )

        with self.component_locks[component_name]:
            self.components[component_name].metrics[metric_name] = metric_value
            self.metrics_history[f"{component_name}.{metric_name}"].append(metric_value)

        # 检查阈值告警
        self._check_thresholds(component_name, metric_name, value)

    def _check_thresholds(self, component_name: str, metric_name: str, value: float) -> None:
        """
        检查指标阈值

        Args:
            component_name: 组件名称
            metric_name: 指标名称
            value: 指标值
        """
        threshold_key = f"{component_name}.{metric_name}"
        threshold = self.thresholds.get(threshold_key)

        if threshold is not None:
            if value > threshold:
                self.alert_manager.send_alert(
                    level="warning",
                    message=f"组件 {component_name} 指标 {metric_name} 超过阈值: {value} > {threshold}",
                    component=component_name,
                    metric=metric_name,
                    value=value,
                    threshold=threshold
                )

    def get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """
        获取组件指标

        Args:
            component_name: 组件名称

        Returns:
            组件指标字典
        """
        if component_name not in self.components:
            return {}

        component = self.components[component_name]
        metrics = {}

        for metric_name, metric_value in component.metrics.items():
            metrics[metric_name] = {
                'value': metric_value.value,
                'timestamp': metric_value.timestamp,
                'type': metric_value.metric_type.value,
                'labels': metric_value.labels
            }

        return metrics

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有组件指标

        Returns:
            所有组件指标字典
        """
        all_metrics = {}

        for component_name in self.components:
            all_metrics[component_name] = self.get_component_metrics(component_name)

        return all_metrics

    def get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        获取组件状态

        Args:
            component_name: 组件名称

        Returns:
            组件状态信息
        """
        if component_name not in self.components:
            return None

        component = self.components[component_name]

        return {
            'name': component.name,
            'type': component.component_type,
            'status': component.status,
            'start_time': component.start_time,
            'last_update': component.last_update,
            'uptime': time.time() - component.start_time if component.start_time else 0
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有组件状态

        Returns:
            所有组件状态字典
        """
        all_status = {}

        for component_name in self.components:
            all_status[component_name] = self.get_component_status(component_name)

        return all_status

    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_monitoring:
            logger.warning("监控已启动")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("特征层监控已启动")

    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_monitoring:
            logger.warning("监控未启动")
            return

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("特征层监控已停止")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 分析性能
                self._analyze_performance()

                # 检查组件健康状态
                self._check_component_health()

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(self.monitor_interval)

    def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            self.collect_metrics("system", "cpu_usage", cpu_percent)

            memory = psutil.virtual_memory()
            self.collect_metrics("system", "memory_usage", memory.percent)

            disk = psutil.disk_usage('/')
            self.collect_metrics("system", "disk_usage", disk.percent)

        except ImportError:
            logger.warning("psutil未安装，无法收集系统指标")
        except Exception:
            logger.warning("收集系统指标失败，使用默认占位值")
            self.collect_metrics("system", "cpu_usage", 0.0)
            self.collect_metrics("system", "memory_usage", 0.0)
            self.collect_metrics("system", "disk_usage", 0.0)

    def _analyze_performance(self) -> None:
        """分析性能"""
        try:
            # 分析响应时间趋势
            for component_name in self.components:
                response_time_key = f"{component_name}.response_time"
                if response_time_key in self.metrics_history:
                    history = list(self.metrics_history[response_time_key])
                    if len(history) > 10:
                        recent_values = [m.value for m in history[-10:]]
                        avg_response_time = np.mean(recent_values)

                        # 检查响应时间趋势
                        if avg_response_time > self.thresholds.get('response_time', 1.0):
                            self.alert_manager.send_alert(
                                level="warning",
                                message=f"组件 {component_name} 平均响应时间过高: {avg_response_time:.3f}s",
                                component=component_name,
                                metric="response_time",
                                value=avg_response_time
                            )

        except Exception:
            logger.warning("性能分析失败，跳过本轮分析")

    def _check_component_health(self) -> None:
        """检查组件健康状态"""
        current_time = time.time()

        for component_name, component in self.components.items():
            # 检查组件是否超时
            if component.last_update:
                timeout = self.config.get('component_timeout', 300)  # 5分钟
                if current_time - component.last_update > timeout:
                    self.alert_manager.send_alert(
                        level="error",
                        message=f"组件 {component_name} 超时未更新",
                        component=component_name,
                        timeout=timeout
                    )

    def export_metrics(self, file_path: str) -> None:
        """
        导出指标数据

        Args:
            file_path: 导出文件路径
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'components': self.get_all_status(),
                'metrics': self.get_all_metrics(),
                'alerts': self.alert_manager.get_recent_alerts()
            }

            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"指标数据已导出到: {file_path}")

        except Exception as e:
            logger.error(f"导出指标数据失败: {str(e)}")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告

        Returns:
            性能报告字典
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_components': len(self.components),
                    'active_components': len([c for c in self.components.values() if c.status == 'active']),
                    'alerts_count': len(self.alert_manager.get_recent_alerts())
                },
                'components': self.get_all_status(),
                'performance_analysis': self.performance_analyzer.analyze_performance(self.metrics_history),
                'recent_alerts': self.alert_manager.get_recent_alerts()
            }

            return report

        except Exception as e:
            logger.error(f"生成性能报告失败: {str(e)}")
            return {}

    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()


# 全局监控器实例
_global_monitor: Optional[FeaturesMonitor] = None


def get_monitor(config: Optional[Dict] = None) -> FeaturesMonitor:
    """
    获取全局监控器实例

    Args:
        config: 监控配置

    Returns:
        监控器实例
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = FeaturesMonitor(config)

    return _global_monitor


def monitor_operation(component_name: str, operation_name: str):
    """
    监控操作装饰器

    Args:
        component_name: 组件名称
        operation_name: 操作名称
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            monitor = get_monitor()
            start_time = time.time()

            try:
                # 更新组件状态
                monitor.update_component_status(component_name, "running")

                # 执行操作
                result = func(*args, **kwargs)

                # 记录成功指标
                execution_time = time.time() - start_time
                monitor.collect_metrics(
                    component_name, f"{operation_name}_success", 1, MetricType.COUNTER)
                monitor.collect_metrics(
                    component_name, f"{operation_name}_duration", execution_time)

                # 更新组件状态
                monitor.update_component_status(component_name, "active")

                return result

            except Exception as e:
                # 记录失败指标
                execution_time = time.time() - start_time
                monitor.collect_metrics(
                    component_name, f"{operation_name}_error", 1, MetricType.COUNTER)
                monitor.collect_metrics(
                    component_name, f"{operation_name}_duration", execution_time)

                # 更新组件状态
                monitor.update_component_status(component_name, "error")

                raise

        return wrapper
    return decorator
