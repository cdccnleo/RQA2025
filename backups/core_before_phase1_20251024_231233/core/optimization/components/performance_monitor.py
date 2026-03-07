"""
性能监控器组件
"""

import logging
import time
import threading
from typing import Dict, List, Any
from dataclasses import dataclass

import psutil

from ...base import BaseComponent

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标"""

    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "general"


class PerformanceMonitor(BaseComponent):
    """性能监控器"""

    def __init__(self, monitoring_interval: int = 60):

        super().__init__("PerformanceMonitor")
        self.monitoring_interval = monitoring_interval
        self.metrics: List[PerformanceMetric] = []
        self.monitoring_thread = None
        self.is_monitoring = False
        self.metric_handlers = {}

        # 注册默认指标处理器
        self._register_default_handlers()

        logger.info("性能监控器初始化完成")

    def _register_default_handlers(self):
        """注册默认指标处理器"""
        self.metric_handlers.update(
            {
                "cpu_usage": self._collect_cpu_usage,
                "memory_usage": self._collect_memory_usage,
                "disk_usage": self._collect_disk_usage,
                "response_time": self._collect_response_time,
                "throughput": self._collect_throughput,
                "error_rate": self._collect_error_rate,
            }
        )

    def add_metric(self, metric: str):
        """添加监控指标"""
        if metric not in self.metric_handlers:
            logger.warning(f"未知的监控指标: {metric}")
            return

        logger.info(f"添加监控指标: {metric}")

    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info("性能监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("性能监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)

    def _collect_metrics(self):
        """收集指标"""
        timestamp = time.time()

        for metric_name, handler in self.metric_handlers.items():
            try:
                value = handler()
                metric = PerformanceMetric(
                    name=metric_name,
                    value=value,
                    unit=self._get_metric_unit(metric_name),
                    timestamp=timestamp,
                    category=self._get_metric_category(metric_name),
                )
                self.metrics.append(metric)
            except Exception as e:
                logger.error(f"收集指标 {metric_name} 失败: {e}")

    def _collect_cpu_usage(self) -> float:
        """收集CPU使用率"""
        return psutil.cpu_percent(interval=1)

    def _collect_memory_usage(self) -> float:
        """收集内存使用率"""
        memory = psutil.virtual_memory()
        return memory.percent

    def _collect_disk_usage(self) -> float:
        """收集磁盘使用率"""
        disk = psutil.disk_usage("/")
        return (disk.used / disk.total) * 100

    def _collect_response_time(self) -> float:
        """收集响应时间"""
        # 这里应该实现实际的响应时间收集逻辑
        return 0.0

    def _collect_throughput(self) -> float:
        """收集吞吐量"""
        # 这里应该实现实际的吞吐量收集逻辑
        return 0.0

    def _collect_error_rate(self) -> float:
        """收集错误率"""
        # 这里应该实现实际的错误率收集逻辑
        return 0.0

    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位"""
        units = {
            "cpu_usage": "%",
            "memory_usage": "%",
            "disk_usage": "%",
            "response_time": "ms",
            "throughput": "req / s",
            "error_rate": "%",
        }
        return units.get(metric_name, "")

    def _get_metric_category(self, metric_name: str) -> str:
        """获取指标类别"""
        categories = {
            "cpu_usage": "system",
            "memory_usage": "system",
            "disk_usage": "system",
            "response_time": "application",
            "throughput": "application",
            "error_rate": "application",
        }
        return categories.get(metric_name, "unknown")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics:
            return {"total_metrics": 0, "categories": {}}

        categories = {}
        for metric in self.metrics:
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append(metric)

        summary = {
            "total_metrics": len(self.metrics),
            "categories": {},
            "latest_metrics": {},
        }

        for category, metrics in categories.items():
            summary["categories"][category] = len(metrics)
            if metrics:
                latest = max(metrics, key=lambda x: x.timestamp)
                summary["latest_metrics"][category] = {
                    "name": latest.name,
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp,
                }

        return summary

    def shutdown(self) -> bool:
        """关闭性能监控器"""
        try:
            logger.info("开始关闭性能监控器")
            self.stop_monitoring()
            logger.info("性能监控器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭性能监控器失败: {e}")
            return False
