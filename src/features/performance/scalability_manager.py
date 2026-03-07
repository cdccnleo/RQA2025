# src / features / performance / scalability_manager.py
"""
扩展性管理器
实现负载均衡、自动扩缩容和资源管理功能
"""

import logging
import threading
import time
import queue
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import numpy as np
from collections import deque
from functools import wraps

from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):

    """扩缩容策略"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):

    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"


@dataclass
class WorkerNode:

    """工作节点"""
    id: str
    capacity: int = 100
    current_load: int = 0
    response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    weight: float = 1.0


@dataclass
class ScalingMetrics:

    """扩缩容指标"""
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    queue_threshold: int = 100
    min_workers: int = 2
    max_workers: int = 10
    scale_up_cooldown: int = 60  # 秒
    scale_down_cooldown: int = 300  # 秒


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):

        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.current_index = 0
        self.lock = threading.RLock()

    def add_worker(self, worker: WorkerNode):
        """添加工作节点"""
        with self.lock:
            self.workers[worker.id] = worker
            logger.info(f"添加工作节点: {worker.id}")

    def remove_worker(self, worker_id: str):
        """移除工作节点"""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"移除工作节点: {worker_id}")

    def get_next_worker(self) -> Optional[WorkerNode]:
        """获取下一个工作节点"""
        with self.lock:
            if not self.workers:
                return None

            healthy_workers = [w for w in self.workers.values() if w.is_healthy]
            if not healthy_workers:
                return None

            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker = healthy_workers[self.current_index % len(healthy_workers)]
                self.current_index += 1
                return worker

            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_workers, key=lambda w: w.current_load)

            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                # 基于权重的轮询
                total_weight = sum(w.weight for w in healthy_workers)
                if total_weight == 0:
                    return healthy_workers[0]

                # 简化的权重轮询实现
                worker = healthy_workers[self.current_index % len(healthy_workers)]
                self.current_index += 1
                return worker

            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return min(healthy_workers, key=lambda w: w.response_time_ms)

            return healthy_workers[0]

    def update_worker_metrics(self, worker_id: str, **metrics):
        """更新工作节点指标"""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                for key, value in metrics.items():
                    if hasattr(worker, key):
                        setattr(worker, key, value)

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计信息"""
        with self.lock:
            total_workers = len(self.workers)
            healthy_workers = len([w for w in self.workers.values() if w.is_healthy])
            total_load = sum(w.current_load for w in self.workers.values())
            avg_response_time = np.mean(
                [w.response_time_ms for w in self.workers.values()]) if self.workers else 0

            return {
                "strategy": self.strategy.value,
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "total_load": total_load,
                "average_response_time_ms": avg_response_time,
                "workers": [
                    {
                        "id": w.id,
                        "load": w.current_load,
                        "response_time": w.response_time_ms,
                        "healthy": w.is_healthy
                    }
                    for w in self.workers.values()
                ]
            }


class AutoScaler:

    """自动扩缩容器"""

    def __init__(self, metrics: ScalingMetrics, strategy: ScalingStrategy = ScalingStrategy.HYBRID):

        self.metrics = metrics
        self.strategy = strategy
        self.current_workers = metrics.min_workers
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.scaling_history: List[Dict[str, Any]] = []

        # 性能指标历史
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.queue_history = deque(maxlen=100)

    def should_scale_up(self, current_metrics: Dict[str, float]) -> bool:
        """判断是否需要扩容"""
        current_time = time.time()

        # 检查冷却时间
        if current_time - self.last_scale_up < self.metrics.scale_up_cooldown:
            return False

        # 检查是否已达到最大工作节点数
        if self.current_workers >= self.metrics.max_workers:
            return False

        # 根据策略判断
        if self.strategy == ScalingStrategy.CPU_BASED:
            return current_metrics.get('cpu_usage', 0) > self.metrics.cpu_threshold

        elif self.strategy == ScalingStrategy.MEMORY_BASED:
            return current_metrics.get('memory_usage', 0) > self.metrics.memory_threshold

        elif self.strategy == ScalingStrategy.QUEUE_BASED:
            return current_metrics.get('queue_size', 0) > self.metrics.queue_threshold

        elif self.strategy == ScalingStrategy.HYBRID:
            cpu_high = current_metrics.get('cpu_usage', 0) > self.metrics.cpu_threshold
            memory_high = current_metrics.get('memory_usage', 0) > self.metrics.memory_threshold
            queue_high = current_metrics.get('queue_size', 0) > self.metrics.queue_threshold

            return cpu_high or memory_high or queue_high

        return False

    def should_scale_down(self, current_metrics: Dict[str, float]) -> bool:
        """判断是否需要缩容"""
        current_time = time.time()

        # 检查冷却时间
        if current_time - self.last_scale_down < self.metrics.scale_down_cooldown:
            return False

        # 检查是否已达到最小工作节点数
        if self.current_workers <= self.metrics.min_workers:
            return False

        # 根据策略判断
        if self.strategy == ScalingStrategy.CPU_BASED:
            return current_metrics.get('cpu_usage', 0) < self.metrics.cpu_threshold * 0.5

        elif self.strategy == ScalingStrategy.MEMORY_BASED:
            return current_metrics.get('memory_usage', 0) < self.metrics.memory_threshold * 0.5

        elif self.strategy == ScalingStrategy.QUEUE_BASED:
            return current_metrics.get('queue_size', 0) < self.metrics.queue_threshold * 0.3

        elif self.strategy == ScalingStrategy.HYBRID:
            cpu_low = current_metrics.get('cpu_usage', 0) < self.metrics.cpu_threshold * 0.5
            memory_low = current_metrics.get(
                'memory_usage', 0) < self.metrics.memory_threshold * 0.5
            queue_low = current_metrics.get('queue_size', 0) < self.metrics.queue_threshold * 0.3

            return cpu_low and memory_low and queue_low

        return False

    def update_metrics(self, metrics: Dict[str, float]):
        """更新性能指标"""
        self.cpu_history.append(metrics.get('cpu_usage', 0))
        self.memory_history.append(metrics.get('memory_usage', 0))
        self.queue_history.append(metrics.get('queue_size', 0))

    def scale_up(self) -> int:
        """扩容"""
        old_workers = self.current_workers
        self.current_workers = min(self.current_workers + 1, self.metrics.max_workers)
        self.last_scale_up = time.time()

        scaling_event = {
            "type": "scale_up",
            "timestamp": time.time(),
            "old_workers": old_workers,
            "new_workers": self.current_workers,
            "reason": f"触发{self.strategy.value}扩容策略"
        }
        self.scaling_history.append(scaling_event)

        logger.info(f"扩容: {old_workers} -> {self.current_workers} 个工作节点")
        return self.current_workers

    def scale_down(self) -> int:
        """缩容"""
        old_workers = self.current_workers
        self.current_workers = max(self.current_workers - 1, self.metrics.min_workers)
        self.last_scale_down = time.time()

        scaling_event = {
            "type": "scale_down",
            "timestamp": time.time(),
            "old_workers": old_workers,
            "new_workers": self.current_workers,
            "reason": f"触发{self.strategy.value}缩容策略"
        }
        self.scaling_history.append(scaling_event)

        logger.info(f"缩容: {old_workers} -> {self.current_workers} 个工作节点")
        return self.current_workers

    def get_scaling_stats(self) -> Dict[str, Any]:
        """获取扩缩容统计信息"""
        return {
            "current_workers": self.current_workers,
            "strategy": self.strategy.value,
            "metrics": {
                "cpu_threshold": self.metrics.cpu_threshold,
                "memory_threshold": self.metrics.memory_threshold,
                "queue_threshold": self.metrics.queue_threshold,
                "min_workers": self.metrics.min_workers,
                "max_workers": self.metrics.max_workers
            },
            "recent_scaling_events": self.scaling_history[-10:],
            "performance_trends": {
                "cpu_avg": np.mean(list(self.cpu_history)) if self.cpu_history else 0,
                "memory_avg": np.mean(list(self.memory_history)) if self.memory_history else 0,
                "queue_avg": np.mean(list(self.queue_history)) if self.queue_history else 0
            }
        }


class ResourceManager:

    """资源管理器"""

    def __init__(self, max_memory_mb: int = 2048, max_cpu_percent: float = 80.0):

        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.resource_usage_history: List[Dict[str, float]] = []
        self.alerts: List[Dict[str, Any]] = []

    def check_resource_usage(self) -> Dict[str, float]:
        """检查资源使用情况"""
        process = psutil.Process()

        # 内存使用
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = (memory_mb / self.max_memory_mb) * 100

        # CPU使用
        cpu_percent = process.cpu_percent()

        # 磁盘使用
        disk_usage = psutil.disk_usage('/')
        disk_percent = disk_usage.percent

        usage = {
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "disk_percent": disk_percent
        }

        self.resource_usage_history.append(usage)

        # 保持历史记录在合理范围内
        if len(self.resource_usage_history) > 1000:
            self.resource_usage_history = self.resource_usage_history[-500:]

        return usage

    def check_resource_alerts(self) -> List[Dict[str, Any]]:
        """检查资源告警"""
        current_usage = self.check_resource_usage()
        alerts = []

        # 内存告警
        if current_usage["memory_percent"] > 90:
            alerts.append({
                "type": "memory_high",
                "level": "critical",
                "message": f"内存使用率过高: {current_usage['memory_percent']:.1f}%",
                "value": current_usage["memory_percent"],
                "timestamp": time.time()
            })
        elif current_usage["memory_percent"] > 80:
            alerts.append({
                "type": "memory_high",
                "level": "warning",
                "message": f"内存使用率较高: {current_usage['memory_percent']:.1f}%",
                "value": current_usage["memory_percent"],
                "timestamp": time.time()
            })

        # CPU告警
        if current_usage["cpu_percent"] > 90:
            alerts.append({
                "type": "cpu_high",
                "level": "critical",
                "message": f"CPU使用率过高: {current_usage['cpu_percent']:.1f}%",
                "value": current_usage["cpu_percent"],
                "timestamp": time.time()
            })
        elif current_usage["cpu_percent"] > 80:
            alerts.append({
                "type": "cpu_high",
                "level": "warning",
                "message": f"CPU使用率较高: {current_usage['cpu_percent']:.1f}%",
                "value": current_usage["cpu_percent"],
                "timestamp": time.time()
            })

        # 磁盘告警
        if current_usage["disk_percent"] > 90:
            alerts.append({
                "type": "disk_high",
                "level": "critical",
                "message": f"磁盘使用率过高: {current_usage['disk_percent']:.1f}%",
                "value": current_usage["disk_percent"],
                "timestamp": time.time()
            })

        self.alerts.extend(alerts)
        return alerts

    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        if not self.resource_usage_history:
            return {"error": "没有资源使用数据"}

        recent_usage = self.resource_usage_history[-10:]

        return {
            "current_usage": self.resource_usage_history[-1] if self.resource_usage_history else {},
            "limits": {
                "max_memory_mb": self.max_memory_mb,
                "max_cpu_percent": self.max_cpu_percent
            },
            "averages": {
                "memory_percent": np.mean([u["memory_percent"] for u in recent_usage]),
                "cpu_percent": np.mean([u["cpu_percent"] for u in recent_usage]),
                "disk_percent": np.mean([u["disk_percent"] for u in recent_usage])
            },
            "recent_alerts": self.alerts[-10:],
            "history_length": len(self.resource_usage_history)
        }


class ScalabilityManager:

    """扩展性管理器主类"""

    def __init__(


        self,
        scaling_metrics: ScalingMetrics,
        scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        config_manager=None
    ):
        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        # 初始化组件
        self.load_balancer = LoadBalancer(load_balancing_strategy)
        self.auto_scaler = AutoScaler(scaling_metrics, scaling_strategy)
        self.resource_manager = ResourceManager()

        # 任务队列
        self.task_queue = queue.Queue()
        self.monitoring_enabled = True
        self.monitoring_thread = None

        # 启动监控
        self._start_monitoring()

        logger.info(f"扩展性管理器初始化完成")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.PROCESSING:
            if key == "monitoring_enabled":
                self.monitoring_enabled = value
                logger.info(f"更新监控状态: {value}")

    def _start_monitoring(self):
        """启动监控"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitor_scalability, daemon=True)
            self.monitoring_thread.start()

    def _monitor_scalability(self):
        """扩展性监控循环"""
        while self.monitoring_enabled:
            try:
                # 检查资源使用
                resource_usage = self.resource_manager.check_resource_usage()
                alerts = self.resource_manager.check_resource_alerts()

                # 更新自动扩缩容指标
                self.auto_scaler.update_metrics({
                    'cpu_usage': resource_usage['cpu_percent'] / 100,
                    'memory_usage': resource_usage['memory_percent'] / 100,
                    'queue_size': self.task_queue.qsize()
                })

                # 检查是否需要扩缩容
                if self.auto_scaler.should_scale_up(resource_usage):
                    self.auto_scaler.scale_up()

                elif self.auto_scaler.should_scale_down(resource_usage):
                    self.auto_scaler.scale_down()

                # 处理告警
                for alert in alerts:
                    logger.warning(f"资源告警: {alert['message']}")

                time.sleep(10)  # 每10秒检查一次

            except Exception as e:
                logger.error(f"扩展性监控错误: {e}")
                time.sleep(30)

    def submit_task(self, task_func: Callable, *args, **kwargs):
        """提交任务"""
        worker = self.load_balancer.get_next_worker()
        if worker:
            # 更新工作节点负载
            worker.current_load += 1
            self.load_balancer.update_worker_metrics(worker.id, current_load=worker.current_load)

            # 提交任务到队列
            self.task_queue.put((task_func, args, kwargs))
            return True
        else:
            logger.warning("没有可用的工作节点")
            return False

    def add_worker_node(self, worker_id: str, capacity: int = 100):
        """添加工作节点"""
        worker = WorkerNode(
            id=worker_id,
            capacity=capacity,
            current_load=0
        )
        self.load_balancer.add_worker(worker)

    def remove_worker_node(self, worker_id: str):
        """移除工作节点"""
        self.load_balancer.remove_worker(worker_id)

    def get_scalability_report(self) -> Dict[str, Any]:
        """获取扩展性报告"""
        return {
            "load_balancer": self.load_balancer.get_load_balancer_stats(),
            "auto_scaler": self.auto_scaler.get_scaling_stats(),
            "resource_manager": self.resource_manager.get_resource_stats(),
            "task_queue_size": self.task_queue.qsize(),
            "monitoring_enabled": self.monitoring_enabled
        }

    def shutdown(self):
        """关闭扩展性管理器"""
        self.monitoring_enabled = False
        logger.info("扩展性管理器已关闭")


# 全局扩展性管理器实例
_scalability_manager = None


def get_scalability_manager() -> ScalabilityManager:
    """获取全局扩展性管理器实例"""
    global _scalability_manager
    if _scalability_manager is None:
        metrics = ScalingMetrics()
        _scalability_manager = ScalabilityManager(metrics)
    return _scalability_manager


def scale_task(func: Callable):
    """任务扩缩容装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):

        manager = get_scalability_manager()
        return manager.submit_task(func, *args, **kwargs)
    return wrapper
