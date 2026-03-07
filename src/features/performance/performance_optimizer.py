# src / features / performance / performance_optimizer.py
"""
特征层性能优化器
实现内存优化、并发处理和缓存增强功能
"""

import logging
import threading
import time
import gc
import psutil
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import numpy as np
import pandas as pd
import hashlib
from dataclasses import dataclass, field
from enum import Enum

from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):

    """优化级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class CacheStrategy(Enum):

    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


@dataclass
class PerformanceMetrics:

    """性能指标"""
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    response_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    concurrent_requests: int = 0
    active_threads: int = 0
    gc_collections: int = 0
    timestamp: float = field(default_factory=time.time)


class MemoryOptimizer:

    """内存优化器"""

    def __init__(self, max_memory_mb: int = 1024, gc_threshold: float = 0.8):

        self.max_memory_mb = max_memory_mb
        self.gc_threshold = gc_threshold
        self.memory_history: List[float] = []
        self.gc_stats = {"collections": 0, "last_collection": 0}

    def check_memory_usage(self) -> float:
        """检查内存使用情况"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_history.append(memory_mb)

        # 保持历史记录在合理范围内
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-50:]

        return memory_mb

    def optimize_memory(self) -> Dict[str, Any]:
        """执行内存优化"""
        current_memory = self.check_memory_usage()
        optimization_result = {
            "before_memory_mb": current_memory,
            "after_memory_mb": current_memory,
            "optimizations_applied": [],
            "gc_triggered": False
        }

        # 检查是否需要垃圾回收
        if current_memory > self.max_memory_mb * self.gc_threshold:
            logger.info(f"内存使用率过高 ({current_memory:.2f}MB)，触发垃圾回收")
            gc.collect()
            self.gc_stats["collections"] += 1
            self.gc_stats["last_collection"] = time.time()
            optimization_result["gc_triggered"] = True

            # 重新检查内存使用
            after_memory = self.check_memory_usage()
            optimization_result["after_memory_mb"] = after_memory
            optimization_result["optimizations_applied"].append("garbage_collection")

        # 检查内存泄漏
        if len(self.memory_history) > 10:
            recent_trend = np.polyfit(range(len(self.memory_history[-10:])),
                                      self.memory_history[-10:], 1)[0]
            if recent_trend > 1.0:  # 内存持续增长
                logger.warning(f"检测到可能的内存泄漏，增长趋势: {recent_trend:.2f}MB / 次")
                optimization_result["optimizations_applied"].append("memory_leak_detection")

        return optimization_result

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current_memory = self.check_memory_usage()
        return {
            "current_memory_mb": current_memory,
            "max_memory_mb": self.max_memory_mb,
            "usage_percent": (current_memory / self.max_memory_mb) * 100,
            "memory_history": self.memory_history[-10:],
            "gc_stats": self.gc_stats
        }


class CacheOptimizer:

    """缓存优化器"""

    def __init__(self, max_cache_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):

        self.max_cache_size = max_cache_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                self.cache_stats["hits"] += 1
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self.cache[key]
            else:
                self.cache_stats["misses"] += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        with self.lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_cache_size:
                self._evict_entries()

            self.cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            self.access_times[key] = time.time()
            self.access_counts[key] = 0
            self.cache_stats["size"] = len(self.cache)

    def _evict_entries(self) -> None:
        """根据策略淘汰缓存项"""
        if self.strategy == CacheStrategy.LRU:
            # LRU: 淘汰最久未访问的
            oldest_key = min(self.access_times.keys(),
                             key=lambda k: self.access_times[k])
            self._remove_entry(oldest_key)
        elif self.strategy == CacheStrategy.LFU:
            # LFU: 淘汰访问次数最少的
            least_frequent_key = min(self.access_counts.keys(),
                                     key=lambda k: self.access_counts[k])
            self._remove_entry(least_frequent_key)
        elif self.strategy == CacheStrategy.FIFO:
            # FIFO: 淘汰最早加入的
            oldest_key = min(self.cache.keys(),
                             key=lambda k: self.cache[k]["timestamp"])
            self._remove_entry(oldest_key)

    def _remove_entry(self, key: str) -> None:
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
            self.cache_stats["evictions"] += 1
            self.cache_stats["size"] = len(self.cache)

    def clear_expired(self) -> int:
        """清理过期缓存项"""
        current_time = time.time()
        expired_keys = []

        for key, item in self.cache.items():
            if item["ttl"] and (current_time - item["timestamp"]) > item["ttl"]:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

        return len(expired_keys)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "size": self.cache_stats["size"],
            "max_size": self.max_cache_size,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "evictions": self.cache_stats["evictions"],
            "strategy": self.strategy.value
        }


class ConcurrencyOptimizer:

    """并发优化器"""

    def __init__(self, max_workers: int = 4, max_processes: int = 2):

        self.max_workers = max_workers
        self.max_processes = max_processes
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.RLock()

    def submit_task(self, func: Callable, *args, **kwargs):
        """提交任务到线程池"""
        with self.lock:
            self.active_tasks += 1

        future = self.thread_pool.submit(func, *args, **kwargs)
        future.add_done_callback(self._task_completed)
        return future

    def submit_process_task(self, func: Callable, *args, **kwargs):
        """提交任务到进程池"""
        with self.lock:
            self.active_tasks += 1

        future = self.process_pool.submit(func, *args, **kwargs)
        future.add_done_callback(self._task_completed)
        return future

    def _task_completed(self, future):
        """任务完成回调"""
        with self.lock:
            self.active_tasks -= 1
            self.completed_tasks += 1

            if future.exception():
                self.failed_tasks += 1
                logger.error(f"任务执行失败: {future.exception()}")

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """获取并发统计信息"""
        return {
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "max_workers": self.max_workers,
            "max_processes": self.max_processes,
            "thread_pool_size": len(self.thread_pool._threads),
            "process_pool_size": len(self.process_pool._processes)
        }

    def shutdown(self):
        """关闭线程池和进程池"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class PerformanceOptimizer:

    """性能优化器主类"""

    def __init__(


        self,
        optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM,
        config_manager=None
    ):
        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        # 优化级别
        self.optimization_level = optimization_level

        # 初始化优化组件
        self.memory_optimizer = MemoryOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.concurrency_optimizer = ConcurrencyOptimizer()

        # 性能监控
        self.performance_history: List[PerformanceMetrics] = []
        self.monitoring_enabled = True
        self.monitoring_thread = None

        # 启动监控
        self._start_monitoring()

        logger.info(f"性能优化器初始化完成，优化级别: {optimization_level.value}")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.PROCESSING:
            if key == "optimization_level":
                self.optimization_level = OptimizationLevel(value)
                logger.info(f"更新优化级别: {value}")
            elif key == "monitoring_enabled":
                self.monitoring_enabled = value
                logger.info(f"更新监控状态: {value}")

    def _start_monitoring(self):
        """启动性能监控"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()

    def _monitor_performance(self):
        """性能监控循环"""
        while self.monitoring_enabled:
            try:
                metrics = self._collect_performance_metrics()
                self.performance_history.append(metrics)

                # 保持历史记录在合理范围内
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-500:]

                # 根据优化级别执行优化
                self._apply_optimizations()

                time.sleep(5)  # 每5秒收集一次指标

            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                time.sleep(10)

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        process = psutil.Process()

        # 内存使用
        memory_mb = process.memory_info().rss / 1024 / 1024

        # CPU使用
        cpu_percent = process.cpu_percent()

        # 缓存命中率
        cache_stats = self.cache_optimizer.get_cache_stats()
        cache_hit_rate = cache_stats.get("hit_rate", 0.0)

        # 并发统计
        concurrency_stats = self.concurrency_optimizer.get_concurrency_stats()

        return PerformanceMetrics(
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            cache_hit_rate=cache_hit_rate,
            concurrent_requests=concurrency_stats["active_tasks"],
            active_threads=concurrency_stats["thread_pool_size"],
            gc_collections=self.memory_optimizer.gc_stats["collections"]
        )

    def _apply_optimizations(self):
        """根据优化级别应用优化策略"""
        if self.optimization_level == OptimizationLevel.LOW:
            # 低级别优化：基本监控
            pass
        elif self.optimization_level == OptimizationLevel.MEDIUM:
            # 中级别优化：内存优化 + 缓存清理
            self.memory_optimizer.optimize_memory()
            self.cache_optimizer.clear_expired()
        elif self.optimization_level == OptimizationLevel.HIGH:
            # 高级别优化：主动内存管理 + 缓存优化
            self.memory_optimizer.optimize_memory()
            self.cache_optimizer.clear_expired()
            # 可以添加更多高级优化
        elif self.optimization_level == OptimizationLevel.EXTREME:
            # 极高级别优化：所有优化策略
            self.memory_optimizer.optimize_memory()
            self.cache_optimizer.clear_expired()
            # 可以添加更激进的优化策略

    def optimize_data_processing(self, data: pd.DataFrame, operation: str) -> pd.DataFrame:
        """优化数据处理"""
        # 根据数据大小和操作类型选择优化策略
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024

        if data_size_mb > 100:  # 大数据集
            logger.info(f"检测到大数据集 ({data_size_mb:.2f}MB)，应用大数据优化策略")
            return self._optimize_large_data(data, operation)
        else:
            return self._optimize_small_data(data, operation)

    def _optimize_large_data(self, data: pd.DataFrame, operation: str) -> pd.DataFrame:
        """大数据优化策略"""
        # 分块处理
        chunk_size = 10000
        chunks = []

        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            # 这里可以添加具体的优化逻辑
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    def _optimize_small_data(self, data: pd.DataFrame, operation: str) -> pd.DataFrame:
        """小数据优化策略"""
        # 对于小数据集，直接返回
        return data

    def cache_function_result(self, func: Callable, *args, **kwargs):
        """缓存函数结果装饰器"""

        def wrapper(*func_args, **func_kwargs):

            # 生成缓存键
            cache_key = self._generate_cache_key(func.__name__, func_args, func_kwargs)

            # 尝试从缓存获取
            cached_result = self.cache_optimizer.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行函数
            result = func(*func_args, **func_kwargs)

            # 缓存结果
            self.cache_optimizer.set(cache_key, result, ttl=3600)  # 1小时过期

            return result

        return wrapper

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 将参数序列化并生成哈希
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {"error": "没有性能数据"}

        latest_metrics = self.performance_history[-1]

        # 计算趋势
        if len(self.performance_history) > 10:
            recent_memory = [m.memory_usage_mb for m in self.performance_history[-10:]]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        else:
            memory_trend = 0.0

        return {
            "current_metrics": {
                "memory_usage_mb": latest_metrics.memory_usage_mb,
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "cache_hit_rate": latest_metrics.cache_hit_rate,
                "concurrent_requests": latest_metrics.concurrent_requests,
                "active_threads": latest_metrics.active_threads
            },
            "optimization_level": self.optimization_level.value,
            "memory_trend_mb_per_minute": memory_trend * 12,  # 转换为每分钟
            "cache_stats": self.cache_optimizer.get_cache_stats(),
            "concurrency_stats": self.concurrency_optimizer.get_concurrency_stats(),
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "history_length": len(self.performance_history)
        }

    def set_optimization_level(self, level: OptimizationLevel):
        """设置优化级别"""
        self.optimization_level = level
        logger.info(f"优化级别已设置为: {level.value}")

    def shutdown(self):
        """关闭性能优化器"""
        self.monitoring_enabled = False
        self.concurrency_optimizer.shutdown()
        logger.info("性能优化器已关闭")


# 全局性能优化器实例
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def optimize_performance(level: OptimizationLevel = OptimizationLevel.MEDIUM):
    """性能优化装饰器"""

    def decorator(func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):

            optimizer = get_performance_optimizer()
            optimizer.set_optimization_level(level)

            # 应用缓存优化
            cached_func = optimizer.cache_function_result(func)
            return cached_func(*args, **kwargs)

        return wrapper
    return decorator
