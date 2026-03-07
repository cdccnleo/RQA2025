#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略服务层性能优化器
Strategy Service Layer Performance Optimizer

基于业务流程驱动架构，实现策略服务的性能优化：
1. 智能缓存管理
2. 并发处理优化
3. 内存使用优化
4. 计算资源调度
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from functools import wraps
import time

from core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:

    """性能指标"""
    response_time: float          # 响应时间 (ms)
    throughput: float            # 吞吐量 (ops / sec)
    memory_usage: float          # 内存使用 (MB)
    cpu_usage: float            # CPU使用率 (%)
    cache_hit_rate: float       # 缓存命中率 (%)
    error_rate: float           # 错误率 (%)
    timestamp: datetime         # 时间戳


@dataclass
class OptimizationConfig:

    """优化配置"""
    max_concurrent_strategies: int = 10      # 最大并发策略数
    cache_ttl_seconds: int = 300            # 缓存TTL (秒)
    memory_limit_mb: int = 500              # 内存限制 (MB)
    cpu_limit_percent: int = 80             # CPU限制 (%)
    batch_size: int = 100                   # 批处理大小
    enable_async_processing: bool = True    # 启用异步处理
    enable_memory_optimization: bool = True  # 启用内存优化
    enable_cache_optimization: bool = True  # 启用缓存优化


class StrategyCacheManager:

    """
    策略缓存管理器
    Strategy Cache Manager

    提供多级缓存管理和智能缓存策略。
    """

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self.adapter_factory = get_unified_adapter_factory()
        self.cache_adapter = self.adapter_factory.get_adapter("cache")

        # L1缓存 - 内存缓存 (快速访问)
        self.l1_cache: Dict[str, Dict[str, Any]] = {}
        self.l1_timestamps: Dict[str, datetime] = {}

        # L2缓存 - 分布式缓存 (持久化)
        self.l2_cache_enabled = True

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        logger.info("StrategyCacheManager initialized")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        # L1缓存检查
        if key in self.l1_cache:
            if self._is_cache_valid(key):
                self.cache_hits += 1
                return self.l1_cache[key]
            else:
                # 缓存过期，删除
                del self.l1_cache[key]
                del self.l1_timestamps[key]
                self.cache_evictions += 1

        # L2缓存检查
        if self.l2_cache_enabled:
            l2_data = self.cache_adapter.get(key)
        if l2_data:
            # 加载到L1缓存
            self.l1_cache[key] = l2_data
            self.l1_timestamps[key] = datetime.now()
            self.cache_hits += 1
            return l2_data

        self.cache_misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存数据"""
        ttl = ttl or self.config.cache_ttl_seconds

        # L1缓存
        self.l1_cache[key] = value
        self.l1_timestamps[key] = datetime.now()

        # L2缓存
        if self.l2_cache_enabled:
            self.cache_adapter.set(key, value, ttl=ttl)

    def delete(self, key: str) -> None:
        """删除缓存数据"""
        # L1缓存
        if key in self.l1_cache:
            del self.l1_cache[key]
        if key in self.l1_timestamps:
            del self.l1_timestamps[key]

        # L2缓存
        if self.l2_cache_enabled:
            self.cache_adapter.delete(key)

    def clear_expired(self) -> int:
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []

        for key, timestamp in self.l1_timestamps.items():
            if (current_time - timestamp).seconds > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.l1_cache[key]
            del self.l1_timestamps[key]

        self.cache_evictions += len(expired_keys)
        return len(expired_keys)

    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self.l1_timestamps:
            return False

        age = (datetime.now() - self.l1_timestamps[key]).seconds
        return age < self.config.cache_ttl_seconds

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "l1_cache_size": len(self.l1_cache),
            "total_requests": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "hit_rate_percent": hit_rate,
            "l2_cache_enabled": self.l2_cache_enabled
        }


class ConcurrentStrategyExecutor:

    """
    并发策略执行器
    Concurrent Strategy Executor

    提供策略的并发执行和管理功能。
    """

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_strategies)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_strategies)
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # 性能监控
        self.execution_times: List[float] = []
        self.error_count = 0

        logger.info(
            f"ConcurrentStrategyExecutor initialized with {config.max_concurrent_strategies} workers")

    async def execute_strategy_async(self, strategy_id: str, func: Callable, *args, **kwargs) -> Any:
        """
        异步执行策略

        Args:
            strategy_id: 策略ID
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            执行结果
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                # 创建任务
                task = asyncio.create_task(self._execute_in_thread(func, *args, **kwargs))
                self.active_tasks[strategy_id] = task

                # 执行任务
                result = await task

                # 记录执行时间
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                logger.debug(f"Strategy {strategy_id} executed in {execution_time:.3f}s")
                return result

            except Exception as e:
                self.error_count += 1
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                logger.error(
                    f"Strategy {strategy_id} execution failed after {execution_time:.3f}s: {e}")
                raise
            finally:
                # 清理任务
                if strategy_id in self.active_tasks:
                    del self.active_tasks[strategy_id]

    async def _execute_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """在线程池中执行函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    def execute_batch(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """
        批量执行策略

        Args:
            tasks: 任务列表，每个任务包含 'strategy_id', 'func', 'args', 'kwargs'

        Returns:
            执行结果列表
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_strategies) as executor:
            # 提交所有任务
            future_to_task = {}
            for task in tasks:
                strategy_id = task['strategy_id']
                func = task['func']
                args = task.get('args', ())
                kwargs = task.get('kwargs', {})

                future = executor.submit(func, *args, **kwargs)
                future_to_task[future] = strategy_id

            # 收集结果
            for future in as_completed(future_to_task):
                strategy_id = future_to_task[future]
                try:
                    result = future.result()
                    results.append({
                        'strategy_id': strategy_id,
                        'result': result,
                        'status': 'success'
                    })
                except Exception as e:
                    self.error_count += 1
                    results.append({
                        'strategy_id': strategy_id,
                        'error': str(e),
                        'status': 'error'
                    })
                    logger.error(f"Batch execution failed for strategy {strategy_id}: {e}")

        return results

    def get_active_tasks_count(self) -> int:
        """获取活跃任务数量"""
        return len(self.active_tasks)

    def cancel_task(self, strategy_id: str) -> bool:
        """取消任务"""
        if strategy_id in self.active_tasks:
            task = self.active_tasks[strategy_id]
        if not task.done():
            task.cancel()
            return True
        return False

    def shutdown(self) -> None:
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("ConcurrentStrategyExecutor shut down")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total_executions = len(self.execution_times)
        avg_execution_time = sum(self.execution_times) / \
            total_executions if total_executions > 0 else 0
        error_rate = (self.error_count / total_executions * 100) if total_executions > 0 else 0

        return {
            "total_executions": total_executions,
            "avg_execution_time": avg_execution_time,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "active_tasks": len(self.active_tasks),
            "max_workers": self.config.max_concurrent_strategies
        }


class MemoryOptimizer:

    """
    内存优化器
    Memory Optimizer

    提供内存使用优化和垃圾回收功能。
    """

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # 内存监控
        self.memory_checkpoints: List[Dict[str, Any]] = []
        self.gc_threshold = 100  # GC阈值 (MB)

        logger.info(f"MemoryOptimizer initialized, initial memory: {self.initial_memory:.1f}MB")

    def check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss / 1024 / 1024  # MB

        # 获取系统内存信息
        system_memory = psutil.virtual_memory()

        memory_stats = {
            "current_memory_mb": current_memory,
            "memory_increase_mb": current_memory - self.initial_memory,
            "system_memory_percent": system_memory.percent,
            "available_memory_mb": system_memory.available / 1024 / 1024,
            "memory_limit_mb": self.config.memory_limit_mb,
            "timestamp": datetime.now()
        }

        # 记录检查点
        self.memory_checkpoints.append(memory_stats)

        # 只保留最近100个检查点
        if len(self.memory_checkpoints) > 100:
            self.memory_checkpoints = self.memory_checkpoints[-100:]

        return memory_stats

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        before_memory = self.check_memory_usage()["current_memory_mb"]

        # 强制垃圾回收
        collected_objects = gc.collect()
        gc.collect()  # 二次GC

        after_memory = self.check_memory_usage()["current_memory_mb"]
        memory_saved = before_memory - after_memory

        # 清理缓存
        self._clear_caches()

        result = {
            "before_memory_mb": before_memory,
            "after_memory_mb": after_memory,
            "memory_saved_mb": memory_saved,
            "collected_objects": collected_objects,
            "optimization_timestamp": datetime.now()
        }

        if memory_saved > 0:
            logger.info(f"Memory optimization: saved {memory_saved:.1f}MB")
        else:
            logger.debug("Memory optimization completed, no significant savings")

        return result

    def _clear_caches(self) -> None:
        """清理各种缓存"""
        try:
            # 清理模块缓存
            import sys
            modules_to_clear = []
            for module_name in sys.modules:
                if any(pattern in module_name for pattern in ['cache', 'temp', 'tmp']):
                    modules_to_clear.append(module_name)

            for module_name in modules_to_clear[:10]:  # 限制清理数量
                if module_name in sys.modules:
                    del sys.modules[module_name]

            logger.debug(f"Cleared {len(modules_to_clear)} cached modules")

        except Exception as e:
            logger.warning(f"Error clearing caches: {e}")

    def should_trigger_gc(self) -> bool:
        """判断是否应该触发垃圾回收"""
        memory_stats = self.check_memory_usage()

        # 检查内存使用是否超过阈值
        if memory_stats["current_memory_mb"] > self.config.memory_limit_mb:
            return True

        # 检查内存增长是否过快
        if len(self.memory_checkpoints) >= 5:
            recent_checkpoints = self.memory_checkpoints[-5:]
            memory_growth = recent_checkpoints[-1]["current_memory_mb"] - \
                recent_checkpoints[0]["current_memory_mb"]

        if memory_growth > self.gc_threshold:
            return True

        return False

    def get_memory_history(self) -> List[Dict[str, Any]]:
        """获取内存历史记录"""
        return self.memory_checkpoints.copy()

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.memory_checkpoints:
            return {}

        checkpoints = self.memory_checkpoints
        current = checkpoints[-1]

        # 计算统计信息
        memory_values = [cp["current_memory_mb"] for cp in checkpoints]
        memory_increases = [cp["memory_increase_mb"] for cp in checkpoints]

        return {
            "current_memory_mb": current["current_memory_mb"],
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "total_increase_mb": current["memory_increase_mb"],
            "memory_growth_rate": memory_increases[-1] - memory_increases[0] if len(memory_increases) > 1 else 0,
            "checkpoints_count": len(checkpoints),
            "last_checkpoint": current["timestamp"]
        }


class PerformanceOptimizer:

    """
    策略服务层性能优化器主类
    Strategy Service Layer Performance Optimizer Main Class

    整合所有性能优化功能，提供统一的性能优化接口。
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):

        self.config = config or OptimizationConfig()
        self.cache_manager = StrategyCacheManager(self.config)
        self.executor = ConcurrentStrategyExecutor(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)

        # 性能监控
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_thread = None
        self.running = False

        logger.info("PerformanceOptimizer initialized")

    def start_optimization(self) -> None:
        """启动性能优化"""
        if self.running:
            logger.warning("Performance optimization already running")
            return

        self.running = True

        # 启动后台优化线程
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()

        logger.info("Performance optimization started")

    def stop_optimization(self) -> None:
        """停止性能优化"""
        self.running = False

        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)

        self.executor.shutdown()
        logger.info("Performance optimization stopped")

    def _optimization_loop(self) -> None:
        """优化循环"""
        while self.running:
            try:
                # 收集性能指标
                self._collect_performance_metrics()

                # 内存优化
                if self.config.enable_memory_optimization:
                    if self.memory_optimizer.should_trigger_gc():
                        self.memory_optimizer.optimize_memory_usage()

                # 缓存清理
                if self.config.enable_cache_optimization:
                    self.cache_manager.clear_expired()

                # 等待下一轮
                time.sleep(60)  # 每分钟优化一次

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(30)  # 出错时等待较短时间

    def _collect_performance_metrics(self) -> None:
        """收集性能指标"""
        try:
            # CPU和内存使用
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            # 缓存统计
            cache_stats = self.cache_manager.get_cache_stats()
            executor_stats = self.executor.get_performance_stats()
            memory_stats = self.memory_optimizer.get_memory_stats()

            # 计算综合指标
            total_requests = cache_stats.get("total_requests", 0)
            cache_hit_rate = cache_stats.get("hit_rate_percent", 0)
            throughput = executor_stats.get("total_executions", 0) / \
                max(1, len(self.performance_history) + 1)

            metrics = PerformanceMetrics(
                response_time=executor_stats.get("avg_execution_time", 0) * 1000,  # 转换为ms
                throughput=throughput,
                memory_usage=memory_stats.get("current_memory_mb", 0),
                cpu_usage=cpu_percent,
                cache_hit_rate=cache_hit_rate,
                error_rate=executor_stats.get("error_rate_percent", 0),
                timestamp=datetime.now()
            )

            self.performance_history.append(metrics)

            # 只保留最近1000个指标
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        cache_stats = self.cache_manager.get_cache_stats()
        executor_stats = self.executor.get_performance_stats()
        memory_stats = self.memory_optimizer.get_memory_stats()

        return {
            "is_running": self.running,
            "cache_stats": cache_stats,
            "executor_stats": executor_stats,
            "memory_stats": memory_stats,
            "config": {
                "max_concurrent_strategies": self.config.max_concurrent_strategies,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "memory_limit_mb": self.config.memory_limit_mb,
                "enable_async_processing": self.config.enable_async_processing,
                "enable_memory_optimization": self.config.enable_memory_optimization,
                "enable_cache_optimization": self.config.enable_cache_optimization
            }
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {"message": "No performance data available"}

        recent_metrics = self.performance_history[-10:] if len(
            self.performance_history) > 10 else self.performance_history

        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)

        return {
            "time_range": {
                "start": recent_metrics[0].timestamp.isoformat(),
                "end": recent_metrics[-1].timestamp.isoformat()
            },
            "average_metrics": {
                "response_time_ms": avg_response_time,
                "throughput_ops_per_sec": avg_throughput,
                "memory_usage_mb": avg_memory_usage,
                "cpu_usage_percent": avg_cpu_usage,
                "cache_hit_rate_percent": avg_cache_hit_rate
            },
            "latest_metrics": {
                "response_time_ms": recent_metrics[-1].response_time,
                "throughput_ops_per_sec": recent_metrics[-1].throughput,
                "memory_usage_mb": recent_metrics[-1].memory_usage,
                "cpu_usage_percent": recent_metrics[-1].cpu_usage,
                "cache_hit_rate_percent": recent_metrics[-1].cache_hit_rate,
                "error_rate_percent": recent_metrics[-1].error_rate
            },
            "data_points": len(recent_metrics),
            "performance_score": self._calculate_performance_score(recent_metrics[-1])
        }

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """计算性能评分"""
        # 基于多个指标计算综合评分 (0 - 100)
        score = 100.0

        # 响应时间评分 (权重30%)
        if metrics.response_time > 100:  # >100ms
            score -= 30
        elif metrics.response_time > 50:  # >50ms
            score -= 15

        # 内存使用评分 (权重20%)
        if metrics.memory_usage > 800:  # >800MB
            score -= 20
        elif metrics.memory_usage > 500:  # >500MB
            score -= 10

        # CPU使用评分 (权重20%)
        if metrics.cpu_usage > 90:  # >90%
            score -= 20
        elif metrics.cpu_usage > 70:  # >70%
            score -= 10

        # 缓存命中率评分 (权重15%)
        if metrics.cache_hit_rate < 50:  # <50%
            score -= 15
        elif metrics.cache_hit_rate < 70:  # <70%
            score -= 7.5

        # 错误率评分 (权重15%)
        if metrics.error_rate > 5:  # >5%
            score -= 15
        elif metrics.error_rate > 1:  # >1%
            score -= 7.5

        return max(0.0, min(100.0, score))


# 全局性能优化器实例
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def performance_monitor(func: Callable) -> Callable:
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


def cached_result(ttl_seconds: int = 300):
    """缓存结果装饰器"""

    def decorator(func: Callable) -> Callable:

        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):

            # 生成缓存键
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 检查缓存
            if key in cache:
                cached_time, cached_result = cache[key]
                if (datetime.now() - cached_time).seconds < ttl_seconds:
                    return cached_result
                else:
                    del cache[key]

            # 执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            cache[key] = (datetime.now(), result)

            return result
        return wrapper
    return decorator

