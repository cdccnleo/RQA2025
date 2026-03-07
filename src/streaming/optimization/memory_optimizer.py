"""
Memory Optimizer Module
内存优化器模块

This module provides memory optimization capabilities for streaming operations
此模块为流操作提供内存优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import threading
import time
import gc
import psutil
from collections import deque

logger = logging.getLogger(__name__)


class MemoryOptimizer:

    """
    Streaming Memory Optimizer
    流内存优化器

    Optimizes memory usage in streaming data processing operations
    优化流数据处理操作中的内存使用
    """

    def __init__(self, target_memory_percent: float = 75.0,


                 cleanup_interval: float = 30.0):
        """
        Initialize the memory optimizer
        初始化内存优化器

        Args:
            target_memory_percent: Target memory usage percentage
                                 目标内存使用百分比
            cleanup_interval: Interval between memory cleanups (seconds)
                             内存清理间隔（秒）
        """
        self.target_memory_percent = target_memory_percent
        self.cleanup_interval = cleanup_interval

        # Memory monitoring
        self.memory_history = deque(maxlen=100)
        self.is_running = False
        self.monitoring_thread = None

        # Optimization strategies
        self.optimization_strategies = []
        self.register_default_strategies()

        # Memory pools for object reuse
        self.object_pools = {}
        self.pool_sizes = {}

        # Weak references for garbage tracking
        self.weak_refs = []

        logger.info(
            f"Memory optimizer initialized with target {target_memory_percent}% memory usage")

    def start_optimization(self) -> bool:
        """
        Start memory optimization monitoring
        开始内存优化监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Memory optimizer is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Memory optimization monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start memory optimization: {str(e)}")
            self.is_running = False
            return False

    def stop_optimization(self) -> bool:
        """
        Stop memory optimization monitoring
        停止内存优化监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Memory optimizer is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Memory optimization monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop memory optimization: {str(e)}")
            return False

    def register_default_strategies(self) -> None:
        """
        Register default memory optimization strategies
        注册默认的内存优化策略
        """
        self.optimization_strategies = [
            self._garbage_collection_strategy,
            self._object_pool_strategy,
            self._buffer_cleanup_strategy,
            self._cache_cleanup_strategy
        ]

    def add_optimization_strategy(self, strategy: Callable) -> None:
        """
        Add a custom optimization strategy
        添加自定义优化策略

        Args:
            strategy: Optimization strategy function
                     优化策略函数
        """
        self.optimization_strategies.append(strategy)
        logger.info("Added custom memory optimization strategy")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        获取当前内存使用统计信息

        Returns:
            dict: Memory usage data
                  内存使用数据
        """
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()

            system_percent = memory.percent
            process_percent = process.memory_percent()

            memory_info = {
                'timestamp': datetime.now(),
                'system': {
                    'total_gb': memory.total / (1024 ** 3),
                    'used_gb': memory.used / (1024 ** 3),
                    'free_gb': memory.free / (1024 ** 3),
                    'available_gb': memory.available / (1024 ** 3),
                    'percentage': system_percent
                },
                'process': {
                    'rss_gb': process.memory_info().rss / (1024 ** 3),
                    'vms_gb': process.memory_info().vms / (1024 ** 3),
                    'percentage': process_percent
                },
                # 保持向后兼容的顶层字段（旧版测试所需）
                'percent': system_percent,
                'used_mb': memory.used / (1024 ** 2),
                'available_mb': memory.available / (1024 ** 2),
                'process_percent': process_percent,
                'process_used_mb': process.memory_info().rss / (1024 ** 2),
                'process_vms_mb': process.memory_info().vms / (1024 ** 2)
            }

            # Store in history
            self.memory_history.append(memory_info)

            return memory_info

        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            return {}

    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform memory optimization
        执行内存优化

        Returns:
            dict: Optimization results
                  优化结果
        """
        results = {
            'timestamp': datetime.now(),
            'strategies_applied': [],
            'memory_before': self.get_memory_usage(),
            'optimizations': []
        }

        try:
            # Apply all optimization strategies
            for strategy in self.optimization_strategies:
                try:
                    strategy_result = strategy()
                    if strategy_result:
                        results['optimizations'].append(strategy_result)
                        results['strategies_applied'].append(strategy.__name__)
                except Exception as e:
                    logger.error(f"Strategy {strategy.__name__} failed: {str(e)}")

            # Get memory after optimization
            results['memory_after'] = self.get_memory_usage()

            # Calculate improvement
            before_percent = results['memory_before'].get('system', {}).get('percentage', 0)
            after_percent = results['memory_after'].get('system', {}).get('percentage', 0)
            results['improvement_percent'] = before_percent - after_percent

            logger.info(
                f"Memory optimization completed: {len(results['optimizations'])} strategies applied")

        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
            results['error'] = str(e)

        return results

    def create_object_pool(self, object_type: str, factory: Callable,


                           initial_size: int = 10, max_size: int = 100) -> None:
        """
        Create an object pool for memory reuse
        创建对象池以重用内存

        Args:
            object_type: Type of objects in the pool
                        对象池中的对象类型
            factory: Factory function to create new objects
                    创建新对象的工厂函数
            initial_size: Initial pool size
                         初始池大小
            max_size: Maximum pool size
                     最大池大小
        """
        if object_type in self.object_pools:
            logger.warning(f"Object pool for {object_type} already exists")
            return

        pool = deque()
        for _ in range(initial_size):
            try:
                obj = factory()
                pool.append(obj)
            except Exception as e:
                logger.error(f"Failed to create object for pool {object_type}: {str(e)}")

        self.object_pools[object_type] = pool
        self.pool_sizes[object_type] = {'current': len(pool), 'max': max_size}

        logger.info(f"Created object pool for {object_type} with {len(pool)} objects")

    def get_from_pool(self, object_type: str) -> Optional[Any]:
        """
        Get an object from the pool
        从池中获取对象

        Args:
            object_type: Type of object to get
                        要获取的对象类型

        Returns:
            Object from pool or None if pool is empty
            池中的对象，如果池为空则返回None
        """
        if object_type not in self.object_pools:
            return None

        pool = self.object_pools[object_type]
        if pool:
            obj = pool.popleft()
            self.pool_sizes[object_type]['current'] = len(pool)
            return obj

        return None

    def return_to_pool(self, object_type: str, obj: Any) -> bool:
        """
        Return an object to the pool
        将对象返回到池中

        Args:
            object_type: Type of object
                        对象类型
            obj: Object to return
                要返回的对象

        Returns:
            bool: True if object was returned, False otherwise
                  如果对象被返回则返回True，否则返回False
        """
        if object_type not in self.object_pools:
            return False

        pool = self.object_pools[object_type]
        max_size = self.pool_sizes[object_type]['max']

        if len(pool) < max_size:
            pool.append(obj)
            self.pool_sizes[object_type]['current'] = len(pool)
            return True

        return False

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop for memory optimization
        内存优化的主要监控循环
        """
        logger.info("Memory optimization monitoring loop started")

        while self.is_running:
            try:
                # Check memory usage
                memory_info = self.get_memory_usage()
                current_percent = memory_info.get('system', {}).get('percentage', 0)

                # Trigger optimization if memory usage is high
                if current_percent > self.target_memory_percent:
                    logger.warning(
                        f"Memory usage {current_percent:.1f}% exceeds target {self.target_memory_percent:.1f}%")
                    optimization_result = self.optimize_memory()
                    if optimization_result.get('improvement_percent', 0) > 0:
                        logger.info(
                            f"Memory optimization successful: {optimization_result['improvement_percent']:.1f}% improvement")

                # Wait before next check
                time.sleep(self.cleanup_interval)

            except Exception as e:
                logger.error(f"Memory monitoring loop error: {str(e)}")
                time.sleep(self.cleanup_interval)

        logger.info("Memory optimization monitoring loop stopped")

    def _garbage_collection_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Garbage collection optimization strategy
        垃圾回收优化策略

        Returns:
            dict: Strategy execution results or None
                  策略执行结果或None
        """
        try:
            before_count = len(gc.get_objects())
            collected = gc.collect()
            after_count = len(gc.get_objects())

            return {
                'strategy': 'garbage_collection',
                'objects_before': before_count,
                'objects_collected': collected,
                'objects_after': after_count
            }

        except Exception as e:
            logger.error(f"Garbage collection strategy failed: {str(e)}")
            return None

    def _object_pool_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Object pool optimization strategy
        对象池优化策略

        Returns:
            dict: Strategy execution results or None
                  策略执行结果或None
        """
        try:
            total_pooled = sum(len(pool) for pool in self.object_pools.values())
            return {
                'strategy': 'object_pool_maintenance',
                'total_pooled_objects': total_pooled,
                'pools_count': len(self.object_pools)
            }

        except Exception as e:
            logger.error(f"Object pool strategy failed: {str(e)}")
            return None

    def _buffer_cleanup_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Buffer cleanup optimization strategy
        缓冲区清理优化策略

        Returns:
            dict: Strategy execution results or None
                  策略执行结果或None
        """
        try:
            # Clear any large data structures if they exist
            # This is a placeholder for actual buffer cleanup logic
            return {
                'strategy': 'buffer_cleanup',
                'status': 'completed'
            }

        except Exception as e:
            logger.error(f"Buffer cleanup strategy failed: {str(e)}")
            return None

    def _cache_cleanup_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Cache cleanup optimization strategy
        缓存清理优化策略

        Returns:
            dict: Strategy execution results or None
                  策略执行结果或None
        """
        try:
            # Clear weak references that have been garbage collected
            alive_refs = [ref for ref in self.weak_refs if ref() is not None]
            cleaned_count = len(self.weak_refs) - len(alive_refs)
            self.weak_refs = alive_refs

            return {
                'strategy': 'cache_cleanup',
                'weak_refs_cleaned': cleaned_count,
                'weak_refs_remaining': len(alive_refs)
            }

        except Exception as e:
            logger.error(f"Cache cleanup strategy failed: {str(e)}")
            return None

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get memory optimization statistics
        获取内存优化统计信息

        Returns:
            dict: Optimization statistics
                  优化统计信息
        """
        return {
            'is_running': self.is_running,
            'target_memory_percent': self.target_memory_percent,
            'cleanup_interval': self.cleanup_interval,
            'current_memory': self.get_memory_usage(),
            'object_pools': {
                pool_type: {
                    'size': len(pool),
                    'max_size': self.pool_sizes.get(pool_type, {}).get('max', 0)
                }
                for pool_type, pool in self.object_pools.items()
            },
            'memory_history_size': len(self.memory_history),
            'optimization_strategies_count': len(self.optimization_strategies)
        }


# Global memory optimizer instance - only created when needed
# 全局内存优化器实例 - 仅在需要时创建
_memory_optimizer = None


def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器实例"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


# For backward compatibility
memory_optimizer = None

__all__ = ['MemoryOptimizer', 'memory_optimizer']
