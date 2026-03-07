"""
Memory Optimization Module
内存优化模块

This module provides memory optimization capabilities for quantitative trading systems
此模块为量化交易系统提供内存优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import gc
import psutil
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryPool:

    """
    Memory Pool Class
    内存池类

    Provides object pooling to reduce memory allocation overhead
    提供对象池以减少内存分配开销
    """

    def __init__(self, object_factory: Callable, max_size: int = 1000):
        """
        Initialize memory pool
        初始化内存池

        Args:
            object_factory: Function to create new objects
                           创建新对象的函数
            max_size: Maximum pool size
                     最大池大小
        """
        self.object_factory = object_factory
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.created_count = 0
        self.reused_count = 0
        self.lock = threading.Lock()

    def acquire(self) -> Any:
        """
        Acquire an object from the pool
        从池中获取对象

        Returns:
            Object from pool or newly created
            池中的对象或新创建的对象
        """
        with self.lock:
            if self.pool:
                self.reused_count += 1
                return self.pool.popleft()
            else:
                self.created_count += 1
                return self.object_factory()

    def release(self, obj: Any) -> None:
        """
        Release an object back to the pool
        将对象释放回池中

        Args:
            obj: Object to release
                要释放的对象
        """
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except Exception:
                        pass  # Ignore reset errors
                self.pool.append(obj)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics
        获取池统计信息

        Returns:
            dict: Pool statistics
                  池统计信息
        """
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': self.reused_count / max(self.created_count + self.reused_count, 1) * 100
            }


class WeakReferenceCache:

    """
    Weak Reference Cache Class
    弱引用缓存类

    Cache that automatically cleans up when objects are garbage collected
    当对象被垃圾回收时自动清理的缓存
    """

    def __init__(self):

        self.cache = weakref.WeakValueDictionary()
        self.access_times = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        从缓存中获取项目

        Args:
            key: Cache key
                缓存键

        Returns:
            Cached item or None
            缓存的项目或None
        """
        with self.lock:
            item = self.cache.get(key)
            if item is not None:
                self.access_times[key] = datetime.now()
            return item

    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache
        将项目放入缓存

        Args:
            key: Cache key
            缓存键
            value: Value to cache
                  要缓存的值
        """
        with self.lock:
            self.cache[key] = value
            self.access_times[key] = datetime.now()

    def remove(self, key: str) -> bool:
        """
        Remove item from cache
        从缓存中移除项目

        Args:
            key: Cache key
            缓存键

        Returns:
            bool: True if removed, False otherwise
            如果移除则返回True，否则返回False
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False

    def clear_expired(self, max_age: timedelta) -> int:
        """
        Clear expired items from cache
        从缓存中清除过期项目

        Args:
            max_age: Maximum age for cache items
                    缓存项目的最大年龄

        Returns:
            int: Number of items cleared
                清除的项目数量
        """
        with self.lock:
            now = datetime.now()
            expired_keys = [
                key for key, access_time in self.access_times.items()
                if (now - access_time) > max_age
            ]

            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        获取缓存统计信息

        Returns:
            dict: Cache statistics
                  缓存统计信息
        """
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'total_access_times': len(self.access_times)
            }


class MemoryMonitor:

    """
    Memory Monitor Class
    内存监控类

    Monitors memory usage and provides optimization recommendations
    监控内存使用情况并提供优化建议
    """

    def __init__(self, monitoring_interval: float = 60.0):
        """
        Initialize memory monitor
        初始化内存监控器

        Args:
            monitoring_interval: Monitoring interval in seconds
                                监控间隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.memory_history = deque(maxlen=100)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alerts = []
        self.thresholds = {
            'memory_percent': 80.0,
            'memory_mb': 1024,  # 1GB
            'gc_threshold': 1000
        }

        logger.info("Memory monitor initialized")

    def start_monitoring(self) -> bool:
        """
        Start memory monitoring
        开始内存监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_monitoring:
            logger.warning("Memory monitoring already running")
            return False

        try:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Memory monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start memory monitoring: {str(e)}")
            self.is_monitoring = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop memory monitoring
        停止内存监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_monitoring:
            logger.warning("Memory monitoring not running")
            return False

        try:
            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Memory monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop memory monitoring: {str(e)}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics
        获取当前内存统计信息

        Returns:
            dict: Memory statistics
                  内存统计信息
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'timestamp': datetime.now(),
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'memory_percent': process.memory_percent(),
                'system_memory': self._get_system_memory_stats(),
                'gc_stats': self._get_gc_stats()
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {'error': str(e)}

    def _get_system_memory_stats(self) -> Dict[str, Any]:
        """Get system memory statistics"""
        try:
            system_memory = psutil.virtual_memory()
            return {
                'total_mb': system_memory.total / (1024 * 1024),
                'available_mb': system_memory.available / (1024 * 1024),
                'used_mb': system_memory.used / (1024 * 1024),
                'free_mb': system_memory.free / (1024 * 1024),
                'usage_percent': system_memory.percent
            }
        except Exception:
            return {}

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        try:
            gc_stats = {}
            for i, stats in enumerate(gc.get_stats()):
                gc_stats[f'gen_{i}'] = {
                    'collections': stats['collections'],
                    'collected': stats['collected'],
                    'uncollectable': stats['uncollectable']
                }
            return gc_stats
        except Exception:
            return {}

    def check_memory_thresholds(self, memory_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check memory usage against thresholds
        检查内存使用是否超过阈值

        Args:
            memory_stats: Memory statistics
                         内存统计信息

        Returns:
            list: List of alerts generated
                  生成的警报列表
        """
        alerts = []

        try:
            memory_percent = memory_stats.get('memory_percent', 0)
            rss_mb = memory_stats.get('rss_mb', 0)

            if memory_percent > self.thresholds['memory_percent']:
                alerts.append({
                    'type': 'memory_percent',
                    'severity': 'warning',
                    'value': memory_percent,
                    'threshold': self.thresholds['memory_percent'],
                    'message': f"Memory usage {memory_percent:.1f}% exceeds threshold {self.thresholds['memory_percent']}%",
                    'timestamp': datetime.now()
                })

            if rss_mb > self.thresholds['memory_mb']:
                alerts.append({
                    'type': 'memory_mb',
                    'severity': 'warning',
                    'value': rss_mb,
                    'threshold': self.thresholds['memory_mb'],
                    'message': f"RSS memory {rss_mb:.1f}MB exceeds threshold {self.thresholds['memory_mb']}MB",
                    'timestamp': datetime.now()
                })

        except Exception as e:
            logger.error(f"Failed to check memory thresholds: {str(e)}")

        return alerts

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info("Memory monitoring loop started")

        while self.is_monitoring:
            try:
                # Get memory statistics
                memory_stats = self.get_memory_stats()
                self.memory_history.append(memory_stats)

                # Check thresholds
                alerts = self.check_memory_thresholds(memory_stats)
                self.alerts.extend(alerts)

                # Log alerts
                for alert in alerts:
                    logger.warning(f"Memory alert: {alert['message']}")

                # Clean up old alerts (keep last 50)
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]

                # Sleep before next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Memory monitoring loop error: {str(e)}")
                time.sleep(self.monitoring_interval)

        logger.info("Memory monitoring loop stopped")

    def get_memory_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get memory usage history
        获取内存使用历史

        Args:
            limit: Maximum number of history entries to return
                  返回的最大历史条目数

        Returns:
            list: Memory history
                  内存历史
        """
        history = list(self.memory_history)
        if limit:
            history = history[-limit:]
        return history

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection
        强制垃圾回收

        Returns:
            dict: Garbage collection results
                  垃圾回收结果
        """
        try:
            start_time = time.time()
            collected = gc.collect()
            end_time = time.time()

            result = {
                'objects_collected': collected,
                'execution_time': end_time - start_time,
                'timestamp': datetime.now()
            }

            logger.info(f"Garbage collection completed: {collected} objects collected")
            return result

        except Exception as e:
            logger.error(f"Failed to perform garbage collection: {str(e)}")
            return {'error': str(e)}


class MemoryOptimizer:

    """
    Memory Optimizer Class
    内存优化器类

    Provides comprehensive memory optimization capabilities
    提供全面的内存优化能力
    """

    def __init__(self):

        self.monitor = MemoryMonitor()
        self.pools = {}
        self.caches = {}
        self.weak_cache = WeakReferenceCache()
        self.optimization_stats = {
            'objects_pooled': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_reclaimed': 0
        }

        logger.info("Memory optimizer initialized")

    def create_object_pool(self, pool_name: str, object_factory: Callable, max_size: int = 100) -> str:
        """
        Create an object pool
        创建对象池

        Args:
            pool_name: Name of the pool
                      池的名称
            object_factory: Function to create objects
                           创建对象的函数
            max_size: Maximum pool size
                     最大池大小

        Returns:
            str: Pool name
                池名称
        """
        self.pools[pool_name] = MemoryPool(object_factory, max_size)
        logger.info(f"Created object pool: {pool_name} (max_size={max_size})")
        return pool_name

    def get_from_pool(self, pool_name: str) -> Any:
        """
        Get an object from pool
        从池中获取对象

        Args:
            pool_name: Name of the pool
                      池的名称

        Returns:
            Object from pool
            池中的对象
        """
        if pool_name not in self.pools:
            raise ValueError(f"Pool {pool_name} does not exist")

        obj = self.pools[pool_name].acquire()
        self.optimization_stats['objects_pooled'] += 1
        return obj

    def return_to_pool(self, pool_name: str, obj: Any) -> None:
        """
        Return an object to pool
        将对象返回到池中

        Args:
            pool_name: Name of the pool
                      池的名称
            obj: Object to return
                要返回的对象
        """
        if pool_name in self.pools:
            self.pools[pool_name].release(obj)

    def cache_object(self, key: str, obj: Any, use_weak_refs: bool = True) -> None:
        """
        Cache an object
        缓存对象

        Args:
            key: Cache key
                缓存键
            obj: Object to cache
                要缓存的对象
            use_weak_refs: Whether to use weak references
                          是否使用弱引用
        """
        if use_weak_refs:
            self.weak_cache.put(key, obj)
        else:
            # Use regular dictionary for strong references
            if not hasattr(self, 'strong_cache'):
                self.strong_cache = {}
            self.strong_cache[key] = obj

    def get_cached_object(self, key: str, use_weak_refs: bool = True) -> Optional[Any]:
        """
        Get cached object
        获取缓存的对象

        Args:
            key: Cache key
                缓存键
            use_weak_refs: Whether to use weak references
                          是否使用弱引用

        Returns:
            Cached object or None
            缓存的对象或None
        """
        if use_weak_refs:
            obj = self.weak_cache.get(key)
        else:
            obj = getattr(self, 'strong_cache', {}).get(key)

        if obj is not None:
            self.optimization_stats['cache_hits'] += 1
        else:
            self.optimization_stats['cache_misses'] += 1

        return obj

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Perform comprehensive memory optimization
        执行全面的内存优化

        Returns:
            dict: Optimization results
                  优化结果
        """
        try:
            results = {
                'timestamp': datetime.now(),
                'actions_taken': [],
                'memory_before': self.monitor.get_memory_stats(),
                'memory_after': None,
                'objects_collected': 0,
                'cache_cleaned': 0
            }

            # Force garbage collection
            gc_result = self.monitor.force_garbage_collection()
            results['objects_collected'] = gc_result.get('objects_collected', 0)
            results['actions_taken'].append('garbage_collection')

            # Clean weak reference cache
            expired_count = self.weak_cache.clear_expired(timedelta(hours=1))
            results['cache_cleaned'] = expired_count
            if expired_count > 0:
                results['actions_taken'].append('cache_cleanup')

            # Clean memory history if too large
            if len(self.monitor.memory_history) > 50:
                # Keep only last 50 entries
                self.monitor.memory_history = deque(
                    list(self.monitor.memory_history)[-50:],
                    maxlen=100
                )
                results['actions_taken'].append('history_cleanup')

            # Get memory stats after optimization
            results['memory_after'] = self.monitor.get_memory_stats()

            # Calculate memory reclaimed
            memory_before = results['memory_before'].get('rss_mb', 0)
            memory_after = results['memory_after'].get('rss_mb', 0)
            results['memory_reclaimed'] = max(0, memory_before - memory_after)

            self.optimization_stats['memory_reclaimed'] += results['memory_reclaimed']

            logger.info(f"Memory optimization completed: {results['objects_collected']} objects collected, "
                        f"{results['memory_reclaimed']:.1f}MB reclaimed")

            return results

        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        获取优化统计信息

        Returns:
            dict: Optimization statistics
                  优化统计信息
        """
        stats = self.optimization_stats.copy()

        # Add pool statistics
        pool_stats = {}
        for pool_name, pool in self.pools.items():
            pool_stats[pool_name] = pool.get_stats()
        stats['pool_stats'] = pool_stats

        # Add cache statistics
        stats['weak_cache_stats'] = self.weak_cache.get_stats()

        # Add memory monitor statistics
        stats['memory_stats'] = self.monitor.get_memory_stats()
        stats['memory_history_count'] = len(self.monitor.memory_history)

        return stats

    def start_monitoring(self) -> bool:
        """
        Start memory monitoring
        开始内存监控

        Returns:
            bool: True if started successfully
                  启动成功返回True
        """
        return self.monitor.start_monitoring()

    def stop_monitoring(self) -> bool:
        """
        Stop memory monitoring
        停止内存监控

        Returns:
            bool: True if stopped successfully
                  停止成功返回True
        """
        return self.monitor.stop_monitoring()

    def get_memory_alerts(self) -> List[Dict[str, Any]]:
        """
        Get memory alerts
        获取内存警报

        Returns:
            list: List of memory alerts
                  内存警报列表
        """
        return self.monitor.alerts.copy()


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

__all__ = [
    'MemoryPool',
    'WeakReferenceCache',
    'MemoryMonitor',
    'MemoryOptimizer',
    'memory_optimizer'
]
