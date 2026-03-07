#!/usr/bin/env python3
"""
内存优化器

实现大数据量风险计算的内存管理优化，包括内存池、垃圾回收、内存监控等
"""

import logging
import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Callable, Iterator, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import tracemalloc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入内存分析器
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
logger.warning("memory_profiler不可用，内存分析功能将被限制")

try:
    import objgraph
    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False
    logger.warning("objgraph不可用，对象图分析将被禁用")


class MemoryPoolType(Enum):

    "内存池类型"
    NUMPY_ARRAYS = "numpy_arrays",      # NumPy数组器
    PANDAS_DATAFRAMES = "pandas_dfs",   # Pandas DataFrame器
    COMPUTATION_RESULTS = "results",    # 计算结果器
    CACHE_DATA = "cache_data"         # 缓存数据器


class MemoryWarningLevel(Enum):

    "内存告警级别"
    LOW = "low",          # 低内存使器
    MEDIUM = "medium",    # 中等内存使用
    HIGH = "high",        # 高内存使器
    CRITICAL = "critical"  # 严重内存使用


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory: float  # 总内存(MB)
    available_memory: float  # 可用内存(MB)
    used_memory: float  # 已用内存(MB)
    memory_percent: float  # 内存使用百分比
    process_memory: float  # 进程内存使用(MB)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def memory_pressure(self) -> MemoryWarningLevel:
        """内存压力级别"""
        if self.memory_percent > 90:
            return MemoryWarningLevel.CRITICAL
        elif self.memory_percent > 75:
            return MemoryWarningLevel.HIGH
        elif self.memory_percent > 50:
            return MemoryWarningLevel.MEDIUM
        else:
            return MemoryWarningLevel.LOW


@dataclass
class MemoryLeakReport:
    """内存泄漏报告"""
    leak_detected: bool
    leaked_objects: List[str]
    memory_growth: float
    top_consumers: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryPool:
    """内存池"""
    pool_type: MemoryPoolType
    max_size: int  # 最大对象数器
    max_memory_mb: float  # 最大内存使器MB)
    objects: Dict[str, Any] = field(default_factory=dict)
    object_sizes: Dict[str, float] = field(default_factory=dict)
    access_times: Dict[str, datetime] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def current_size(self) -> int:
        """当前对象数量"""
        return len(self.objects)

    @property
    def current_memory_mb(self) -> float:
        """当前内存使用(MB)"""
        return sum(self.object_sizes.values())

    @property
    def utilization_percent(self) -> float:
        """利用率百分比"""
        if self.max_memory_mb > 0:
            return (self.current_memory_mb / self.max_memory_mb) * 100
        return 0.0


@dataclass
class MemoryMonitor:
    """内存监控器"""

    def __init__(self, check_interval: int = 30, enable_tracemalloc: bool = True):
        self.check_interval = check_interval
        self.enable_tracemalloc = enable_tracemalloc

        # 内存统计历史
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 1000

        # 告警回调
        self.warning_callbacks: List[Callable] = []

        # 监控标志
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # tracemalloc
        if self.enable_tracemalloc:
            tracemalloc.start()
        self.initial_snapshot = tracemalloc.take_snapshot()

    logger.info("内存监控器初始化完成")

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("内存监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("内存监控已停止")

    def get_current_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            stats = MemoryStats(
                total_memory=memory_info.total / 1024 / 1024,  # MB
                available_memory=memory_info.available / 1024 / 1024,
                used_memory=memory_info.available / 1024 / 1024,
                memory_percent=memory_info.percent,
                process_memory=process.memory_info().rss / 1024 / 1024
            )
            # 添加到历史记录
            self.memory_history.append(stats)
            if len(self.memory_history) > self.max_history_size:
                self.memory_history.pop(0)
            return stats
        except Exception as e:
            logger.error(f"获取内存统计失败: {e}")
            return MemoryStats(0, 0, 0, 0, 0)

    def register_warning_callback(self, callback: Callable):
        """注册告警回调"""
        self.warning_callbacks.append(callback)

    def _monitor_loop(self):
        """监控循环"""
        last_warning_time = None
        warning_cooldown = 300  # 5分钟冷却时间

        while self.monitoring:
            try:
                current_stats = self.get_current_stats()

                # 检查内存压力
                if current_stats.memory_pressure in [MemoryWarningLevel.HIGH, MemoryWarningLevel.CRITICAL]:
                    current_time = datetime.now()

                    # 检查冷却时间
                    if (last_warning_time is None
                        or (current_time - last_warning_time).seconds >= warning_cooldown):

                        self._trigger_warning(current_stats)
                        last_warning_time = current_time

                # 定期清理
                if len(self.memory_history) % 10 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"内存监控循环异常: {e}")

            # 使用更短的睡眠时间，以便更快响应停止信号
            time.sleep(min(self.check_interval, 1.0))

    def _trigger_warning(self, stats: MemoryStats):
        "触发告警"
        warning_msg = {
            'level': stats.memory_pressure.value,
            'message': f'内存使用率过高 {stats.memory_percent:.1f}%',
            'stats': {
                'total_memory': stats.total_memory,
                'used_memory': stats.used_memory,
                'available_memory': stats.available_memory,
                'process_memory': stats.process_memory
            },
            'timestamp': stats.timestamp.isoformat()
        }

        logger.warning(f"内存告警: {warning_msg}")

        for callback in self.warning_callbacks:
            try:
                callback(warning_msg)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")


    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """获取内存趋势"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_stats = [s for s in self.memory_history if s.timestamp >= cutoff_time]
            if not recent_stats:
                return {'error': '没有足够的历史数据'}
            memory_usage = [s.memory_percent for s in recent_stats]
            process_memory = [s.process_memory for s in recent_stats]
            return {
                'time_range_hours': hours,
                'data_points': len(recent_stats),
                'memory_usage': {
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'avg': sum(memory_usage) / len(memory_usage),
                    'trend': 'increasing' if memory_usage[-1] > memory_usage[0] else 'decreasing'
                },
                'process_memory': {
                    'min': min(process_memory),
                    'max': max(process_memory),
                    'avg': sum(process_memory) / len(process_memory)
                },
                'timestamps': [s.timestamp.isoformat() for s in recent_stats[-10:]]  # 最近10个时间点
            }
        except Exception as e:
            logger.error(f"获取内存趋势失败: {e}")
            return {'error': str(e)}

    def detect_memory_leak(self, baseline_hours: int = 1) -> MemoryLeakReport:
        """检测内存泄漏"""
        try:
            current_stats = self.get_current_stats()
            # 获取基线数据
            cutoff_time = datetime.now() - timedelta(hours=baseline_hours)
            baseline_stats = [s for s in self.memory_history if s.timestamp >= cutoff_time]
            if not baseline_stats:
                return MemoryLeakReport(leak_detected=False, leaked_objects=[], memory_growth=0, top_consumers=[])

            # 计算内存增长
            baseline_memory = sum(s.process_memory for s in baseline_stats) / len(baseline_stats)
            memory_growth = current_stats.process_memory - baseline_memory

            # 检测泄漏
            leak_detected = memory_growth > (baseline_memory * 0.1)  # 增长超过10%

            # 获取对象统计（如果启用tracemalloc）
            leaked_objects = []
            top_consumers = []
            if self.enable_tracemalloc:
                try:
                    current_snapshot = tracemalloc.take_snapshot()
                    stats = current_snapshot.compare_to(self.initial_snapshot, 'lineno')
                    # 获取前10个内存消费者
                    top_consumers = [
                        {
                            'file': stat.traceback[0].filename if stat.traceback else 'unknown',
                            'line': stat.traceback[0].lineno if stat.traceback else 0,
                            'size': stat.size / 1024 / 1024,  # MB
                            'count': stat.count
                        }
                        for stat in stats[:10]
                    ]
                    # 检测可能的泄漏对象
                    for stat in stats:
                        if stat.size > 10 * 1024 * 1024:  # 超过10MB
                            leaked_objects.append(f"{stat.traceback[0].filename}:{stat.traceback[0].lineno}")
                except Exception as e:
                    logger.error(f"tracemalloc分析失败: {e}")

            return MemoryLeakReport(leak_detected=leak_detected, leaked_objects=leaked_objects, memory_growth=memory_growth, top_consumers=top_consumers)
        except Exception as e:
            logger.error(f"内存泄漏检测失败: {e}")
            return MemoryLeakReport(leak_detected=False, leaked_objects=[], memory_growth=0, top_consumers=[], timestamp=datetime.now())


class MemoryPoolManager:
    """内存池管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pools: Dict[MemoryPoolType, MemoryPool] = {}
        self.lock = threading.RLock()

        # 初始化内存池
        self._initialize_pools()

        # 清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        logger.info("内存池管理器初始化完成")


    def _initialize_pools(self):
        """初始化内存池"""
        pool_configs = {
            MemoryPoolType.NUMPY_ARRAYS: {
                'max_size': self.config.get('numpy_pool_max_size', 50),
                'max_memory_mb': self.config.get('numpy_pool_max_memory', 1024)
            },
            MemoryPoolType.PANDAS_DATAFRAMES: {
                'max_size': self.config.get('pandas_pool_max_size', 20),
                'max_memory_mb': self.config.get('pandas_pool_max_memory', 2048)
            },
            MemoryPoolType.COMPUTATION_RESULTS: {
                'max_size': self.config.get('results_pool_max_size', 100),
                'max_memory_mb': self.config.get('results_pool_max_memory', 512)
            },
            MemoryPoolType.CACHE_DATA: {
                'max_size': self.config.get('cache_pool_max_size', 200),
                'max_memory_mb': self.config.get('cache_pool_max_memory', 1024)
            }
        }

        for pool_type, pool_config in pool_configs.items():
            self.pools[pool_type] = MemoryPool(
                pool_type=pool_type,
                max_size=pool_config['max_size'],
                max_memory_mb=pool_config['max_memory_mb']
            )


    def get_object(self, pool_type: MemoryPoolType, key: str) -> Optional[Any]:
        """从内存池获取对象"""
        with self.lock:
            pool = self.pools.get(pool_type)
            if not pool:
                return None
            if key in pool.objects:
                pool.access_times[key] = datetime.now()
                return pool.objects[key]
            return None

    def put_object(self, pool_type: MemoryPoolType, key: str, obj: Any, estimate_size: bool = True) -> bool:
        """将对象放入内存池"""
        with self.lock:
            pool = self.pools.get(pool_type)
            if not pool:
                return False
            try:
                # 估算对象大小
                if estimate_size:
                    size_mb = self._estimate_object_size(obj)
                else:
                    size_mb = 0
                # 检查池容量
                if pool.current_size >= pool.max_size:
                    self._evict_objects(pool)
                # 检查内存限制
                if pool.current_memory_mb + size_mb > pool.max_memory_mb:
                    self._evict_objects(pool)
                # 存储对象
                pool.objects[key] = obj
                pool.object_sizes[key] = size_mb
                pool.access_times[key] = datetime.now()
                return True
            except Exception as e:
                logger.error(f"放入内存池失败: {e}")
                return False

    def remove_object(self, pool_type: MemoryPoolType, key: str) -> bool:
        """从内存池移除对象"""
        with self.lock:
            pool = self.pools.get(pool_type)
            if not pool or key not in pool.objects:
                return False
            del pool.objects[key]
            if key in pool.object_sizes:
                del pool.object_sizes[key]
            if key in pool.access_times:
                del pool.access_times[key]
            return True

    def clear_pool(self, pool_type: MemoryPoolType) -> bool:
        """清空内存池"""
        with self.lock:
            pool = self.pools.get(pool_type)
            if not pool:
                return False
            pool.objects.clear()
            pool.object_sizes.clear()
            pool.access_times.clear()
            # 强制垃圾回收
            gc.collect()
            return True

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取内存池统计"""
        with self.lock:
            stats = {}
            for pool_type, pool in self.pools.items():
                stats[pool_type.value] = {
                    'current_size': pool.current_size,
                    'max_size': pool.max_size,
                    'current_memory_mb': pool.current_memory_mb,
                    'max_memory_mb': pool.max_memory_mb,
                    'utilization_percent': pool.utilization_percent,
                    'object_count': len(pool.objects)
                }

            return stats


    def _estimate_object_size(self, obj: Any) -> float:
        """估算对象大小(MB)"""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes / 1024 / 1024
            elif isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(obj, dict):
                total_size = 0
                for key, value in obj.items():
                    total_size += len(str(key).encode()) + self._estimate_object_size(value)
                return total_size / 1024 / 1024
            elif isinstance(obj, list):
                total_size = sum(self._estimate_object_size(item) for item in obj)
                return total_size
            else:
                # 使用sys.getsizeof估算
                import sys
                return sys.getsizeof(obj) / 1024 / 1024

        except Exception:
            return 0.1  # 默认100KB


    def _evict_objects(self, pool: MemoryPool):
        """驱逐对象"""
        try:
            # 按访问时间排序，最久未使用的先驱逐
            sorted_keys = sorted(
                pool.access_times.keys(),
                key=lambda k: pool.access_times[k]
            )

            # 驱逐最旧的对象直到有足够空间
            evicted_count = 0
            target_memory = pool.max_memory_mb * 0.8  # 目标使用率80%

            for key in sorted_keys:
                if pool.current_memory_mb <= target_memory:
                    break

                if key in pool.objects:
                    del pool.objects[key]
                if key in pool.object_sizes:
                    pool.current_memory_mb -= pool.object_sizes[key]
                    del pool.object_sizes[key]
                if key in pool.access_times:
                    del pool.access_times[key]
                evicted_count += 1

            if evicted_count > 0:
                logger.info(f"从内存池驱逐了 {evicted_count} 个对象")

        except Exception as e:
            logger.error(f"对象驱逐失败: {e}")


    def _cleanup_worker(self):
        """清理工作线程"""
        while True:
            try:
                time.sleep(300)  # 5分钟清理一次

                with self.lock:
                    for pool in self.pools.values():
                        current_time = datetime.now()

                        # 清理过期对象（超过1小时未访问）
                        expired_keys = []
                        for key, access_time in pool.access_times.items():
                            if (current_time - access_time).seconds > 3600:
                                expired_keys.append(key)

                        for key in expired_keys:
                            self.remove_object(pool.pool_type, key)

                        if expired_keys:
                            logger.info(f"清理了{len(expired_keys)} 个过期对象")

            except Exception as e:
                logger.error(f"内存池清理异常: {e}")


class DataBatchProcessor:

    "数据批处理器"


    def __init__(self, batch_size: int = 1000, max_memory_mb: float = 512):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb


    def process_large_dataset(self, data: Any, processor: Callable, batch_processor: Optional[Callable] = None) -> Any:
        """
        处理大数据集
        Args:
            data: 输入数据
            processor: 单批次处理器
            batch_processor: 批次结果处理器
        Returns:
            处理结果
        """
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe(data, processor, batch_processor)
        elif isinstance(data, np.ndarray):
            return self._process_array(data, processor, batch_processor)
        elif isinstance(data, list):
            return self._process_list(data, processor, batch_processor)
        else:
            return processor(data)


    def _process_dataframe(self, df: pd.DataFrame, processor: Callable, batch_processor: Optional[Callable]) -> Any:
        """处理DataFrame"""
        total_rows = len(df)
        results = []
        for start_idx in range(0, total_rows, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            # 检查内存使用
            if self._check_memory_usage():
                gc.collect()

            batch_result = processor(batch_df)
            results.append(batch_result)

        # 合并结果
        if batch_processor:
            return batch_processor(results)
        else:
            return self._merge_results(results)


    def _process_array(self, array: np.ndarray, processor: Callable, batch_processor: Optional[Callable]) -> Any:
        """处理NumPy数组"""
        total_size = len(array)
        results = []
        for start_idx in range(0, total_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_size)
            batch_array = array[start_idx:end_idx]

            # 检查内存使用
            if self._check_memory_usage():
                gc.collect()

            batch_result = processor(batch_array)
            results.append(batch_result)

        # 合并结果
        if batch_processor:
            return batch_processor(results)
        else:
            return self._merge_results(results)


    def _process_list(self, data_list: list, processor: Callable, batch_processor: Optional[Callable]) -> Any:
        """处理列表"""
        total_size = len(data_list)
        results = []
        for start_idx in range(0, total_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_size)
            batch_list = data_list[start_idx:end_idx]

            # 检查内存使用
            if self._check_memory_usage():
                gc.collect()

            batch_result = processor(batch_list)
            results.append(batch_result)

        # 合并结果
        if batch_processor:
            return batch_processor(results)
        else:
            return self._merge_results(results)


    def _merge_results(self, results: List[Any]) -> Any:
        """合并结果"""
        if not results:
            return None
        # 根据结果类型合并
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], list):
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        elif isinstance(results[0], dict):
            merged = {}
            for result in results:
                merged.update(result)
            return merged
        else:
            return results


    def _check_memory_usage(self) -> bool:
        """检查内存使用是否过高"""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.percent > 80  # 超过80%触发清理
        except Exception:
            return False


class MemoryOptimizer:
    """内存优化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 内存监控器
        self.monitor = MemoryMonitor(
            check_interval=self.config.get('monitor_interval', 30),
            enable_tracemalloc=self.config.get('enable_tracemalloc', True)
        )

        # 内存池管理器
        self.pool_manager = MemoryPoolManager(self.config)

        # 数据批处理器
        self.batch_processor = DataBatchProcessor(
            batch_size=self.config.get('batch_size', 1000),
    max_memory_mb=self.config.get('max_memory_mb', 512)
    )

        # 优化策略
        self.gc_threshold = self.config.get('gc_threshold', 1000)
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 1024)

        # 设置GC阈值
        gc.set_threshold(self.gc_threshold, 10, 10)

        # 注册内存告警回调
        self.monitor.register_warning_callback(self._memory_warning_handler)

        logger.info("内存优化器初始化完成")


    def start(self):
        "启动内存优化器"
        self.monitor.start_monitoring()
        logger.info("内存优化器已启动")


    def stop(self):
        "停止内存优化器"
        self.monitor.stop_monitoring()
        logger.info("内存优化器已停止")


    def optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            stats = self.monitor.get_current_stats()
            optimizations = []
            # 强制垃圾回收
            collected = gc.collect()
            if collected > 0:
                optimizations.append(f"垃圾回收释放了{collected} 个对象")

            # 清理内存池
            pool_stats = self.pool_manager.get_pool_stats()
            total_evicted = 0
            for pool_type, pool_stat in pool_stats.items():
                if pool_stat['utilization_percent'] > 90:
                    evicted = self.pool_manager.clear_pool(MemoryPoolType(pool_type))
                    if evicted:
                        total_evicted += 1

            if total_evicted > 0:
                optimizations.append(f"清理了{total_evicted} 个内存池")

            # 检测内存泄漏
            leak_report = self.monitor.detect_memory_leak()
            if leak_report.leak_detected:
                optimizations.append(f"检测到内存泄漏，增长 {leak_report.memory_growth:.2f}MB")

            return {
                'success': True,
                'optimizations': optimizations,
                'current_memory': stats.process_memory,
                'memory_percent': stats.memory_percent,
                'leak_detected': leak_report.leak_detected
            }

        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存报告"""
        try:
            current_stats = self.monitor.get_current_stats()
            memory_trend = self.monitor.get_memory_trend()
            pool_stats = self.pool_manager.get_pool_stats()
            leak_report = self.monitor.detect_memory_leak()
            return {
                'current_stats': {
                    'total_memory': current_stats.total_memory,
                    'used_memory': current_stats.used_memory,
                    'available_memory': current_stats.available_memory,
                    'memory_percent': current_stats.memory_percent,
                    'process_memory': current_stats.process_memory
                },
                'memory_trend': memory_trend,
                'pool_stats': pool_stats,
                'leak_report': {
                    'leak_detected': leak_report.leak_detected,
                    'memory_growth': leak_report.memory_growth,
                    'leaked_objects_count': len(leak_report.leaked_objects),
                    'top_consumers_count': len(leak_report.top_consumers)
                },
                'gc_stats': {
                    'collected_objects': len(gc.get_objects()),
                    'gc_generations': gc.get_count()
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取内存报告失败: {e}")
            return {'error': str(e)}

    def process_large_data(self, data: Any, processor: Callable, batch_processor: Optional[Callable] = None) -> Any:
        """处理大数据"""
        return self.batch_processor.process_large_dataset(data, processor, batch_processor)

    def cache_computation_result(self, key: str, result: Any, pool_type: MemoryPoolType = MemoryPoolType.COMPUTATION_RESULTS) -> bool:
        """缓存计算结果"""
        return self.pool_manager.put_object(pool_type, key, result)

    def get_cached_result(self, key: str, pool_type: MemoryPoolType = MemoryPoolType.COMPUTATION_RESULTS) -> Optional[Any]:
        """获取缓存的计算结果"""
        return self.pool_manager.get_object(pool_type, key)

    def _memory_warning_handler(self, warning: Dict[str, Any]):
        """内存告警处理器"""
        logger.warning(f"内存告警: {warning}")

        # 自动优化
        if warning['level'] in ['high', 'critical']:
            self.optimize_memory_usage()


    def set_gc_strategy(self, strategy: str):
        """设置垃圾回收策略"""
        if strategy == 'aggressive':
            gc.set_threshold(100, 5, 5)
        elif strategy == 'conservative':
            gc.set_threshold(1000, 10, 10)
        elif strategy == 'disabled':
            gc.disable()
        else:
            gc.set_threshold(self.gc_threshold, 10, 10)

        logger.info(f"垃圾回收策略已设置为: {strategy}")


    def force_memory_cleanup(self) -> Dict[str, Any]:
        """强制内存清理"""
        try:
            # 强制垃圾回收
            collected = gc.collect()
            # 清理内存池
            pool_cleaned = 0
            for pool_type in MemoryPoolType:
                if self.pool_manager.clear_pool(pool_type):
                    pool_cleaned += 1

            # 清理未使用的对象
            if OBJGRAPH_AVAILABLE:
                try:
                    objgraph.garbage.clear()
                except Exception:
                    pass

            return {
                'success': True,
                'collected_objects': collected,
                'pools_cleaned': pool_cleaned,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"强制内存清理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
