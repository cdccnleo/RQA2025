import logging
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
RQA2025 交易层内存优化模块

实现内存池和对象重用机制，提升高频交易性能
"""

import threading
import time
from typing import Any, Dict, Optional, Type, TypeVar, Generic
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import gc

# 导入统一基础设施集成层
try:
    from src.core.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryPoolStats:

    """内存池统计信息"""
    pool_name: str
    object_type: str
    pool_size: int
    active_objects: int
    available_objects: int
    total_allocated: int
    total_deallocated: int
    peak_usage: int
    cache_hits: int
    cache_misses: int
    average_allocation_time: float
    average_deallocation_time: float


class MemoryPool(Generic[T]):

    """通用内存池实现"""

    def __init__(self, object_type: Type[T], pool_size: int = 1000,


                 max_pool_size: int = 10000, name: str = None):
        self.object_type = object_type
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.name = name or f"{object_type.__name__}_pool"

        # 内存池存储
        self._pool = deque(maxlen=max_pool_size)
        self._lock = threading.RLock()

        # 统计信息
        self.stats = MemoryPoolStats(
            pool_name=self.name,
            object_type=object_type.__name__,
            pool_size=pool_size,
            active_objects=0,
            available_objects=0,
            total_allocated=0,
            total_deallocated=0,
            peak_usage=0,
            cache_hits=0,
            cache_misses=0,
            average_allocation_time=0.0,
            average_deallocation_time=0.0
        )

        # 性能监控
        self._allocation_times = deque(maxlen=1000)
        self._deallocation_times = deque(maxlen=1000)

        # 初始化对象池
        self._initialize_pool()

        logger.info(f"内存池初始化完成: {self.name}, 大小: {pool_size}")

    def _initialize_pool(self):
        """初始化对象池"""
        for _ in range(self.pool_size):
            try:
                # 尝试无参数初始化
                obj = self.object_type()
                self._pool.append(obj)
            except TypeError as e:
                # 如果无参数初始化失败，尝试提供默认参数
                logger.debug(f"无参数初始化失败，尝试提供默认参数: {e}")
                try:
                    obj = self._create_object_with_defaults()
                    if obj:
                        self._pool.append(obj)
                    else:
                        logger.error(f"无法创建对象: {self.object_type.__name__}")
                except Exception as e2:
                    logger.error(f"创建对象失败: {e2}")
            except Exception as e:
                logger.error(f"初始化对象失败: {e}")

    def _create_object_with_defaults(self):
        """使用默认参数创建对象"""
        from datetime import datetime

        class_name = self.object_type.__name__

        if class_name == 'OrderBookEntry':
            # 为OrderBookEntry提供默认参数
            return self.object_type(
                price=0.0,
                quantity=0.0,
                timestamp=datetime.now()
            )
        elif class_name == 'HFTrade':
            # 为HFTrade提供默认参数
            from ...hft.execution.hft_execution_engine import HFTStrategy
            return self.object_type(
                trade_id="",
                symbol="",
                side="",
                quantity=0.0,
                price=0.0,
                timestamp=datetime.now(),
                latency_us=0,
                strategy=HFTStrategy.MARKET_MAKING
            )
        elif class_name == 'Order':
            # 为Order提供默认参数
            return self.object_type(
                order_id="",
                symbol="",
                order_type="",
                quantity=0.0
            )
        else:
            # 对于其他类，使用无参数初始化
            return None

        self.stats.available_objects = len(self._pool)

    def acquire(self) -> T:
        """获取对象"""
        start_time = time.perf_counter()

        with self._lock:
            obj = None

            # 尝试从池中获取
            if self._pool:
                obj = self._pool.popleft()
                self.stats.cache_hits += 1
            else:
                # 池为空，创建新对象
                try:
                    obj = self.object_type()
                    self.stats.cache_misses += 1
                except Exception as e:
                    logger.error(f"创建对象失败: {e}")
                    return None

            if obj is not None:
                self.stats.active_objects += 1
                self.stats.total_allocated += 1
                self.stats.peak_usage = max(self.stats.peak_usage, self.stats.active_objects)

                # 记录分配时间
                allocation_time = time.perf_counter() - start_time
                self._allocation_times.append(allocation_time)

        if len(self._allocation_times) > 0:
            self.stats.average_allocation_time = sum(
                self._allocation_times) / len(self._allocation_times)

        return obj

    def release(self, obj: T) -> bool:
        """释放对象回池"""
        if obj is None:
            return False

        start_time = time.perf_counter()

        with self._lock:
            try:
                # 重置对象状态（如果有reset方法）
                if hasattr(obj, 'reset'):
                    obj.reset()

                # 检查池是否已满
                if len(self._pool) < self.max_pool_size:
                    self._pool.append(obj)
                    self.stats.available_objects += 1
                else:
                    # 池已满，丢弃对象
                    logger.debug(f"对象池已满，丢弃对象: {self.name}")

                self.stats.active_objects -= 1
                self.stats.total_deallocated += 1

                # 记录释放时间
                deallocation_time = time.perf_counter() - start_time
                self._deallocation_times.append(deallocation_time)

                if len(self._deallocation_times) > 0:
                    self.stats.average_deallocation_time = sum(
                        self._deallocation_times) / len(self._deallocation_times)

                return True

            except Exception as e:
                logger.error(f"释放对象失败: {e}")
                return False

    def get_stats(self) -> MemoryPoolStats:
        """获取统计信息"""
        with self._lock:
            return self.stats

    def resize_pool(self, new_size: int):
        """调整池大小"""
        with self._lock:
            if new_size > self.max_pool_size:
                new_size = self.max_pool_size

            if new_size > self.pool_size:
                # 扩大池
                for _ in range(new_size - self.pool_size):
                    try:
                        obj = self.object_type()
                        self._pool.append(obj)
                    except Exception as e:
                        logger.error(f"扩大池时创建对象失败: {e}")
            else:
                # 缩小池
                while len(self._pool) > new_size and self._pool:
                    self._pool.pop()

            self.pool_size = new_size
            self.stats.pool_size = new_size
            self.stats.available_objects = len(self._pool)

    def clear_pool(self):
        """清空池"""
        with self._lock:
            self._pool.clear()
            self.stats.available_objects = 0
            self.stats.active_objects = 0

            # 强制垃圾回收
            gc.collect()


class TradingObjectPool:

    """交易对象专用内存池"""

    def __init__(self):

        # 基础设施集成
        self._infrastructure_adapter = None
        self._monitoring = None

        # 对象池集合
        self._pools: Dict[str, MemoryPool] = {}
        self._lock = threading.RLock()

        # 初始化基础设施集成
        self._init_infrastructure_integration()

        # 初始化交易对象池
        self._initialize_trading_pools()

        logger.info("交易对象池初始化完成")

    def _init_infrastructure_integration(self):
        """初始化基础设施集成"""
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            logger.warning("统一基础设施集成层不可用，内存池使用降级模式")
            return

        try:
            self._infrastructure_adapter = get_trading_layer_adapter()

            if self._infrastructure_adapter:
                services = self._infrastructure_adapter.get_infrastructure_services()
                self._monitoring = services.get('monitoring')

                logger.info("交易对象池成功连接统一基础设施集成层")
        except Exception as e:
            logger.error(f"基础设施集成初始化失败: {e}")

    def _initialize_trading_pools(self):
        """初始化交易对象池"""
        # 订单对象池
        from ...order_manager import Order
        self.create_pool("order_pool", Order, pool_size=5000)

        # 交易对象池
        from ...hft.execution.hft_execution_engine import HFTrade
        self.create_pool("trade_pool", HFTrade, pool_size=10000)

        # 订单簿条目池
        from ...hft.execution.hft_execution_engine import OrderBookEntry
        self.create_pool("orderbook_entry_pool", OrderBookEntry, pool_size=20000)

        # 市场数据池
        self.create_pool("market_data_pool", dict, pool_size=5000)

        logger.info("交易专用对象池初始化完成")

    def create_pool(self, pool_name: str, object_type: Type[T],


                    pool_size: int = 1000, max_pool_size: int = 10000):
        """创建对象池"""
        with self._lock:
            if pool_name in self._pools:
                logger.warning(f"对象池已存在: {pool_name}")
                return

            pool = MemoryPool(object_type, pool_size, max_pool_size, pool_name)
            self._pools[pool_name] = pool

            logger.info(f"创建对象池: {pool_name}, 类型: {object_type.__name__}, 大小: {pool_size}")

    def get_pool(self, pool_name: str) -> Optional[MemoryPool]:
        """获取对象池"""
        with self._lock:
            return self._pools.get(pool_name)

    def acquire_object(self, pool_name: str) -> Optional[Any]:
        """从指定池获取对象"""
        pool = self.get_pool(pool_name)
        if pool:
            obj = pool.acquire()

            # 基础设施集成：记录监控指标
            if self._monitoring:
                try:
                    self._monitoring.record_metric(
                        'memory_pool_acquire',
                        1,
                        {
                            'pool_name': pool_name,
                            'layer': 'trading'
                        }
                    )
                except Exception as e:
                    logger.warning(f"记录内存池获取指标失败: {e}")

            return obj
        else:
            logger.warning(f"对象池不存在: {pool_name}")
            return None

    def release_object(self, pool_name: str, obj: Any) -> bool:
        """释放对象到指定池"""
        pool = self.get_pool(pool_name)
        if pool:
            success = pool.release(obj)

            # 基础设施集成：记录监控指标
            if self._monitoring:
                try:
                    self._monitoring.record_metric(
                        'memory_pool_release',
                        1,
                        {
                            'pool_name': pool_name,
                            'success': success,
                            'layer': 'trading'
                        }
                    )
                except Exception as e:
                    logger.warning(f"记录内存池释放指标失败: {e}")

            return success
        else:
            logger.warning(f"对象池不存在: {pool_name}")
            return False

    def get_all_stats(self) -> Dict[str, MemoryPoolStats]:
        """获取所有池的统计信息"""
        stats = {}
        with self._lock:
            for pool_name, pool in self._pools.items():
                stats[pool_name] = pool.get_stats()
        return stats

    def resize_pool(self, pool_name: str, new_size: int):
        """调整池大小"""
        pool = self.get_pool(pool_name)
        if pool:
            pool.resize_pool(new_size)
            logger.info(f"调整对象池大小: {pool_name} -> {new_size}")

    def clear_all_pools(self):
        """清空所有池"""
        with self._lock:
            for pool_name, pool in self._pools.items():
                pool.clear_pool()
            logger.info("清空所有对象池")

    def optimize_pools(self):
        """优化池配置"""
        stats = self.get_all_stats()

        for pool_name, pool_stats in stats.items():
            # 根据使用率调整池大小
            usage_rate = pool_stats.active_objects / pool_stats.pool_size if pool_stats.pool_size > 0 else 0

            if usage_rate > 0.8:  # 使用率超过80%
                # 扩大池
                new_size = int(pool_stats.pool_size * 1.5)
                self.resize_pool(pool_name, new_size)
            elif usage_rate < 0.2 and pool_stats.pool_size > 100:  # 使用率低于20%
                # 缩小池
                new_size = int(pool_stats.pool_size * 0.8)
                self.resize_pool(pool_name, new_size)

        logger.info("对象池优化完成")

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            'component': 'TradingObjectPool',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'pools_count': len(self._pools),
            'infrastructure_integration': INFRASTRUCTURE_INTEGRATION_AVAILABLE,
            'pools_stats': {},
            'warnings': [],
            'critical_issues': []
        }

        stats = self.get_all_stats()
        total_active = 0
        total_available = 0

        for pool_name, pool_stats in stats.items():
            health_info['pools_stats'][pool_name] = {
                'active_objects': pool_stats.active_objects,
                'available_objects': pool_stats.available_objects,
                'pool_size': pool_stats.pool_size,
                'cache_hit_rate': (pool_stats.cache_hits
                                   / (pool_stats.cache_hits + pool_stats.cache_misses)
                                   if (pool_stats.cache_hits + pool_stats.cache_misses) > 0 else 0)
            }

            total_active += pool_stats.active_objects
            total_available += pool_stats.available_objects

            # 检查池状态
        if pool_stats.active_objects > pool_stats.pool_size:
            health_info['warnings'].append(f"池 {pool_name} 活跃对象超过池大小")

        if pool_stats.available_objects == 0:
            health_info['critical_issues'].append(f"池 {pool_name} 已无可用对象")

        # 总体状态评估
        if health_info['critical_issues']:
            health_info['status'] = 'critical'
        elif health_info['warnings']:
            health_info['status'] = 'warning'

        health_info['summary'] = {
            'total_active_objects': total_active,
            'total_available_objects': total_available,
            'pool_utilization_rate': total_active / (total_active + total_available) if (total_active + total_available) > 0 else 0
        }

        return health_info


class MemoryManager:

    """内存管理器"""

    def __init__(self):

        self.object_pool = TradingObjectPool()
        self._gc_stats = {
            'collections': 0,
            'collected_objects': 0,
            'uncollectable_objects': 0
        }

        # 启动内存监控线程
        self._monitoring_thread = threading.Thread(
            target=self._memory_monitor,
            name='memory_monitor',
            daemon=True
        )
        self._monitoring_thread.start()

        logger.info("内存管理器初始化完成")

    def _memory_monitor(self):
        """内存监控线程"""
        while True:
            try:
                # 定期优化池配置
                self.object_pool.optimize_pools()

                # 收集GC统计信息
                import gc
                gc_stats = gc.get_stats()
                if gc_stats:
                    total_collections = sum(stat['collections'] for stat in gc_stats)
                    total_collected = sum(stat['collected'] for stat in gc_stats)
                    total_uncollectable = sum(stat['uncollectable'] for stat in gc_stats)

                    self._gc_stats = {
                        'collections': total_collections,
                        'collected_objects': total_collected,
                        'uncollectable_objects': total_uncollectable
                    }

                time.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"内存监控异常: {e}")
                time.sleep(30)

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        return {
            'memory_usage': {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent()
            },
            'pool_stats': self.object_pool.get_all_stats(),
            'gc_stats': self._gc_stats,
            'timestamp': datetime.now().isoformat()
        }

    def force_gc(self):
        """强制垃圾回收"""
        import gc
        collected = gc.collect()
        logger.info(f"强制垃圾回收完成，回收对象数: {collected}")

    def optimize_memory(self):
        """优化内存使用"""
        # 优化对象池
        self.object_pool.optimize_pools()

        # 强制垃圾回收
        self.force_gc()

        # 清空不必要的缓存
        self._clear_caches()

        logger.info("内存优化完成")

    def _clear_caches(self):
        """清空缓存"""
        try:
            # 清空Python内部缓存
            import sys
            sys.path_importer_cache.clear()

            # 清空模块缓存（谨慎使用）
            # modules_to_clear = [name for name in sys.modules.keys() if name.startswith('src.trading')]
            # for module_name in modules_to_clear:
            #     if module_name in sys.modules:
            #         del sys.modules[module_name]

            logger.debug("缓存清理完成")

        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = self.object_pool.health_check()
        health_info['component'] = 'MemoryManager'

        # 添加内存使用情况
        memory_stats = self.get_memory_stats()
        health_info['memory_usage'] = memory_stats['memory_usage']
        health_info['gc_stats'] = memory_stats['gc_stats']

        return health_info


# 全局内存管理器实例
_memory_manager = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器实例"""
    global _memory_manager

    if _memory_manager is None:
        with _memory_manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()

    return _memory_manager


def acquire_trading_object(pool_name: str) -> Optional[Any]:
    """从交易对象池获取对象"""
    manager = get_memory_manager()
    return manager.object_pool.acquire_object(pool_name)


def release_trading_object(pool_name: str, obj: Any) -> bool:
    """释放对象到交易对象池"""
    manager = get_memory_manager()
    return manager.object_pool.release_object(pool_name, obj)
