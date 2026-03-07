#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层智能缓存系统

集成基础设施层的智能缓存算法，实现更智能的数据缓存策略。
"""

from ..interfaces.ICacheBackend import ICacheBackend
from ..interfaces.standard_interfaces import DataSourceType, IDataCache
import threading
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 日志降级处理


def get_data_logger(name: str):
    """获取数据层日志器，支持降级"""
    try:
        from src.infrastructure.logging import UnifiedLogger
        return UnifiedLogger(name)
    except ImportError:
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger('smart_data_cache')

# 导入基础设施层智能缓存
try:
    from src.infrastructure.cache.smart_cache_strategies import (
        LFUCache, LRUKCache, AdaptiveCache, PriorityCache, CostAwareCache
    )
    from infrastructure.cache.interfaces import CacheEvictionStrategy as CacheStrategy
    INFRASTRUCTURE_CACHE_AVAILABLE = True
except ImportError:
    logger.warning("无法导入基础设施层智能缓存，使用本地实现")
    INFRASTRUCTURE_CACHE_AVAILABLE = False
    from types import SimpleNamespace

    class _PlaceholderCache:
        def __init__(self, capacity, *args, **kwargs):
            self.capacity = capacity
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def put(self, key, value):
            self.store[key] = value

        def exists(self, key):
            return key in self.store

        def clear(self):
            self.store.clear()

        def size(self):
            return len(self.store)

    LFUCache = _PlaceholderCache
    LRUKCache = _PlaceholderCache
    AdaptiveCache = _PlaceholderCache
    PriorityCache = _PlaceholderCache
    CostAwareCache = _PlaceholderCache
    CacheStrategy = SimpleNamespace(
        LFU="LFU",
        LRU_K="LRU_K",
        ADAPTIVE="ADAPTIVE",
        PRIORITY="PRIORITY",
        COST_AWARE="COST_AWARE",
    )

# 导入标准接口


@dataclass
class DataCacheConfig:

    """数据缓存配置"""
    strategy: str = "lfu"  # 缓存策略：lfu, lru_k, adaptive, priority, cost_aware
    capacity: int = 1000   # 缓存容量
    lru_k: int = 2         # LRU - K的K值
    adaptive_window: int = 100  # 自适应窗口大小
    priority_levels: int = 3    # 优先级级别
    cost_threshold: float = 10.0  # 成本感知阈值
    enable_stats: bool = True   # 启用统计
    cleanup_interval: int = 300  # 清理间隔(秒)


@dataclass
class CacheStats:

    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    last_cleanup: datetime = field(default_factory=datetime.now)

    def update_hit_rate(self):
        """更新命中率"""
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests
        else:
            self.hit_rate = 0.0


class SmartDataCacheBackend(ICacheBackend):

    """
    智能数据缓存后端

    集成基础设施层的智能缓存算法
    """

    def __init__(self, config: DataCacheConfig):

        self.config = config
        self._lock = threading.RLock()
        self.stats = CacheStats()

        # 根据策略创建缓存实例
        if INFRASTRUCTURE_CACHE_AVAILABLE:
            self._create_infrastructure_cache()
        else:
            self._create_fallback_cache()

        logger.info(f"智能缓存后端初始化完成，策略: {config.strategy}")

    def _create_infrastructure_cache(self):
        """创建基础设施层缓存实例"""
        strategy_map = {
            "lfu": (CacheStrategy.LFU, lambda: LFUCache(self.config.capacity)),
            "lru_k": (CacheStrategy.LRU_K, lambda: LRUKCache(self.config.capacity, self.config.lru_k)),
            "adaptive": (CacheStrategy.ADAPTIVE, lambda: AdaptiveCache(self.config.capacity, self.config.adaptive_window)),
            "priority": (CacheStrategy.PRIORITY, lambda: PriorityCache(self.config.capacity, self.config.priority_levels)),
            "cost_aware": (CacheStrategy.COST_AWARE, lambda: CostAwareCache(self.config.capacity, self.config.cost_threshold)),
        }

        strategy, cache_factory = strategy_map.get(self.config.strategy, strategy_map["lfu"])
        self.cache = cache_factory()
        self.strategy = strategy

    def _create_fallback_cache(self):
        """创建降级缓存实例"""
        from collections import OrderedDict

        class FallbackCache:

            def __init__(self, capacity):

                self.capacity = capacity
                self.cache = OrderedDict()

            def get(self, key):

                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                return None

            def put(self, key, value):

                if key in self.cache:
                    self.cache.move_to_end(key)
                else:
                    if len(self.cache) >= self.capacity:
                        self.cache.popitem(last=False)
                self.cache[key] = value

            def pop(self, key, default=None):

                return self.cache.pop(key, default)

            def size(self):

                return len(self.cache)

        self.cache = FallbackCache(self.config.capacity)
        self.strategy = "fallback_lru"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self._lock:
            self.stats.total_requests += 1

            value = self.cache.get(key)
            if value is not None:
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return value
            else:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        with self._lock:
            try:
                self.stats.sets += 1
                self.cache.put(key, value)
                return True
            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存数据"""
        with self._lock:
            try:
                # 检查缓存是否有delete方法
                if hasattr(self.cache, 'delete'):
                    return self.cache.delete(key)
                elif hasattr(self.cache, 'pop'):
                    self.cache.pop(key, None)
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"删除缓存失败: {e}")
                return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        with self._lock:
            if hasattr(self.cache, 'exists'):
                return self.cache.exists(key)
            else:
                return self.get(key) is not None

    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            try:
                if hasattr(self.cache, 'clear'):
                    self.cache.clear()
                else:
                    # 重新创建缓存实例
                    if INFRASTRUCTURE_CACHE_AVAILABLE:
                        self._create_infrastructure_cache()
                    else:
                        self._create_fallback_cache()

                self.stats = CacheStats()
                return True
            except Exception as e:
                logger.error(f"清空缓存失败: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                "strategy": self.config.strategy,
                "capacity": self.config.capacity,
                "current_size": self.cache.size() if hasattr(self.cache, 'size') else len(self.cache.cache) if hasattr(self.cache, 'cache') else 0,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "sets": self.stats.sets,
                "evictions": self.stats.evictions,
                "total_requests": self.stats.total_requests,
                "hit_rate": round(self.stats.hit_rate * 100, 2),
                "last_cleanup": self.stats.last_cleanup.isoformat(),
                "infrastructure_cache": INFRASTRUCTURE_CACHE_AVAILABLE
            }


class SmartDataCache(IDataCache):

    """
    智能数据缓存

    基于数据类型的智能缓存策略，支持多种缓存算法
    """

    def __init__(self, config: Optional[DataCacheConfig] = None,


                 backend: Optional[ICacheBackend] = None):
        self.config = config or DataCacheConfig()

        if backend:
            self.backend = backend
        else:
            self.backend = SmartDataCacheBackend(self.config)

        # 按数据类型配置不同的缓存策略
        self._type_configs = self._create_type_configs()

        logger.info("智能数据缓存初始化完成")

    def _create_type_configs(self) -> Dict[DataSourceType, Dict[str, Any]]:
        """创建按数据类型的配置"""
        # 不同数据类型的缓存策略配置
        configs = {
            "default": {"ttl": 3600, "priority": 2}
        }

        type_mappings = {
            "DATABASE": {"ttl": 3600, "priority": 3},
            "API": {"ttl": 1800, "priority": 2},
            "STREAM": {"ttl": 300, "priority": 3},
            "FILE": {"ttl": 86400, "priority": 1},
            "CACHE": {"ttl": 600, "priority": 2},
        }

        for attr, cfg in type_mappings.items():
            member = getattr(DataSourceType, attr, None)
            if member is not None:
                configs[member] = cfg

        return configs

    def get(self, key: str, data_type: DataSourceType) -> Optional[Any]:
        """获取缓存数据"""
        try:
            return self.backend.get(key)
        except Exception as e:
            logger.error(f"获取缓存数据失败: {e}")
            return None

    def set(self, key: str, value: Any, data_type: DataSourceType, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            # 根据数据类型获取TTL
            if ttl is None:
                type_config = self._type_configs.get(data_type, self._type_configs["default"])
                ttl = type_config.get("ttl", self.config.capacity)

            return self.backend.set(key, value, ttl)
        except Exception as e:
            logger.error(f"设置缓存数据失败: {e}")
            return False

    def invalidate(self, pattern: str) -> int:
        """按模式失效缓存"""
        try:
            # 这里简化实现，实际可以根据pattern进行更复杂的匹配
            if pattern == "*":
                return 1 if self.backend.clear() else 0
            else:
                # 尝试删除匹配的键
                deleted_count = 0
                # 这里需要实现更复杂的模式匹配逻辑
                return deleted_count
        except Exception as e:
            logger.error(f"失效缓存失败: {e}")
            return 0

    def delete(self, key: str, data_type: Optional[DataSourceType] = None) -> bool:
        """删除指定键的缓存数据"""
        try:
            return self.backend.delete(key)
        except Exception as e:
            logger.error(f"删除缓存数据失败: {e}")
            return False

    def clear(self) -> bool:
        """清空缓存"""
        try:
            return self.backend.clear()
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            if hasattr(self.backend, "exists"):
                return self.backend.exists(key)
            return self.backend.get(key) is not None
        except Exception as e:
            logger.error(f"检查缓存存在性失败: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        try:
            backend_stats = self.backend.get_stats()
            return {
                **backend_stats,
                "data_type_configs": self._type_configs,
                "config": {
                    "strategy": self.config.strategy,
                    "capacity": self.config.capacity,
                    "enable_stats": self.config.enable_stats
                }
            }
        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")
            return {}

    def optimize_for_data_type(self, data_type: DataSourceType) -> bool:
        """为特定数据类型优化缓存策略"""
        try:
            type_config = self._type_configs.get(data_type, self._type_configs["default"])

            # 根据数据类型调整缓存行为
            if data_type == getattr(DataSourceType, "STREAM", None):
                # 加密货币数据更新频繁，使用更短的TTL
                logger.info(f"为 {data_type.value} 优化缓存策略: 高频更新")
            elif data_type == getattr(DataSourceType, "FILE", None):
                # 宏观经济数据相对稳定，使用更长的TTL
                logger.info(f"为 {data_type.value} 优化缓存策略: 长期稳定")

            return True
        except Exception as e:
            logger.error(f"优化缓存策略失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            return self.backend.get_stats()
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


# 工厂函数

def create_smart_data_cache(config: Optional[DataCacheConfig] = None) -> SmartDataCache:
    """创建智能数据缓存"""
    return SmartDataCache(config)


def create_data_cache_backend(config: Optional[DataCacheConfig] = None) -> SmartDataCacheBackend:
    """创建数据缓存后端"""
    config = config or DataCacheConfig()
    return SmartDataCacheBackend(config)


# 导出主要类和函数
__all__ = [
    'DataCacheConfig',
    'CacheStats',
    'SmartDataCacheBackend',
    'SmartDataCache',
    'create_smart_data_cache',
    'create_data_cache_backend'
]
