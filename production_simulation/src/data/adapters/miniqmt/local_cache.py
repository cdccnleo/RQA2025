#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
MiniQMT本地缓存
在网络中断时提供有限服务，支持数据缓存和降级处理
"""

import time
import threading
import logging
import pickle
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)


class CacheType(Enum):

    """缓存类型"""
    MARKET_DATA = "market_data"      # 行情数据
    ORDER_DATA = "order_data"        # 订单数据
    ACCOUNT_DATA = "account_data"    # 账户数据
    CONFIG_DATA = "config_data"      # 配置数据


class CacheStrategy(Enum):

    """缓存策略"""
    LRU = "lru"              # 最近最少使用
    LFU = "lfu"              # 最少使用频率
    FIFO = "fifo"            # 先进先出
    TTL = "ttl"              # 基于时间过期


@dataclass
class CacheItem:

    """缓存项"""
    key: str
    value: Any
    cache_type: CacheType
    created_time: float
    last_access_time: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0


class LocalCache:

    """MiniQMT本地缓存管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地缓存

        Args:
            config: 缓存配置
        """
        self.config = config
        self.max_size = config.get('max_size', 100 * 1024 * 1024)  # 100MB
        self.max_items = config.get('max_items', 10000)
        self.default_ttl = config.get('default_ttl', 300)  # 5分钟
        self.strategy = CacheStrategy(config.get("approach", 'lru'))

        # 缓存存储
        self._cache: Dict[str, CacheItem] = OrderedDict()
        self._type_index: Dict[CacheType, List[str]] = {
            cache_type: [] for cache_type in CacheType
        }

        # 锁和状态
        self._lock = threading.RLock()
        self._running = False
        self._cleanup_thread = None

        # 统计信息
        self._stats = {
            'total_items': 0,
            'current_size': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

        # 持久化配置
        self.persistence_enabled = config.get('persistence_enabled', True)
        self.persistence_file = config.get('persistence_file', 'miniqmt_cache.dat')
        self.persistence_interval = config.get('persistence_interval', 60)  # 60秒

    def start(self):
        """启动缓存管理器"""
        if self._running:
            return

        self._running = True

        # 加载持久化数据
        if self.persistence_enabled:
            self._load_persistence()

        # 启动清理线程
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="LocalCache - Cleanup"
        )
        self._cleanup_thread.start()

        logger.info("MiniQMT本地缓存已启动")

    def stop(self):
        """停止缓存管理器"""
        self._running = False

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        # 保存持久化数据
        if self.persistence_enabled:
            self._save_persistence()

        logger.info("MiniQMT本地缓存已停止")

    def get(self, key: str, cache_type: CacheType = None) -> Optional[Any]:
        """
        获取缓存项

        Args:
            key: 缓存键
            cache_type: 缓存类型

        Returns:
            缓存值，如果不存在或已过期返回None
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None

            item = self._cache[key]

            # 检查类型匹配
            if cache_type and item.cache_type != cache_type:
                self._stats['misses'] += 1
                return None

            # 检查TTL过期
            if item.ttl and time.time() - item.created_time > item.ttl:
                self._remove_item(key)
                self._stats['expirations'] += 1
                self._stats['misses'] += 1
                return None

            # 更新访问信息
            item.last_access_time = time.time()
            item.access_count += 1
            self._stats['hits'] += 1

            # 根据策略调整顺序
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)

            return item.value

    def set(self, key: str, value: Any, cache_type: CacheType,


            ttl: Optional[float] = None, size: Optional[int] = None) -> bool:
        """
        设置缓存项

        Args:
            key: 缓存键
            value: 缓存值
            cache_type: 缓存类型
            ttl: 生存时间
            size: 预估大小

        Returns:
            是否设置成功
        """
        with self._lock:
            # 计算大小
            if size is None:
                size = self._estimate_size(value)

            # 检查是否需要清理空间
            if self._current_size + size > self.max_size:
                self._evict_items(size)

            # 创建缓存项
            item = CacheItem(
                key=key,
                value=value,
                cache_type=cache_type,
                created_time=time.time(),
                last_access_time=time.time(),
                ttl=ttl or self.default_ttl,
                size=size
            )

            # 如果键已存在，先移除
            if key in self._cache:
                self._remove_item(key)

            # 添加新项
            self._cache[key] = item
            self._type_index[cache_type].append(key)
            self._current_size += size
            self._stats['total_items'] += 1

            # 检查数量限制
            if len(self._cache) > self.max_items:
                self._evict_items(0, count=1)

            logger.debug(f"缓存项已设置: {key}, 类型: {cache_type.value}, 大小: {size}")
            return True

    def delete(self, key: str) -> bool:
        """
        删除缓存项

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False

    def clear(self, cache_type: CacheType = None):
        """
        清空缓存

        Args:
            cache_type: 指定类型，如果为None则清空所有
        """
        with self._lock:
            if cache_type:
                keys_to_remove = self._type_index[cache_type].copy()
                for key in keys_to_remove:
                    if key in self._cache:
                        self._remove_item(key)
                self._type_index[cache_type].clear()
            else:
                self._cache.clear()
                for cache_type in CacheType:
                    self._type_index[cache_type].clear()
                self._current_size = 0
                self._stats['total_items'] = 0

    def get_by_type(self, cache_type: CacheType) -> Dict[str, Any]:
        """
        获取指定类型的所有缓存项

        Args:
            cache_type: 缓存类型

        Returns:
            缓存项字典
        """
        with self._lock:
            result = {}
            for key in self._type_index[cache_type]:
                if key in self._cache:
                    item = self._cache[key]
                    # 检查TTL
                    if item.ttl and time.time() - item.created_time > item.ttl:
                        self._remove_item(key)
                        continue
                    result[key] = item.value
            return result

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'current_size': self._current_size,
                'max_size': self.max_size,
                'cache_items': len(self._cache),
                'max_items': self.max_items,
                "approach": self.strategy.value,
                'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            })

            # 按类型统计
            for cache_type in CacheType:
                stats[f'{cache_type.value}_items'] = len(self._type_index[cache_type])

            return stats

    def _remove_item(self, key: str):
        """移除缓存项"""
        if key in self._cache:
            item = self._cache[key]
            self._current_size -= item.size
            self._stats['total_items'] -= 1

            # 从类型索引中移除
            if key in self._type_index[item.cache_type]:
                self._type_index[item.cache_type].remove(key)

            del self._cache[key]

    def _evict_items(self, required_size: int, count: int = 0):
        """清理缓存项"""
        if self.strategy == CacheStrategy.LRU:
            # 移除最久未使用的项
            while (self._current_size + required_size > self.max_size
                   or (count > 0 and len(self._cache) > self.max_items - count)):
                if not self._cache:
                    break
                key = next(iter(self._cache))
                self._remove_item(key)
                self._stats['evictions'] += 1

        elif self.strategy == CacheStrategy.LFU:
            # 移除使用频率最低的项
            while (self._current_size + required_size > self.max_size
                   or (count > 0 and len(self._cache) > self.max_items - count)):
                if not self._cache:
                    break
                min_key = min(self._cache.keys(),
                              key=lambda k: self._cache[k].access_count)
                self._remove_item(min_key)
                self._stats['evictions'] += 1

        elif self.strategy == CacheStrategy.FIFO:
            # 移除最先加入的项
            while (self._current_size + required_size > self.max_size
                   or (count > 0 and len(self._cache) > self.max_items - count)):
                if not self._cache:
                    break
                key = next(iter(self._cache))
                self._remove_item(key)
                self._stats['evictions'] += 1

    def _estimate_size(self, value: Any) -> int:
        """估算缓存项大小"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(v) for v in value.values())
            else:
                # 使用pickle估算
                return len(pickle.dumps(value))
        except BaseException:
            return 1024  # 默认1KB

    def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                self._cleanup_expired_items()
                time.sleep(30)  # 每30秒清理一次
            except Exception as e:
                logger.error(f"缓存清理异常: {e}")

    def _cleanup_expired_items(self):
        """清理过期项"""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, item in self._cache.items():
                if item.ttl and current_time - item.created_time > item.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_item(key)
                self._stats['expirations'] += 1

    def _save_persistence(self):
        """保存持久化数据"""
        try:
            data = {
                'cache': {key: asdict(item) for key, item in self._cache.items()},
                'type_index': {cache_type.value: keys for cache_type, keys in self._type_index.items()},
                'stats': self._stats
            }

            with open(self.persistence_file, 'wb') as f:
                pickle.dump(data, f)

            logger.info("缓存数据已持久化")

        except Exception as e:
            logger.error(f"缓存持久化失败: {e}")

    def _load_persistence(self):
        """加载持久化数据"""
        try:
            if not os.path.exists(self.persistence_file):
                return

            with open(self.persistence_file, 'rb') as f:
                data = pickle.load(f)

            # 恢复缓存项
            for key, item_dict in data['cache'].items():
                item = CacheItem(**item_dict)
                self._cache[key] = item

            # 恢复类型索引
            for cache_type_str, keys in data['type_index'].items():
                cache_type = CacheType(cache_type_str)
                self._type_index[cache_type] = keys

            # 恢复统计信息
            self._stats.update(data['stats'])

            logger.info("缓存数据已从持久化文件加载")

        except Exception as e:
            logger.error(f"缓存数据加载失败: {e}")

    @property
    def _current_size(self) -> int:
        """当前缓存大小"""
        return self._stats.get('current_size', 0)

    @_current_size.setter
    def _current_size(self, value: int):
        """设置当前缓存大小"""
        self._stats['current_size'] = value

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
