#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热点数据缓存模块

实现热点数据的自动识别和缓存，提高频繁访问数据的读取速度。
"""

import time
import threading
import logging
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict, OrderedDict

from .cache_manager import CacheManager, CacheConfig

logger = logging.getLogger('hot_data_cache')


@dataclass
class HotDataStats:
    """热点数据统计信息"""
    access_count: int = 0
    last_accessed: float = 0.0
    first_accessed: float = 0.0
    access_frequency: float = 0.0  # 访问频率（次/秒）
    size: int = 0  # 数据大小（字节）


class HotDataCache:
    """热点数据缓存"""

    def __init__(self, config: CacheConfig = None, hot_threshold: int = 10, check_interval: int = 60):
        """
        初始化热点数据缓存

        Args:
            config: 缓存配置
            hot_threshold: 热点阈值（访问次数）
            check_interval: 检查间隔（秒）
        """
        self.cache = CacheManager(config)
        self.hot_threshold = hot_threshold
        self.check_interval = check_interval
        self._stats: Dict[str, HotDataStats] = defaultdict(HotDataStats)
        self._hot_keys: Dict[str, float] = {}  # 热点键及其热度分数
        self._lock = threading.RLock()
        self._running = False
        self._check_thread = None

        # 启动热点检测线程
        self._start_check_thread()

    def _start_check_thread(self):
        """启动热点检测线程"""
        def check_worker():
            """热点检测工作线程"""
            while self._running:
                try:
                    self._detect_hot_data()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Error in hot data check thread: {e}")
                    time.sleep(1)

        self._running = True
        self._check_thread = threading.Thread(target=check_worker, daemon=True)
        self._check_thread.start()

    def _detect_hot_data(self):
        """检测热点数据"""
        with self._lock:
            current_time = time.time()
            hot_keys = {}

            for key, stats in self._stats.items():
                # 计算访问频率
                if stats.first_accessed > 0:
                    time_window = current_time - stats.first_accessed
                    if time_window > 0:
                        stats.access_frequency = stats.access_count / time_window

                # 判断是否为热点数据
                if stats.access_count >= self.hot_threshold or stats.access_frequency >= 0.1:
                    # 计算热度分数
                    heat_score = stats.access_count * 0.7 + stats.access_frequency * 30
                    hot_keys[key] = heat_score

            # 更新热点键
            self._hot_keys = hot_keys

            if hot_keys:
                logger.info(f"Detected {len(hot_keys)} hot keys: {list(hot_keys.keys())[:5]}")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            # 更新访问统计
            stats = self._stats[key]
            stats.access_count += 1
            stats.last_accessed = time.time()
            if stats.first_accessed == 0:
                stats.first_accessed = stats.last_accessed

            # 从缓存获取数据
            value = self.cache.get(key)
            
            # 记录缓存命中/未命中
            if value is not None:
                from src.monitoring.core.real_time_monitor import update_business_metric
                try:
                    update_business_metric('cache_hit', 1.0)
                except Exception:
                    pass
            else:
                from src.monitoring.core.real_time_monitor import update_business_metric
                try:
                    update_business_metric('cache_miss', 1.0)
                except Exception:
                    pass

            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            # 设置缓存
            result = self.cache.set(key, value, ttl)
            
            # 更新统计信息
            if result:
                stats = self._stats[key]
                stats.size = len(str(value))  # 简化计算

            return result

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            result = self.cache.delete(key)
            if result and key in self._stats:
                del self._stats[key]
            if key in self._hot_keys:
                del self._hot_keys[key]
            return result

    def clear(self) -> int:
        """清空缓存"""
        with self._lock:
            result = self.cache.clear()
            self._stats.clear()
            self._hot_keys.clear()
            return result

    def get_hot_keys(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """获取热点键列表"""
        with self._lock:
            sorted_hot = sorted(self._hot_keys.items(), key=lambda x: x[1], reverse=True)
            return sorted_hot[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            base_stats = self.cache.get_stats()
            hot_stats = {
                'hot_keys_count': len(self._hot_keys),
                'total_keys_count': len(self._stats),
                'top_hot_keys': self.get_hot_keys(5),
                'hot_threshold': self.hot_threshold,
                'check_interval': self.check_interval
            }
            base_stats.update({'hot_data': hot_stats})
            return base_stats

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.cache.exists(key)

    def list_keys(self) -> List[str]:
        """列出所有键"""
        return self.cache.list_keys()

    def stop(self):
        """停止缓存管理器"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        self.cache.stop()

    def __del__(self):
        """析构函数"""
        try:
            self.stop()
        except Exception:
            pass


class HotDataCacheManager:
    """热点数据缓存管理器"""

    def __init__(self):
        self._caches: Dict[str, HotDataCache] = {}
        self._lock = threading.RLock()

    def get_cache(self, name: str, config: CacheConfig = None) -> HotDataCache:
        """获取或创建缓存实例"""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = HotDataCache(config)
            return self._caches[name]

    def remove_cache(self, name: str) -> bool:
        """移除缓存实例"""
        with self._lock:
            if name in self._caches:
                cache = self._caches[name]
                cache.stop()
                del self._caches[name]
                return True
            return False

    def get_all_caches(self) -> Dict[str, HotDataCache]:
        """获取所有缓存实例"""
        with self._lock:
            return self._caches.copy()

    def stop_all(self):
        """停止所有缓存实例"""
        with self._lock:
            for cache in self._caches.values():
                cache.stop()
            self._caches.clear()


# 全局热点数据缓存管理器实例
_hot_data_cache_manager = None


def get_hot_data_cache_manager() -> HotDataCacheManager:
    """获取全局热点数据缓存管理器实例"""
    global _hot_data_cache_manager
    if _hot_data_cache_manager is None:
        _hot_data_cache_manager = HotDataCacheManager()
    return _hot_data_cache_manager


def get_hot_data_cache(name: str = 'default', config: CacheConfig = None) -> HotDataCache:
    """获取热点数据缓存实例"""
    manager = get_hot_data_cache_manager()
    return manager.get_cache(name, config)


__all__ = ['HotDataCache', 'HotDataCacheManager', 'get_hot_data_cache_manager', 'get_hot_data_cache']
