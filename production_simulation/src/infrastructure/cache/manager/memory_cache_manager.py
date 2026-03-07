
import threading
import time

from collections import OrderedDict
from typing import Dict, Any, Optional, List
#!/usr/bin/env python3
"""
内存缓存管理器
"""


class MemoryCacheManager:
    """内存缓存管理器"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化内存缓存管理器

        Args:
            max_size: 最大缓存条目数
            ttl: 默认TTL（秒）
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # 检查是否过期
                if time.time() > entry['expires_at']:
                    self._delete(key)
                    self._stats['misses'] += 1
                    return None

                # 移到最后（最近使用）
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                return entry['value']
            else:
                self._stats['misses'] += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒），如果为None则使用默认值
        """
        with self._lock:
            expires_at = time.time() + (ttl or self.default_ttl)

            # 检查是否需要清理空间
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._cache.move_to_end(key)
            self._stats['sets'] += 1

    def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['deletes'] += 1
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()

    def _delete(self, key: str) -> None:
        """内部删除方法"""
        if key in self._cache:
            del self._cache[key]

    def _evict_oldest(self) -> None:
        """驱逐最旧的条目（LRU策略）"""
        if self._cache:
            oldest_key, _ = self._cache.popitem(last=False)
            self._stats['evictions'] += 1

    def cleanup_expired(self) -> int:
        """
        清理过期的条目

        Returns:
            清理的条目数量
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time > entry['expires_at']
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'evictions': self._stats['evictions'],
                'default_ttl': self.default_ttl
            }

    def get_all_keys(self) -> List[str]:
        """
        获取所有缓存键

        Returns:
            缓存键列表
        """
        with self._lock:
            return list(self._cache.keys())

    def has_key(self, key: str) -> bool:
        """
        检查键是否存在且未过期

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                return time.time() <= entry['expires_at']
            return False

    def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        """
        更新键的过期时间

        Args:
            key: 缓存键
            ttl: 新的TTL（秒），如果为None则使用默认值

        Returns:
            是否成功更新
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry['expires_at'] = time.time() + (ttl or self.default_ttl)
                self._cache.move_to_end(key)
                return True
            return False
