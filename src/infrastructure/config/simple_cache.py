from collections import OrderedDict
from threading import RLock
from time import time
from typing import Any, Optional

class SimpleCache:
    """线程安全的LRU缓存实现，支持TTL过期"""

    def __init__(self, max_size: int = 1000, ttl: int = 300, max_weight: int = None):
        """
        Args:
            max_size: 最大缓存项数
            ttl: 缓存过期时间(秒)
            max_weight: 最大权重限制(可选)
        """
        self._max_size = max_size
        self._ttl = ttl
        self._max_weight = max_weight
        self._cache = OrderedDict()
        self._lock = RLock()
        self._current_weight = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，不存在或过期返回None"""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]
            if time() - timestamp > self._ttl:
                del self._cache[key]
                return None

            # 更新访问顺序
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time())

            # 淘汰最久未使用的项
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def batch_get(self, keys: list[str]) -> dict[str, Any]:
        """批量获取缓存值
        
        Args:
            keys: 要获取的键列表
            
        Returns:
            包含键值对的字典，不存在的键不会包含在结果中
        """
        with self._lock:
            results = {}
            for key in keys:
                if key in self._cache:
                    value, timestamp = self._cache[key]
                    if time() - timestamp <= self._ttl:
                        results[key] = value
                        self._cache.move_to_end(key)
                    else:
                        del self._cache[key]
            return results

    def batch_set(self, items: dict[str, Any], weights: dict[str, int] = None) -> None:
        """批量设置缓存值
        
        Args:
            items: 键值对字典
            weights: 可选权重字典，键必须与items一致
        """
        with self._lock:
            for key, value in items.items():
                weight = weights.get(key, 1) if weights else 1
                if key in self._cache:
                    old_weight = self._cache[key][2] if len(self._cache[key]) > 2 else 1
                    self._current_weight -= old_weight
                self._cache[key] = (value, time(), weight)
                self._current_weight += weight
                self._cache.move_to_end(key)
                
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """根据容量和权重执行淘汰"""
        while len(self._cache) > self._max_size or (
            self._max_weight and self._current_weight > self._max_weight
        ):
            key, (_, _, weight) = self._cache.popitem(last=False)
            self._current_weight -= weight

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_weight = 0
