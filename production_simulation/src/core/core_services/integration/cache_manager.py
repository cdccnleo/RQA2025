"""
缓存管理器

提供响应缓存管理功能。
"""

import logging
import time
import threading
from typing import Dict, Any, Optional

from src.core.constants import MAX_QUEUE_SIZE, DEFAULT_TEST_TIMEOUT

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器 - 职责：管理响应缓存"""

    def __init__(self, max_size: int = MAX_QUEUE_SIZE, ttl: int = DEFAULT_TEST_TIMEOUT):
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = max_size
        self._cache_ttl = ttl

    def get(self, cache_key: str) -> Optional[Any]:
        """获取缓存"""
        with self._cache_lock:
            if cache_key in self._response_cache:
                cached_item = self._response_cache[cache_key]
                if time.time() - cached_item['timestamp'] < self._cache_ttl:
                    return cached_item['data']
                else:
                    # 缓存过期，删除
                    del self._response_cache[cache_key]
            return None

    def set(self, cache_key: str, data: Any) -> None:
        """设置缓存"""
        with self._cache_lock:
            # 检查缓存大小限制
            if len(self._response_cache) >= self._max_cache_size:
                # 简单的LRU策略：删除最旧的缓存项
                oldest_key = min(
                    self._response_cache.keys(),
                    key=lambda k: self._response_cache[k]['timestamp']
                )
                del self._response_cache[oldest_key]

            self._response_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }

    def clear(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._response_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._cache_lock:
            return {
                'cache_size': len(self._response_cache),
                'max_cache_size': self._max_cache_size,
                'cache_ttl': self._cache_ttl
            }

