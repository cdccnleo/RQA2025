
import logging
import threading
import time
from typing import Any, Dict, Optional

"""
RQA2025 Config Services Cache Service

配置服务缓存服务
"""

logger = logging.getLogger(__name__)


class CacheService:

    """缓存服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, maxsize: Optional[int] = None):

        self.config = config or {}
        self.maxsize = maxsize or 1000
        self.initialized = False
        self.cache = {}
        self.timestamps = {}  # 存储时间戳用于过期检查
        self.access_times = {}  # 存储访问时间用于LRU
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def initialize(self) -> bool:
        """初始化服务"""
        try:
            self.initialized = True
            logger.info("配置缓存服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"配置缓存服务初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭服务"""
        try:
            self.cache.clear()
            self.initialized = False
            logger.info("配置缓存服务已关闭")
            return True
        except Exception as e:
            logger.error(f"配置缓存服务关闭失败: {e}")
            return False

    def get(self, key: str) -> Any:
        """获取缓存项"""
        if not self.initialized:
            return None

        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if self._is_expired(key):
                    self._remove_item(key)
                    self.misses += 1
                    return None

                # 更新访问时间
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self.initialized:
            return False

        with self.lock:
            current_time = time.time()

            # 检查容量限制
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self._evict_items()

            # 设置缓存项
            self.cache[key] = value
            self.timestamps[key] = current_time + (ttl or 3600)  # 默认1小时
            # 只有在key不存在时才设置初始访问时间，已存在则保持原有访问时间
            if key not in self.access_times:
                self.access_times[key] = current_time

            return True

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self.initialized:
            return False

        with self.lock:
            if key in self.cache:
                self._remove_item(key)
                return True
            return False

    def clear(self) -> bool:
        """清空缓存"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")

        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
            return True

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")

        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate * 100, 2),  # 百分比
                'total_requests': total_requests,
                'initialized': self.initialized
            }

    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self.timestamps:
            return True
        return time.time() > self.timestamps[key]

    def _remove_item(self, key: str):
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.access_times:
            del self.access_times[key]

    def _evict_items(self):
        """驱逐缓存项（LRU策略）"""
        if not self.cache:
            return

        # 找到最少使用的项（最早访问的）
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove_item(lru_key)
        logger.debug(f"LRU驱逐缓存项: {lru_key}")

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'service': 'config_cache_service',
            'status': 'healthy' if self.initialized else 'uninitialized',
            'cache_size': len(self.cache),
            'hit_rate': self.get_stats().get('hit_rate', 0) if self.initialized else 0
        }




