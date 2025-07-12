"""缓存服务 (v3.1)

职责：
1. 管理高性能线程安全缓存
2. 提供监控指标
3. 支持异步操作
"""
from typing import Dict, Any, Optional, List
from src.infrastructure.cache.thread_safe_cache import ThreadSafeTTLCache
from src.infrastructure.error.exceptions import CacheError
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import time

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(
        self,
        maxsize: int = 10000,
        ttl: int = 3600,
        max_memory_mb: int = 1024,
        thread_pool_size: int = 4
    ):
        """初始化缓存服务

        Args:
            maxsize: 最大缓存项数
            ttl: 默认存活时间(秒)
            max_memory_mb: 最大内存使用(MB)
            thread_pool_size: 异步操作线程池大小
        """
        self.cache = ThreadSafeTTLCache(
            maxsize=maxsize,
            ttl=ttl,
            timer=time.time,
            max_memory_mb=max_memory_mb,
            enable_lru=True,
            compression_threshold=1024*1024  # 1MB
        )
        self._stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }
        self._metrics = {
            'thread_pool_size': thread_pool_size,
            'memory_usage': 0
        }
        self._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        """获取缓存项"""
        try:
            value = self.cache[key]
            # 兼容底层缓存返回的(value, expire_at)元组
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (int, float)):
                value = value[0]
            self._stats['hits'] += 1
            return value
        except KeyError:
            self._stats['misses'] += 1
            return None
        except Exception as e:
            raise CacheError(f"缓存读取失败: {str(e)}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存项"""
        try:
            # 使用缓存的TTL支持
            if ttl is not None:
                self.cache.set_with_ttl(key, value, ttl)
            else:
                self.cache[key] = value
            self._stats['writes'] += 1
        except Exception as e:
            raise CacheError(f"缓存写入失败: {str(e)}")

    def invalidate(self, key: str) -> None:
        """失效缓存项"""
        try:
            del self.cache[key]
            self._stats['evictions'] += 1
        except KeyError:
            pass
        except Exception as e:
            raise CacheError(f"缓存失效失败: {str(e)}")

    def bulk_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """批量设置缓存项"""
        if not items:
            return

        try:
            # 使用缓存的批量操作接口
            if ttl is not None:
                self.cache.bulk_set_with_ttl(items, ttl)
            else:
                self.cache.bulk_set(items)
            self._stats['writes'] += len(items)
        except Exception as e:
            raise CacheError(f"批量写入失败: {str(e)}")

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        stats = self._stats.copy()
        stats.update(self.cache.get_metrics())
        return stats

    def get_metrics(self) -> Dict[str, Any]:
        """获取缓存性能指标"""
        metrics = self._metrics.copy()
        metrics['memory_usage'] = self.cache.memory_usage
        metrics.update({
            'hit_rate': self.cache.hit_rate,
            'eviction_rate': self.cache.eviction_rate
        })
        return metrics

    def update_thread_pool_size(self, new_size: int) -> None:
        """动态更新线程池大小
        
        参数:
            new_size: 新的线程池大小
        """
        with self._lock:
            if new_size != self._thread_pool._max_workers:
                # 创建新线程池
                old_pool = self._thread_pool
                self._thread_pool = ThreadPoolExecutor(max_workers=new_size)
                old_pool.shutdown(wait=True)
                
                self._metrics['thread_pool_size'] = new_size
                logger.info(f"线程池大小已更新: {new_size}")

    def _update_bulk_ttl(self, keys: List[str], ttl: int) -> None:
        """异步批量更新TTL"""
        try:
            for key in keys:
                if key in self.cache:
                    self.cache.set_ttl(key, ttl)
        except Exception as e:
            logger.warning(f"批量更新TTL失败: {str(e)}")
