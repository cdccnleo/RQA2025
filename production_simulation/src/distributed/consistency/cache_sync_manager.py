"""
缓存同步管理器

负责分布式缓存的同步和管理。

从cache_consistency.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any

from .consistency_models import CacheEntry, NodeInfo, ConsistencyLevel
from .consistency_manager import ConsistencyManager

logger = logging.getLogger(__name__)


class DistributedCacheManager:
    """
    分布式缓存管理器
    
    负责:
    1. 分布式缓存的读写
    2. 缓存同步
    3. 一致性保证
    4. 缓存失效
    """

    def __init__(self, node_id: str, nodes: List[NodeInfo],
                 consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG):
        self.node_id = node_id
        self.consistency_manager = ConsistencyManager(node_id, nodes, consistency_level)
        
        # 本地缓存
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_reads': 0,
            'total_writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'sync_operations': 0
        }
        
        # 后台清理线程
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        
        logger.info(f"分布式缓存管理器初始化: {node_id}")

    def start(self):
        """启动缓存管理器"""
        if self._running:
            return
        
        self._running = True
        self.consistency_manager.start()
        
        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"分布式缓存管理器已启动: {self.node_id}")

    def stop(self):
        """停止缓存管理器"""
        self._running = False
        self.consistency_manager.stop()
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        logger.info(f"分布式缓存管理器已停止: {self.node_id}")

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存"""
        try:
            # 复制操作到其他节点
            from .consistency_models import OperationType
            if not self.consistency_manager.replicate_operation(
                OperationType.SET, key, value, ttl
            ):
                logger.warning(f"缓存设置失败（复制失败）: {key}")
                return False
            
            # 更新本地缓存
            with self._lock:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    version=self._get_next_version(key),
                    timestamp=time.time(),
                    ttl=ttl,
                    node_id=self.node_id
                )
                self._cache[key] = entry
                self.stats['total_writes'] += 1
            
            logger.debug(f"缓存已设置: {key}")
            return True
            
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存"""
        try:
            with self._lock:
                self.stats['total_reads'] += 1
                
                if key in self._cache:
                    entry = self._cache[key]
                    
                    # 检查是否过期
                    if entry.is_expired():
                        del self._cache[key]
                        self.stats['cache_misses'] += 1
                        return default
                    
                    self.stats['cache_hits'] += 1
                    return entry.value
                
                self.stats['cache_misses'] += 1
                return default
                
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return default

    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            # 复制删除操作
            from .consistency_models import OperationType
            if not self.consistency_manager.replicate_operation(
                OperationType.DELETE, key
            ):
                logger.warning(f"缓存删除失败（复制失败）: {key}")
                return False
            
            # 删除本地缓存
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"缓存已删除: {key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    def clear(self) -> bool:
        """清空所有缓存"""
        try:
            # 复制清空操作
            from .consistency_models import OperationType
            if not self.consistency_manager.replicate_operation(
                OperationType.CLEAR, ""
            ):
                logger.warning("缓存清空失败（复制失败）")
                return False
            
            # 清空本地缓存
            with self._lock:
                self._cache.clear()
                logger.info("缓存已清空")
                return True
                
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self.stats.copy()

    def _get_next_version(self, key: str) -> int:
        """获取下一个版本号"""
        if key in self._cache:
            return self._cache[key].version + 1
        return 1

    def _cleanup_worker(self):
        """清理过期缓存的工作线程"""
        while self._running:
            try:
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                        logger.debug(f"清理过期缓存: {key}")
                
                time.sleep(60)  # 每分钟清理一次
                
            except Exception as e:
                logger.error(f"清理工作线程异常: {e}")
                time.sleep(10)


__all__ = ['DistributedCacheManager']

