"""
feature_cache_manager.py

特征数据缓存管理器模块

提供特征数据缓存和优化功能，支持：
- 特征数据内存缓存
- 特征数据持久化缓存（Redis/PostgreSQL）
- 缓存过期策略
- 缓存预热
- 缓存统计和监控

适用于模型训练场景，避免重复计算特征数据，显著提升训练性能。

作者: RQA2025 Team
日期: 2026-02-13
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    enable_memory_cache: bool = True
    enable_persistent_cache: bool = True
    memory_cache_size: int = 100  # 最大缓存条目数
    cache_ttl_minutes: int = 60   # 缓存过期时间
    compression_enabled: bool = True
    cache_key_prefix: str = "feature_data"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0


class FeatureCacheManager:
    """
    特征数据缓存管理器
    
    管理特征数据的缓存，支持内存缓存和持久化缓存，
    显著提升模型训练性能。
    
    Attributes:
        config: 缓存配置
        _memory_cache: 内存缓存字典
        _access_order: 访问顺序列表（用于LRU淘汰）
        
    Example:
        >>> cache_manager = FeatureCacheManager()
        >>> 
        >>> # 缓存特征数据
        >>> cache_manager.cache_features("task_123", features_df, metadata)
        >>> 
        >>> # 获取缓存的特征数据
        >>> cached_data = cache_manager.get_cached_features("task_123")
        >>> if cached_data:
        ...     features = cached_data["features"]
        ...     print("使用缓存的特征数据")
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化特征缓存管理器
        
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        
        # 初始化Redis连接（如果启用）
        self._redis_client = None
        if self.config.enable_persistent_cache:
            try:
                import redis
                self._redis_client = redis.Redis(
                    host='redis',
                    port=6379,
                    db=0,
                    decode_responses=False
                )
                logger.info("Redis缓存连接成功")
            except Exception as e:
                logger.warning(f"Redis连接失败: {e}")
        
        logger.info(f"FeatureCacheManager 初始化完成: "
                   f"memory={self.config.enable_memory_cache}, "
                   f"persistent={self.config.enable_persistent_cache}")
    
    def _generate_cache_key(self, task_id: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [self.config.cache_key_prefix, task_id]
        
        # 添加额外参数到键
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """检查缓存是否有效"""
        age = datetime.now() - entry.created_at
        return age < timedelta(minutes=self.config.cache_ttl_minutes)
    
    def _update_access(self, key: str):
        """更新访问记录"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_if_needed(self):
        """如果需要，淘汰最久未使用的缓存"""
        while len(self._memory_cache) > self.config.memory_cache_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._memory_cache:
                    del self._memory_cache[oldest_key]
                    logger.debug(f"淘汰缓存: {oldest_key}")
    
    def cache_features(
        self,
        task_id: str,
        features: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        缓存特征数据
        
        Args:
            task_id: 特征工程任务ID
            features: 特征DataFrame
            metadata: 元数据
            **kwargs: 额外的缓存键参数
            
        Returns:
            是否成功缓存
        """
        try:
            cache_key = self._generate_cache_key(task_id, **kwargs)
            
            # 准备缓存数据
            cache_data = {
                "features": features,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat(),
                "task_id": task_id
            }
            
            # 1. 内存缓存
            if self.config.enable_memory_cache:
                self._evict_if_needed()
                
                entry = CacheEntry(
                    key=cache_key,
                    data=cache_data,
                    size_bytes=features.memory_usage(deep=True).sum()
                )
                
                self._memory_cache[cache_key] = entry
                self._update_access(cache_key)
                
                logger.debug(f"特征数据已缓存到内存: {cache_key}, "
                           f"大小: {entry.size_bytes / 1024 / 1024:.2f} MB")
            
            # 2. 持久化缓存（Redis）
            if self.config.enable_persistent_cache and self._redis_client:
                try:
                    # 序列化数据
                    serialized = pickle.dumps(cache_data)
                    
                    if self.config.compression_enabled:
                        import zlib
                        serialized = zlib.compress(serialized)
                    
                    # 存储到Redis
                    self._redis_client.setex(
                        cache_key,
                        timedelta(minutes=self.config.cache_ttl_minutes),
                        serialized
                    )
                    
                    logger.debug(f"特征数据已缓存到Redis: {cache_key}")
                    
                except Exception as e:
                    logger.warning(f"Redis缓存失败: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"缓存特征数据失败: {e}")
            return False
    
    def get_cached_features(
        self,
        task_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的特征数据
        
        Args:
            task_id: 特征工程任务ID
            **kwargs: 额外的缓存键参数
            
        Returns:
            缓存的特征数据，如果没有则返回None
        """
        try:
            cache_key = self._generate_cache_key(task_id, **kwargs)
            
            # 1. 检查内存缓存
            if self.config.enable_memory_cache:
                entry = self._memory_cache.get(cache_key)
                
                if entry and self._is_cache_valid(entry):
                    entry.access_count += 1
                    entry.accessed_at = datetime.now()
                    self._update_access(cache_key)
                    
                    logger.debug(f"从内存缓存获取特征数据: {cache_key}")
                    return entry.data
                
                elif entry:
                    # 缓存过期，删除
                    del self._memory_cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
            
            # 2. 检查持久化缓存（Redis）
            if self.config.enable_persistent_cache and self._redis_client:
                try:
                    serialized = self._redis_client.get(cache_key)
                    
                    if serialized:
                        # 反序列化
                        if self.config.compression_enabled:
                            import zlib
                            serialized = zlib.decompress(serialized)
                        
                        cache_data = pickle.loads(serialized)
                        
                        # 同时缓存到内存
                        if self.config.enable_memory_cache:
                            features = cache_data.get("features")
                            if features is not None:
                                self.cache_features(
                                    task_id,
                                    features,
                                    cache_data.get("metadata"),
                                    **kwargs
                                )
                        
                        logger.debug(f"从Redis缓存获取特征数据: {cache_key}")
                        return cache_data
                        
                except Exception as e:
                    logger.warning(f"从Redis获取缓存失败: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存特征数据失败: {e}")
            return None
    
    def invalidate_cache(self, task_id: str, **kwargs):
        """
        使缓存失效
        
        Args:
            task_id: 特征工程任务ID
            **kwargs: 额外的缓存键参数
        """
        try:
            cache_key = self._generate_cache_key(task_id, **kwargs)
            
            # 1. 清除内存缓存
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            
            # 2. 清除Redis缓存
            if self._redis_client:
                self._redis_client.delete(cache_key)
            
            logger.info(f"缓存已失效: {cache_key}")
            
        except Exception as e:
            logger.error(f"使缓存失效失败: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        stats = {
            "memory_cache": {
                "entries": len(self._memory_cache),
                "max_entries": self.config.memory_cache_size,
                "usage_percent": len(self._memory_cache) / self.config.memory_cache_size * 100
            },
            "persistent_cache": {
                "enabled": self.config.enable_persistent_cache,
                "connected": self._redis_client is not None
            }
        }
        
        # 计算内存使用
        if self._memory_cache:
            total_size = sum(entry.size_bytes for entry in self._memory_cache.values())
            total_accesses = sum(entry.access_count for entry in self._memory_cache.values())
            
            stats["memory_cache"]["total_size_mb"] = round(total_size / 1024 / 1024, 2)
            stats["memory_cache"]["total_accesses"] = total_accesses
            stats["memory_cache"]["avg_accesses"] = round(total_accesses / len(self._memory_cache), 2)
        
        return stats
    
    def clear_all_cache(self):
        """清除所有缓存"""
        try:
            # 清除内存缓存
            self._memory_cache.clear()
            self._access_order.clear()
            
            # 清除Redis缓存
            if self._redis_client:
                pattern = f"{self.config.cache_key_prefix}*"
                for key in self._redis_client.scan_iter(match=pattern):
                    self._redis_client.delete(key)
            
            logger.info("所有缓存已清除")
            
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
    
    def preload_cache(
        self,
        task_ids: List[str],
        feature_loader: callable
    ) -> Dict[str, Any]:
        """
        预加载缓存
        
        Args:
            task_ids: 任务ID列表
            feature_loader: 特征数据加载函数
            
        Returns:
            预加载结果统计
        """
        results = {
            "total": len(task_ids),
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        for task_id in task_ids:
            try:
                # 检查是否已缓存
                if self.get_cached_features(task_id):
                    results["details"].append({
                        "task_id": task_id,
                        "status": "already_cached"
                    })
                    continue
                
                # 加载特征数据
                feature_data = feature_loader(task_id)
                
                if feature_data and "features" in feature_data:
                    # 缓存数据
                    self.cache_features(
                        task_id,
                        feature_data["features"],
                        feature_data.get("metadata")
                    )
                    
                    results["success"] += 1
                    results["details"].append({
                        "task_id": task_id,
                        "status": "cached"
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "task_id": task_id,
                        "status": "load_failed"
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"缓存预加载完成: 总计 {results['total']}, "
                   f"成功 {results['success']}, 失败 {results['failed']}")
        
        return results


# 全局缓存管理器实例（单例模式）
_global_cache_manager: Optional[FeatureCacheManager] = None


def get_feature_cache_manager(config: Optional[CacheConfig] = None) -> FeatureCacheManager:
    """
    获取全局特征缓存管理器实例
    
    Args:
        config: 缓存配置
        
    Returns:
        特征缓存管理器实例
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = FeatureCacheManager(config)
    
    return _global_cache_manager


def close_feature_cache_manager():
    """关闭全局特征缓存管理器实例"""
    global _global_cache_manager
    
    if _global_cache_manager:
        _global_cache_manager.clear_all_cache()
        _global_cache_manager = None
