"""
feature_cache_manager.py

特征数据缓存管理器模块

提供特征数据缓存和优化功能，支持：
- 特征数据内存缓存
- 特征数据持久化缓存（PostgreSQL优先，Redis/文件系统降级）
- 缓存过期策略
- 缓存预热
- 缓存统计和监控

适用于模型训练场景，避免重复计算特征数据，显著提升训练性能。

作者: RQA2025 Team
日期: 2026-02-13
更新: 2026-03-22 (PostgreSQL优先持久化策略)
"""

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    enable_memory_cache: bool = True
    enable_persistent_cache: bool = True
    enable_postgresql: bool = True
    memory_cache_size: int = 100
    cache_ttl_minutes: int = 60
    compression_enabled: bool = True
    cache_key_prefix: str = "feature_data"
    base_path: str = "./feature_cache"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    storage_type: str = "memory"


class FeatureCacheManager:
    """
    特征数据缓存管理器
    
    管理特征数据的缓存，支持内存缓存和持久化缓存，
    实现PostgreSQL优先存储策略。
    
    存储优先级:
    - 主存储: PostgreSQL 数据库
    - 降级存储: Redis / 文件系统
    
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
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化特征缓存管理器
        
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        
        self._pg_config = None
        self._pg_available = False
        self._redis_client = None
        
        if self.config.enable_postgresql:
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self._ensure_tables_exist()
                logger.info("✅ FeatureCacheManager: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ FeatureCacheManager: PostgreSQL 不可用，尝试Redis/文件系统")
        
        if self.config.enable_persistent_cache and not self._pg_available:
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
                logger.warning(f"Redis连接失败: {e}，使用文件系统")
        
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'memory_hits': 0,
            'postgresql_hits': 0,
            'redis_hits': 0,
            'filesystem_hits': 0,
            'misses': 0,
            'total_cached': 0,
        }
        
        logger.info(f"FeatureCacheManager 初始化完成: "
                   f"memory={self.config.enable_memory_cache}, "
                   f"postgresql={self._pg_available}, "
                   f"redis={self._redis_client is not None}")

    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """获取PostgreSQL配置（使用统一配置模块）"""
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
        except ImportError:
            logger.debug("统一数据库配置模块导入失败，尝试从环境变量获取")
            return {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
                "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
                "password": os.getenv("POSTGRES_PASSWORD", "SecurePass123!")
            }
        except Exception as e:
            logger.error(f"获取PostgreSQL配置失败: {e}")
            return None

    def _get_db_connection(self):
        """获取数据库连接（带重试机制）"""
        if not self._pg_config:
            return None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self._pg_config["host"],
                    port=self._pg_config["port"],
                    database=self._pg_config["database"],
                    user=self._pg_config["user"],
                    password=self._pg_config["password"],
                    connect_timeout=5
                )
                return conn
            except Exception as e:
                logger.debug(f"PostgreSQL连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
        
        return None

    def _test_postgresql_connection(self) -> bool:
        """测试PostgreSQL连接是否可用"""
        conn = self._get_db_connection()
        if conn:
            conn.close()
            return True
        return False

    def _ensure_tables_exist(self):
        """确保数据库表存在"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feature_cache (
                        cache_key VARCHAR(64) PRIMARY KEY,
                        task_id VARCHAR(128) NOT NULL,
                        cache_data BYTEA,
                        metadata JSONB DEFAULT '{}',
                        size_bytes BIGINT DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        expires_at TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_task 
                    ON feature_cache(task_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_expires 
                    ON feature_cache(expires_at)
                """)
            
            conn.commit()
            logger.debug("特征缓存表已确保存在")
            
        except Exception as e:
            logger.error(f"创建特征缓存表失败: {e}")
        finally:
            if conn:
                conn.close()

    def _generate_cache_key(self, task_id: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [self.config.cache_key_prefix, task_id]
        
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
        缓存特征数据（PostgreSQL优先，降级到Redis/文件系统）
        
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
            
            cache_data = {
                "features": features,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat(),
                "task_id": task_id
            }
            
            serialized = pickle.dumps(cache_data)
            
            if self.config.compression_enabled:
                import gzip
                serialized = gzip.compress(serialized)
            
            size_bytes = len(serialized)
            
            if self.config.enable_memory_cache:
                self._evict_if_needed()
                
                entry = CacheEntry(
                    key=cache_key,
                    data=cache_data,
                    size_bytes=features.memory_usage(deep=True).sum(),
                    storage_type="memory"
                )
                
                self._memory_cache[cache_key] = entry
                self._update_access(cache_key)
                
                logger.debug(f"特征数据已缓存到内存: {cache_key}")
            
            storage_type = "memory"
            
            if self._pg_available:
                if self._save_to_postgresql(cache_key, task_id, serialized, metadata, size_bytes):
                    storage_type = "postgresql"
                    self.stats['total_cached'] += 1
                else:
                    logger.warning("PostgreSQL缓存失败，尝试Redis/文件系统")
            
            if storage_type == "memory" and self._redis_client:
                try:
                    self._redis_client.setex(
                        cache_key,
                        timedelta(minutes=self.config.cache_ttl_minutes),
                        serialized
                    )
                    storage_type = "redis"
                    logger.debug(f"特征数据已缓存到Redis: {cache_key}")
                except Exception as e:
                    logger.warning(f"Redis缓存失败: {e}")
            
            if storage_type == "memory":
                self._save_to_filesystem(cache_key, serialized)
                storage_type = "filesystem"
            
            return True
            
        except Exception as e:
            logger.error(f"缓存特征数据失败: {e}")
            return False

    def _save_to_postgresql(
        self,
        cache_key: str,
        task_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]],
        size_bytes: int
    ) -> bool:
        """保存缓存到PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            expires_at = datetime.now() + timedelta(minutes=self.config.cache_ttl_minutes)
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_cache (
                        cache_key, task_id, cache_data, metadata, 
                        size_bytes, created_at, accessed_at, access_count, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        task_id = EXCLUDED.task_id,
                        cache_data = EXCLUDED.cache_data,
                        metadata = EXCLUDED.metadata,
                        size_bytes = EXCLUDED.size_bytes,
                        created_at = CURRENT_TIMESTAMP,
                        accessed_at = CURRENT_TIMESTAMP,
                        access_count = 0,
                        expires_at = EXCLUDED.expires_at
                """, (
                    cache_key,
                    task_id,
                    data,
                    json.dumps(metadata or {}),
                    size_bytes,
                    datetime.now(),
                    datetime.now(),
                    0,
                    expires_at
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存到PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _save_to_filesystem(self, cache_key: str, data: bytes) -> bool:
        """保存缓存到文件系统"""
        try:
            cache_file = self.base_path / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error(f"保存到文件系统失败: {e}")
            return False

    def get_cached_features(
        self,
        task_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的特征数据（优先从内存，然后PostgreSQL，最后Redis/文件系统）
        
        Args:
            task_id: 特征工程任务ID
            **kwargs: 额外的缓存键参数
            
        Returns:
            缓存的特征数据，如果没有则返回None
        """
        try:
            cache_key = self._generate_cache_key(task_id, **kwargs)
            
            if self.config.enable_memory_cache:
                entry = self._memory_cache.get(cache_key)
                
                if entry and self._is_cache_valid(entry):
                    entry.access_count += 1
                    entry.accessed_at = datetime.now()
                    self._update_access(cache_key)
                    
                    self.stats['memory_hits'] += 1
                    logger.debug(f"从内存缓存获取特征数据: {cache_key}")
                    return entry.data
                
                elif entry:
                    del self._memory_cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
            
            if self._pg_available:
                result = self._get_from_postgresql(cache_key)
                if result is not None:
                    self.stats['postgresql_hits'] += 1
                    logger.debug(f"从PostgreSQL获取特征数据: {cache_key}")
                    
                    if self.config.enable_memory_cache:
                        self._memory_cache[cache_key] = CacheEntry(
                            key=cache_key,
                            data=result,
                            storage_type="postgresql"
                        )
                        self._update_access(cache_key)
                    
                    return result
            
            if self._redis_client:
                try:
                    serialized = self._redis_client.get(cache_key)
                    
                    if serialized:
                        if self.config.compression_enabled:
                            import gzip
                            serialized = gzip.decompress(serialized)
                        
                        cache_data = pickle.loads(serialized)
                        self.stats['redis_hits'] += 1
                        logger.debug(f"从Redis获取特征数据: {cache_key}")
                        
                        if self.config.enable_memory_cache:
                            self._memory_cache[cache_key] = CacheEntry(
                                key=cache_key,
                                data=cache_data,
                                storage_type="redis"
                            )
                            self._update_access(cache_key)
                        
                        return cache_data
                        
                except Exception as e:
                    logger.warning(f"从Redis获取缓存失败: {e}")
            
            result = self._get_from_filesystem(cache_key)
            if result is not None:
                self.stats['filesystem_hits'] += 1
                logger.debug(f"从文件系统获取特征数据: {cache_key}")
                
                if self.config.enable_memory_cache:
                    self._memory_cache[cache_key] = CacheEntry(
                        key=cache_key,
                        data=result,
                        storage_type="filesystem"
                    )
                    self._update_access(cache_key)
                
                return result
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存特征数据失败: {e}")
            return None

    def _get_from_postgresql(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从PostgreSQL获取缓存"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT cache_data, expires_at FROM feature_cache
                    WHERE cache_key = %s
                """, (cache_key,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                data, expires_at = row
                
                if expires_at and datetime.now() > expires_at:
                    cur.execute("DELETE FROM feature_cache WHERE cache_key = %s", (cache_key,))
                    conn.commit()
                    return None
                
                cur.execute(
                    "UPDATE feature_cache SET access_count = access_count + 1, accessed_at = CURRENT_TIMESTAMP WHERE cache_key = %s",
                    (cache_key,)
                )
                conn.commit()
                
                if self.config.compression_enabled:
                    try:
                        import gzip
                        data = gzip.decompress(data)
                    except Exception:
                        pass
                
                return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"从PostgreSQL获取缓存失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _get_from_filesystem(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从文件系统获取缓存"""
        try:
            cache_file = self.base_path / f"{cache_key}.cache"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = f.read()
            
            if self.config.compression_enabled:
                try:
                    import gzip
                    data = gzip.decompress(data)
                except Exception:
                    pass
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"从文件系统获取缓存失败: {e}")
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
            
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            
            if self._pg_available:
                conn = None
                try:
                    conn = self._get_db_connection()
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM feature_cache WHERE cache_key = %s", (cache_key,))
                        conn.commit()
                except Exception:
                    pass
                finally:
                    if conn:
                        conn.close()
            
            if self._redis_client:
                self._redis_client.delete(cache_key)
            
            cache_file = self.base_path / f"{cache_key}.cache"
            if cache_file.exists():
                cache_file.unlink()
            
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
                "postgresql_enabled": self._pg_available,
                "redis_enabled": self._redis_client is not None
            },
            "hits": {
                "memory": self.stats['memory_hits'],
                "postgresql": self.stats['postgresql_hits'],
                "redis": self.stats['redis_hits'],
                "filesystem": self.stats['filesystem_hits'],
                "misses": self.stats['misses']
            },
            "total_cached": self.stats['total_cached']
        }
        
        total_hits = (self.stats['memory_hits'] + self.stats['postgresql_hits'] + 
                      self.stats['redis_hits'] + self.stats['filesystem_hits'])
        total_requests = total_hits + self.stats['misses']
        stats["hit_rate"] = total_hits / total_requests if total_requests > 0 else 0.0
        
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
            self._memory_cache.clear()
            self._access_order.clear()
            
            if self._pg_available:
                conn = None
                try:
                    conn = self._get_db_connection()
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM feature_cache")
                        conn.commit()
                except Exception:
                    pass
                finally:
                    if conn:
                        conn.close()
            
            if self._redis_client:
                pattern = f"{self.config.cache_key_prefix}*"
                for key in self._redis_client.scan_iter(match=pattern):
                    self._redis_client.delete(key)
            
            for cache_file in self.base_path.glob("*.cache"):
                cache_file.unlink()
            
            logger.info("所有缓存已清除")
            
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")

    def cleanup_expired_cache(self) -> int:
        """清理过期缓存"""
        cleaned = 0
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM feature_cache WHERE expires_at < CURRENT_TIMESTAMP")
                        cleaned = cur.rowcount
                    conn.commit()
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if not self._is_cache_valid(entry)
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            cleaned += 1
        
        if cleaned > 0:
            logger.info(f"清理了 {cleaned} 个过期缓存")
        
        return cleaned

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
                if self.get_cached_features(task_id):
                    results["details"].append({
                        "task_id": task_id,
                        "status": "already_cached"
                    })
                    continue
                
                feature_data = feature_loader(task_id)
                
                if feature_data and "features" in feature_data:
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
