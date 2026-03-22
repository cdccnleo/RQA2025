#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理层持久化实现

实现 PostgreSQL 优先存储策略：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统

遵循已实施层（特征、模型、策略、交易、风险）的重构标准。
"""

import json
import logging
import os
import time
import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntryMetadata:
    """缓存条目元数据"""
    cache_key: str
    cache_type: str
    data_size_bytes: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class LineageRecord:
    """数据血缘记录"""
    id: Optional[int] = None
    data_type: str = ""
    source_info: Dict[str, Any] = field(default_factory=dict)
    transform_info: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetricRecord:
    """数据质量指标记录"""
    id: Optional[int] = None
    data_type: str = ""
    completeness: float = 0.0
    accuracy: float = 0.0
    timeliness: float = 0.0
    consistency: float = 0.0
    uniqueness: float = 0.0
    checked_at: datetime = field(default_factory=datetime.now)
    issues: List[str] = field(default_factory=list)


@dataclass
class ComplianceCheckRecord:
    """合规检查记录"""
    id: Optional[int] = None
    policy_id: str = ""
    data_type: str = ""
    is_compliant: bool = True
    issues: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)
    check_duration_ms: float = 0.0


class BasePersistence(ABC):
    """
    持久化基类
    
    实现 PostgreSQL 优先存储策略，数据库连接失败时自动降级到文件系统。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, storage_dir: str = "data/persistence"):
        """
        初始化持久化基类
        
        Args:
            storage_dir: 文件系统存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        
        self._pg_config = None
        self._pg_available = False
        
        self._init_postgresql()
        
        self._stats = {
            'total_saves': 0,
            'total_loads': 0,
            'postgresql_saves': 0,
            'filesystem_saves': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _init_postgresql(self):
        """
        初始化 PostgreSQL 连接配置
        
        使用统一的数据库配置模块获取配置。
        """
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            self._pg_config = config.to_dict()
            self._pg_available = self._test_postgresql_connection()
            
            if self._pg_available:
                logger.info(f"✅ {self.__class__.__name__}: PostgreSQL 存储已启用")
            else:
                logger.warning(f"⚠️ {self.__class__.__name__}: PostgreSQL 不可用，使用文件系统存储")
        except Exception as e:
            logger.error(f"获取 PostgreSQL 配置失败: {e}")
            self._pg_available = False
    
    def _get_db_connection(self):
        """
        获取数据库连接（带重试机制）
        
        Returns:
            数据库连接对象或 None
        """
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
                logger.debug(f"PostgreSQL 连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
        
        return None
    
    def _test_postgresql_connection(self) -> bool:
        """
        测试 PostgreSQL 连接是否可用
        
        Returns:
            连接是否成功
        """
        conn = self._get_db_connection()
        if conn:
            conn.close()
            return True
        return False
    
    @abstractmethod
    def save(self, *args, **kwargs) -> bool:
        """
        保存数据（子类实现）
        
        Returns:
            是否保存成功
        """
        pass
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """
        加载数据（子类实现）
        
        Returns:
            加载的数据
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats['postgresql_available'] = self._pg_available
        stats['storage_dir'] = str(self.storage_dir)
        return stats


class DataPersistence(BasePersistence):
    """
    数据持久化管理器
    
    管理数据缓存、元数据的持久化存储。
    """
    
    def __init__(self, storage_dir: str = "data/persistence/data"):
        super().__init__(storage_dir)
        self.metadata_dir = self.storage_dir / "metadata"
        self.data_dir = self.storage_dir / "data"
        self.metadata_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
    
    def save_cache_entry(
        self,
        cache_key: str,
        data: Any,
        cache_type: str = "general",
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> bool:
        """
        保存缓存条目（PostgreSQL 优先）
        
        Args:
            cache_key: 缓存键
            data: 缓存数据
            cache_type: 缓存类型
            ttl_seconds: 过期时间（秒）
            tags: 标签列表
            description: 描述
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                metadata = CacheEntryMetadata(
                    cache_key=cache_key,
                    cache_type=cache_type,
                    data_size_bytes=self._estimate_size(data),
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
                    tags=tags or [],
                    description=description
                )
                
                if self._pg_available:
                    success = self._save_cache_to_postgresql(cache_key, data, metadata)
                    if success:
                        self._stats['postgresql_saves'] += 1
                        self._stats['total_saves'] += 1
                        return True
                    logger.warning("PostgreSQL 保存失败，降级到文件系统")
                
                success = self._save_cache_to_filesystem(cache_key, data, metadata)
                if success:
                    self._stats['filesystem_saves'] += 1
                    self._stats['total_saves'] += 1
                return success
                
            except Exception as e:
                logger.error(f"保存缓存条目失败 {cache_key}: {e}")
                return False
    
    def load_cache_entry(self, cache_key: str) -> Optional[Tuple[Any, CacheEntryMetadata]]:
        """
        加载缓存条目（优先从 PostgreSQL）
        
        Args:
            cache_key: 缓存键
        
        Returns:
            (数据, 元数据) 或 None
        """
        with self._lock:
            try:
                if self._pg_available:
                    result = self._load_cache_from_postgresql(cache_key)
                    if result:
                        data, metadata = result
                        if self._is_expired(metadata):
                            self.delete_cache_entry(cache_key)
                            self._stats['cache_misses'] += 1
                            return None
                        self._stats['cache_hits'] += 1
                        self._stats['total_loads'] += 1
                        return data, metadata
                
                result = self._load_cache_from_filesystem(cache_key)
                if result:
                    data, metadata = result
                    if self._is_expired(metadata):
                        self.delete_cache_entry(cache_key)
                        self._stats['cache_misses'] += 1
                        return None
                    self._stats['cache_hits'] += 1
                    self._stats['total_loads'] += 1
                    return data, metadata
                
                self._stats['cache_misses'] += 1
                return None
                
            except Exception as e:
                logger.error(f"加载缓存条目失败 {cache_key}: {e}")
                return None
    
    def delete_cache_entry(self, cache_key: str) -> bool:
        """
        删除缓存条目
        
        Args:
            cache_key: 缓存键
        
        Returns:
            是否删除成功
        """
        with self._lock:
            deleted = False
            
            if self._pg_available:
                deleted = self._delete_cache_from_postgresql(cache_key) or deleted
            
            deleted = self._delete_cache_from_filesystem(cache_key) or deleted
            
            return deleted
    
    def list_cache_entries(
        self,
        cache_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[CacheEntryMetadata]:
        """
        列出缓存条目
        
        Args:
            cache_type: 过滤类型
            tags: 过滤标签
        
        Returns:
            元数据列表
        """
        entries = []
        seen_keys = set()
        
        if self._pg_available:
            pg_entries = self._list_cache_from_postgresql(cache_type, tags)
            for entry in pg_entries:
                if entry.cache_key not in seen_keys:
                    entries.append(entry)
                    seen_keys.add(entry.cache_key)
        
        fs_entries = self._list_cache_from_filesystem(cache_type, tags)
        for entry in fs_entries:
            if entry.cache_key not in seen_keys:
                entries.append(entry)
                seen_keys.add(entry.cache_key)
        
        return entries
    
    def cleanup_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的条目数量
        """
        count = 0
        entries = self.list_cache_entries()
        
        for metadata in entries:
            if self._is_expired(metadata):
                if self.delete_cache_entry(metadata.cache_key):
                    count += 1
        
        logger.info(f"清理了 {count} 个过期缓存条目")
        return count
    
    def _save_cache_to_postgresql(
        self,
        cache_key: str,
        data: Any,
        metadata: CacheEntryMetadata
    ) -> bool:
        """保存缓存到 PostgreSQL"""
        conn = None
        try:
            import pickle
            conn = self._get_db_connection()
            if not conn:
                return False
            
            data_bytes = pickle.dumps(data)
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_cache_entries (
                        cache_key, cache_type, data, metadata, 
                        created_at, expires_at, access_count, tags, description
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        cache_type = EXCLUDED.cache_type,
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata,
                        expires_at = EXCLUDED.expires_at,
                        tags = EXCLUDED.tags,
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    cache_key,
                    metadata.cache_type,
                    data_bytes,
                    json.dumps({'data_size_bytes': metadata.data_size_bytes}),
                    metadata.created_at,
                    metadata.expires_at,
                    metadata.access_count,
                    json.dumps(metadata.tags),
                    metadata.description
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存缓存到 PostgreSQL 失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    def _load_cache_from_postgresql(
        self,
        cache_key: str
    ) -> Optional[Tuple[Any, CacheEntryMetadata]]:
        """从 PostgreSQL 加载缓存"""
        conn = None
        try:
            import pickle
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT cache_type, data, created_at, expires_at, 
                           access_count, tags, description
                    FROM data_cache_entries WHERE cache_key = %s
                """, (cache_key,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                cache_type, data_bytes, created_at, expires_at, access_count, tags, description = row
                
                data = pickle.loads(data_bytes)
                
                metadata = CacheEntryMetadata(
                    cache_key=cache_key,
                    cache_type=cache_type,
                    data_size_bytes=len(data_bytes),
                    created_at=created_at,
                    expires_at=expires_at,
                    access_count=access_count,
                    last_accessed=datetime.now(),
                    tags=json.loads(tags) if tags else [],
                    description=description or ""
                )
                
                cur.execute("""
                    UPDATE data_cache_entries 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = %s
                """, (cache_key,))
                conn.commit()
                
                return data, metadata
                
        except Exception as e:
            logger.error(f"从 PostgreSQL 加载缓存失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def _delete_cache_from_postgresql(self, cache_key: str) -> bool:
        """从 PostgreSQL 删除缓存"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("DELETE FROM data_cache_entries WHERE cache_key = %s", (cache_key,))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"从 PostgreSQL 删除缓存失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    def _list_cache_from_postgresql(
        self,
        cache_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[CacheEntryMetadata]:
        """从 PostgreSQL 列出缓存"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                query = """
                    SELECT cache_key, cache_type, created_at, expires_at,
                           access_count, tags, description
                    FROM data_cache_entries WHERE 1=1
                """
                params = []
                
                if cache_type:
                    query += " AND cache_type = %s"
                    params.append(cache_type)
                
                if tags:
                    query += " AND tags ?| %s"
                    params.append(tags)
                
                query += " ORDER BY created_at DESC"
                
                cur.execute(query, params)
                
                entries = []
                for row in cur.fetchall():
                    cache_key, ct, created_at, expires_at, access_count, tags_json, description = row
                    entries.append(CacheEntryMetadata(
                        cache_key=cache_key,
                        cache_type=ct,
                        data_size_bytes=0,
                        created_at=created_at,
                        expires_at=expires_at,
                        access_count=access_count,
                        tags=json.loads(tags_json) if tags_json else [],
                        description=description or ""
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"从 PostgreSQL 列出缓存失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _save_cache_to_filesystem(
        self,
        cache_key: str,
        data: Any,
        metadata: CacheEntryMetadata
    ) -> bool:
        """保存缓存到文件系统"""
        try:
            import pickle
            
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            data_file = self.data_dir / f"{key_hash}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
            
            metadata_file = self.metadata_dir / f"{key_hash}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"保存缓存到文件系统失败: {e}")
            return False
    
    def _load_cache_from_filesystem(
        self,
        cache_key: str
    ) -> Optional[Tuple[Any, CacheEntryMetadata]]:
        """从文件系统加载缓存"""
        try:
            import pickle
            
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            data_file = self.data_dir / f"{key_hash}.pkl"
            metadata_file = self.metadata_dir / f"{key_hash}.json"
            
            if not data_file.exists() or not metadata_file.exists():
                return None
            
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            metadata = CacheEntryMetadata(
                cache_key=metadata_dict['cache_key'],
                cache_type=metadata_dict['cache_type'],
                data_size_bytes=metadata_dict.get('data_size_bytes', 0),
                created_at=datetime.fromisoformat(metadata_dict['created_at']) if isinstance(metadata_dict['created_at'], str) else metadata_dict['created_at'],
                expires_at=datetime.fromisoformat(metadata_dict['expires_at']) if metadata_dict.get('expires_at') and isinstance(metadata_dict['expires_at'], str) else metadata_dict.get('expires_at'),
                access_count=metadata_dict.get('access_count', 0),
                last_accessed=datetime.now(),
                tags=metadata_dict.get('tags', []),
                description=metadata_dict.get('description', '')
            )
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"从文件系统加载缓存失败: {e}")
            return None
    
    def _delete_cache_from_filesystem(self, cache_key: str) -> bool:
        """从文件系统删除缓存"""
        try:
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            data_file = self.data_dir / f"{key_hash}.pkl"
            metadata_file = self.metadata_dir / f"{key_hash}.json"
            
            deleted = False
            if data_file.exists():
                data_file.unlink()
                deleted = True
            if metadata_file.exists():
                metadata_file.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"从文件系统删除缓存失败: {e}")
            return False
    
    def _list_cache_from_filesystem(
        self,
        cache_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[CacheEntryMetadata]:
        """从文件系统列出缓存"""
        entries = []
        
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    
                    if cache_type and metadata_dict.get('cache_type') != cache_type:
                        continue
                    
                    if tags and not any(tag in metadata_dict.get('tags', []) for tag in tags):
                        continue
                    
                    entries.append(CacheEntryMetadata(
                        cache_key=metadata_dict['cache_key'],
                        cache_type=metadata_dict['cache_type'],
                        data_size_bytes=metadata_dict.get('data_size_bytes', 0),
                        created_at=datetime.fromisoformat(metadata_dict['created_at']) if isinstance(metadata_dict['created_at'], str) else metadata_dict['created_at'],
                        expires_at=datetime.fromisoformat(metadata_dict['expires_at']) if metadata_dict.get('expires_at') and isinstance(metadata_dict['expires_at'], str) else metadata_dict.get('expires_at'),
                        access_count=metadata_dict.get('access_count', 0),
                        tags=metadata_dict.get('tags', []),
                        description=metadata_dict.get('description', '')
                    ))
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"从文件系统列出缓存失败: {e}")
        
        return entries
    
    def _is_expired(self, metadata: CacheEntryMetadata) -> bool:
        """检查缓存是否过期"""
        if metadata.expires_at is None:
            return False
        return datetime.now() > metadata.expires_at
    
    def _estimate_size(self, data: Any) -> int:
        """估算数据大小"""
        try:
            import pickle
            return len(pickle.dumps(data))
        except Exception:
            return 0
    
    def save(self, *args, **kwargs) -> bool:
        """保存数据（兼容基类接口）"""
        return self.save_cache_entry(*args, **kwargs)
    
    def load(self, *args, **kwargs) -> Any:
        """加载数据（兼容基类接口）"""
        return self.load_cache_entry(*args, **kwargs)


class CachePersistence(DataPersistence):
    """
    缓存持久化管理器
    
    继承 DataPersistence，提供缓存特定的功能。
    """
    
    def __init__(self, storage_dir: str = "data/persistence/cache"):
        super().__init__(storage_dir)
    
    def get_or_set(
        self,
        cache_key: str,
        loader_func: callable,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        获取缓存或设置缓存
        
        Args:
            cache_key: 缓存键
            loader_func: 数据加载函数
            ttl_seconds: 过期时间
            **kwargs: 传递给 loader_func 的参数
        
        Returns:
            缓存数据或新加载的数据
        """
        result = self.load_cache_entry(cache_key)
        if result:
            return result[0]
        
        data = loader_func(**kwargs)
        self.save_cache_entry(cache_key, data, ttl_seconds=ttl_seconds)
        return data


class LineagePersistence(BasePersistence):
    """
    数据血缘持久化管理器
    """
    
    def __init__(self, storage_dir: str = "data/persistence/lineage"):
        super().__init__(storage_dir)
    
    def save_lineage(
        self,
        data_type: str,
        source_info: Dict[str, Any],
        transform_info: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> Optional[int]:
        """
        保存数据血缘记录
        
        Args:
            data_type: 数据类型
            source_info: 来源信息
            transform_info: 转换信息
            dependencies: 依赖列表
        
        Returns:
            记录ID或None
        """
        with self._lock:
            try:
                record = LineageRecord(
                    data_type=data_type,
                    source_info=source_info or {},
                    transform_info=transform_info or {},
                    dependencies=dependencies or [],
                    created_at=datetime.now()
                )
                
                if self._pg_available:
                    record_id = self._save_lineage_to_postgresql(record)
                    if record_id:
                        self._stats['total_saves'] += 1
                        return record_id
                
                self._save_lineage_to_filesystem(record)
                self._stats['total_saves'] += 1
                return None
                
            except Exception as e:
                logger.error(f"保存血缘记录失败: {e}")
                return None
    
    def get_lineage(self, data_type: str, limit: int = 100) -> List[LineageRecord]:
        """
        获取数据血缘记录
        
        Args:
            data_type: 数据类型
            limit: 返回数量限制
        
        Returns:
            血缘记录列表
        """
        records = []
        
        if self._pg_available:
            records = self._get_lineage_from_postgresql(data_type, limit)
        
        if not records:
            records = self._get_lineage_from_filesystem(data_type, limit)
        
        self._stats['total_loads'] += 1
        return records
    
    def _save_lineage_to_postgresql(self, record: LineageRecord) -> Optional[int]:
        """保存血缘记录到 PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_lineage_records (
                        data_type, source_info, transform_info, dependencies, created_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    record.data_type,
                    json.dumps(record.source_info),
                    json.dumps(record.transform_info),
                    json.dumps(record.dependencies),
                    record.created_at
                ))
                
                record_id = cur.fetchone()[0]
            
            conn.commit()
            return record_id
            
        except Exception as e:
            logger.error(f"保存血缘记录到 PostgreSQL 失败: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    def _get_lineage_from_postgresql(self, data_type: str, limit: int) -> List[LineageRecord]:
        """从 PostgreSQL 获取血缘记录"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, data_type, source_info, transform_info, dependencies, created_at
                    FROM data_lineage_records
                    WHERE data_type = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (data_type, limit))
                
                records = []
                for row in cur.fetchall():
                    records.append(LineageRecord(
                        id=row[0],
                        data_type=row[1],
                        source_info=json.loads(row[2]) if row[2] else {},
                        transform_info=json.loads(row[3]) if row[3] else {},
                        dependencies=json.loads(row[4]) if row[4] else [],
                        created_at=row[5]
                    ))
                
                return records
                
        except Exception as e:
            logger.error(f"从 PostgreSQL 获取血缘记录失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _save_lineage_to_filesystem(self, record: LineageRecord) -> bool:
        """保存血缘记录到文件系统"""
        try:
            record_file = self.storage_dir / f"lineage_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(record), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"保存血缘记录到文件系统失败: {e}")
            return False
    
    def _get_lineage_from_filesystem(self, data_type: str, limit: int) -> List[LineageRecord]:
        """从文件系统获取血缘记录"""
        records = []
        try:
            for record_file in sorted(self.storage_dir.glob("lineage_*.json"), reverse=True)[:limit]:
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        record_dict = json.load(f)
                    
                    if record_dict.get('data_type') == data_type:
                        records.append(LineageRecord(
                            id=record_dict.get('id'),
                            data_type=record_dict['data_type'],
                            source_info=record_dict.get('source_info', {}),
                            transform_info=record_dict.get('transform_info', {}),
                            dependencies=record_dict.get('dependencies', []),
                            created_at=datetime.fromisoformat(record_dict['created_at']) if isinstance(record_dict['created_at'], str) else record_dict['created_at']
                        ))
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"从文件系统获取血缘记录失败: {e}")
        
        return records[:limit]
    
    def save(self, *args, **kwargs) -> bool:
        """保存数据（兼容基类接口）"""
        return self.save_lineage(*args, **kwargs) is not None
    
    def load(self, *args, **kwargs) -> Any:
        """加载数据（兼容基类接口）"""
        return self.get_lineage(*args, **kwargs)


class QualityPersistence(BasePersistence):
    """
    数据质量持久化管理器
    """
    
    def __init__(self, storage_dir: str = "data/persistence/quality"):
        super().__init__(storage_dir)
    
    def save_quality_metric(
        self,
        data_type: str,
        completeness: float,
        accuracy: float,
        timeliness: float,
        consistency: float,
        uniqueness: float,
        issues: List[str] = None
    ) -> Optional[int]:
        """
        保存数据质量指标
        
        Args:
            data_type: 数据类型
            completeness: 完整性
            accuracy: 准确性
            timeliness: 及时性
            consistency: 一致性
            uniqueness: 唯一性
            issues: 问题列表
        
        Returns:
            记录ID或None
        """
        with self._lock:
            try:
                record = QualityMetricRecord(
                    data_type=data_type,
                    completeness=completeness,
                    accuracy=accuracy,
                    timeliness=timeliness,
                    consistency=consistency,
                    uniqueness=uniqueness,
                    checked_at=datetime.now(),
                    issues=issues or []
                )
                
                if self._pg_available:
                    record_id = self._save_quality_to_postgresql(record)
                    if record_id:
                        self._stats['total_saves'] += 1
                        return record_id
                
                self._save_quality_to_filesystem(record)
                self._stats['total_saves'] += 1
                return None
                
            except Exception as e:
                logger.error(f"保存质量指标失败: {e}")
                return None
    
    def get_quality_history(
        self,
        data_type: str,
        days: int = 7,
        limit: int = 100
    ) -> List[QualityMetricRecord]:
        """
        获取质量指标历史
        
        Args:
            data_type: 数据类型
            days: 查询天数
            limit: 返回数量限制
        
        Returns:
            质量指标记录列表
        """
        records = []
        
        if self._pg_available:
            records = self._get_quality_from_postgresql(data_type, days, limit)
        
        if not records:
            records = self._get_quality_from_filesystem(data_type, days, limit)
        
        self._stats['total_loads'] += 1
        return records
    
    def _save_quality_to_postgresql(self, record: QualityMetricRecord) -> Optional[int]:
        """保存质量指标到 PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_quality_metrics (
                        data_type, completeness, accuracy, timeliness, 
                        consistency, uniqueness, checked_at, issues
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    record.data_type,
                    record.completeness,
                    record.accuracy,
                    record.timeliness,
                    record.consistency,
                    record.uniqueness,
                    record.checked_at,
                    json.dumps(record.issues)
                ))
                
                record_id = cur.fetchone()[0]
            
            conn.commit()
            return record_id
            
        except Exception as e:
            logger.error(f"保存质量指标到 PostgreSQL 失败: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    def _get_quality_from_postgresql(
        self,
        data_type: str,
        days: int,
        limit: int
    ) -> List[QualityMetricRecord]:
        """从 PostgreSQL 获取质量指标"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, data_type, completeness, accuracy, timeliness,
                           consistency, uniqueness, checked_at, issues
                    FROM data_quality_metrics
                    WHERE data_type = %s
                      AND checked_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                    ORDER BY checked_at DESC
                    LIMIT %s
                """, (data_type, days, limit))
                
                records = []
                for row in cur.fetchall():
                    records.append(QualityMetricRecord(
                        id=row[0],
                        data_type=row[1],
                        completeness=row[2],
                        accuracy=row[3],
                        timeliness=row[4],
                        consistency=row[5],
                        uniqueness=row[6],
                        checked_at=row[7],
                        issues=json.loads(row[8]) if row[8] else []
                    ))
                
                return records
                
        except Exception as e:
            logger.error(f"从 PostgreSQL 获取质量指标失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _save_quality_to_filesystem(self, record: QualityMetricRecord) -> bool:
        """保存质量指标到文件系统"""
        try:
            record_file = self.storage_dir / f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(record), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"保存质量指标到文件系统失败: {e}")
            return False
    
    def _get_quality_from_filesystem(
        self,
        data_type: str,
        days: int,
        limit: int
    ) -> List[QualityMetricRecord]:
        """从文件系统获取质量指标"""
        records = []
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            for record_file in sorted(self.storage_dir.glob("quality_*.json"), reverse=True):
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        record_dict = json.load(f)
                    
                    checked_at = record_dict.get('checked_at')
                    if isinstance(checked_at, str):
                        checked_at = datetime.fromisoformat(checked_at)
                    
                    if checked_at < cutoff:
                        continue
                    
                    if record_dict.get('data_type') == data_type:
                        records.append(QualityMetricRecord(
                            id=record_dict.get('id'),
                            data_type=record_dict['data_type'],
                            completeness=record_dict.get('completeness', 0),
                            accuracy=record_dict.get('accuracy', 0),
                            timeliness=record_dict.get('timeliness', 0),
                            consistency=record_dict.get('consistency', 0),
                            uniqueness=record_dict.get('uniqueness', 0),
                            checked_at=checked_at,
                            issues=record_dict.get('issues', [])
                        ))
                        
                        if len(records) >= limit:
                            break
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"从文件系统获取质量指标失败: {e}")
        
        return records
    
    def save(self, *args, **kwargs) -> bool:
        """保存数据（兼容基类接口）"""
        return self.save_quality_metric(*args, **kwargs) is not None
    
    def load(self, *args, **kwargs) -> Any:
        """加载数据（兼容基类接口）"""
        return self.get_quality_history(*args, **kwargs)


class CompliancePersistence(BasePersistence):
    """
    合规检查持久化管理器
    """
    
    def __init__(self, storage_dir: str = "data/persistence/compliance"):
        super().__init__(storage_dir)
    
    def save_compliance_check(
        self,
        policy_id: str,
        data_type: str,
        is_compliant: bool,
        issues: List[str] = None,
        check_duration_ms: float = 0.0
    ) -> Optional[int]:
        """
        保存合规检查记录
        
        Args:
            policy_id: 策略ID
            data_type: 数据类型
            is_compliant: 是否合规
            issues: 问题列表
            check_duration_ms: 检查耗时（毫秒）
        
        Returns:
            记录ID或None
        """
        with self._lock:
            try:
                record = ComplianceCheckRecord(
                    policy_id=policy_id,
                    data_type=data_type,
                    is_compliant=is_compliant,
                    issues=issues or [],
                    checked_at=datetime.now(),
                    check_duration_ms=check_duration_ms
                )
                
                if self._pg_available:
                    record_id = self._save_compliance_to_postgresql(record)
                    if record_id:
                        self._stats['total_saves'] += 1
                        return record_id
                
                self._save_compliance_to_filesystem(record)
                self._stats['total_saves'] += 1
                return None
                
            except Exception as e:
                logger.error(f"保存合规检查记录失败: {e}")
                return None
    
    def get_compliance_history(
        self,
        policy_id: str = None,
        data_type: str = None,
        days: int = 30,
        limit: int = 100
    ) -> List[ComplianceCheckRecord]:
        """
        获取合规检查历史
        
        Args:
            policy_id: 策略ID（可选）
            data_type: 数据类型（可选）
            days: 查询天数
            limit: 返回数量限制
        
        Returns:
            合规检查记录列表
        """
        records = []
        
        if self._pg_available:
            records = self._get_compliance_from_postgresql(policy_id, data_type, days, limit)
        
        if not records:
            records = self._get_compliance_from_filesystem(policy_id, data_type, days, limit)
        
        self._stats['total_loads'] += 1
        return records
    
    def _save_compliance_to_postgresql(self, record: ComplianceCheckRecord) -> Optional[int]:
        """保存合规检查记录到 PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_compliance_checks (
                        policy_id, data_type, is_compliant, issues, checked_at, check_duration_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    record.policy_id,
                    record.data_type,
                    record.is_compliant,
                    json.dumps(record.issues),
                    record.checked_at,
                    record.check_duration_ms
                ))
                
                record_id = cur.fetchone()[0]
            
            conn.commit()
            return record_id
            
        except Exception as e:
            logger.error(f"保存合规检查记录到 PostgreSQL 失败: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    def _get_compliance_from_postgresql(
        self,
        policy_id: Optional[str],
        data_type: Optional[str],
        days: int,
        limit: int
    ) -> List[ComplianceCheckRecord]:
        """从 PostgreSQL 获取合规检查记录"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                query = """
                    SELECT id, policy_id, data_type, is_compliant, issues, checked_at, check_duration_ms
                    FROM data_compliance_checks
                    WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                """
                params = [days]
                
                if policy_id:
                    query += " AND policy_id = %s"
                    params.append(policy_id)
                
                if data_type:
                    query += " AND data_type = %s"
                    params.append(data_type)
                
                query += " ORDER BY checked_at DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                
                records = []
                for row in cur.fetchall():
                    records.append(ComplianceCheckRecord(
                        id=row[0],
                        policy_id=row[1],
                        data_type=row[2],
                        is_compliant=row[3],
                        issues=json.loads(row[4]) if row[4] else [],
                        checked_at=row[5],
                        check_duration_ms=row[6]
                    ))
                
                return records
                
        except Exception as e:
            logger.error(f"从 PostgreSQL 获取合规检查记录失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _save_compliance_to_filesystem(self, record: ComplianceCheckRecord) -> bool:
        """保存合规检查记录到文件系统"""
        try:
            record_file = self.storage_dir / f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(record), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"保存合规检查记录到文件系统失败: {e}")
            return False
    
    def _get_compliance_from_filesystem(
        self,
        policy_id: Optional[str],
        data_type: Optional[str],
        days: int,
        limit: int
    ) -> List[ComplianceCheckRecord]:
        """从文件系统获取合规检查记录"""
        records = []
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            for record_file in sorted(self.storage_dir.glob("compliance_*.json"), reverse=True):
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        record_dict = json.load(f)
                    
                    checked_at = record_dict.get('checked_at')
                    if isinstance(checked_at, str):
                        checked_at = datetime.fromisoformat(checked_at)
                    
                    if checked_at < cutoff:
                        continue
                    
                    if policy_id and record_dict.get('policy_id') != policy_id:
                        continue
                    
                    if data_type and record_dict.get('data_type') != data_type:
                        continue
                    
                    records.append(ComplianceCheckRecord(
                        id=record_dict.get('id'),
                        policy_id=record_dict['policy_id'],
                        data_type=record_dict['data_type'],
                        is_compliant=record_dict['is_compliant'],
                        issues=record_dict.get('issues', []),
                        checked_at=checked_at,
                        check_duration_ms=record_dict.get('check_duration_ms', 0)
                    ))
                    
                    if len(records) >= limit:
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"从文件系统获取合规检查记录失败: {e}")
        
        return records
    
    def save(self, *args, **kwargs) -> bool:
        """保存数据（兼容基类接口）"""
        return self.save_compliance_check(*args, **kwargs) is not None
    
    def load(self, *args, **kwargs) -> Any:
        """加载数据（兼容基类接口）"""
        return self.get_compliance_history(*args, **kwargs)


# ============================================================================
# 全局实例管理
# ============================================================================

_data_persistence: Optional[DataPersistence] = None
_cache_persistence: Optional[CachePersistence] = None
_lineage_persistence: Optional[LineagePersistence] = None
_quality_persistence: Optional[QualityPersistence] = None
_compliance_persistence: Optional[CompliancePersistence] = None


def get_data_persistence() -> DataPersistence:
    """获取数据持久化管理器单例"""
    global _data_persistence
    if _data_persistence is None:
        _data_persistence = DataPersistence()
    return _data_persistence


def get_cache_persistence() -> CachePersistence:
    """获取缓存持久化管理器单例"""
    global _cache_persistence
    if _cache_persistence is None:
        _cache_persistence = CachePersistence()
    return _cache_persistence


def get_lineage_persistence() -> LineagePersistence:
    """获取血缘持久化管理器单例"""
    global _lineage_persistence
    if _lineage_persistence is None:
        _lineage_persistence = LineagePersistence()
    return _lineage_persistence


def get_quality_persistence() -> QualityPersistence:
    """获取质量持久化管理器单例"""
    global _quality_persistence
    if _quality_persistence is None:
        _quality_persistence = QualityPersistence()
    return _quality_persistence


def get_compliance_persistence() -> CompliancePersistence:
    """获取合规持久化管理器单例"""
    global _compliance_persistence
    if _compliance_persistence is None:
        _compliance_persistence = CompliancePersistence()
    return _compliance_persistence
