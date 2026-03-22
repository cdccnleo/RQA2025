#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征存储管理器

支持PostgreSQL优先存储策略：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统 (pickle/parquet)

使用方式:
    store = FeatureStore()
    store.store_feature("sma_20", data, config)
    loaded, metadata = store.load_feature("sma_20", params)
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import threading

import pandas as pd
import numpy as np

from .config import FeatureRegistrationConfig
from .config_integration import get_config_integration_manager, ConfigScope

# 导入存储配额管理器
try:
    from src.infrastructure.persistence.storage_quota_manager import (
        StorageQuotaManager, StorageQuotaConfig, get_storage_quota_manager
    )
    STORAGE_QUOTA_AVAILABLE = True
except ImportError:
    STORAGE_QUOTA_AVAILABLE = False
    StorageQuotaManager = None
    StorageQuotaConfig = None
    get_storage_quota_manager = None

# 导入版本管理器
try:
    from .version_management import FeatureVersionManager, FeatureVersion
    VERSION_MANAGEMENT_AVAILABLE = True
except ImportError:
    VERSION_MANAGEMENT_AVAILABLE = False
    FeatureVersionManager = None
    FeatureVersion = None

# 导入特征血缘追踪器
try:
    from .feature_lineage import FeatureLineageTracker, get_lineage_tracker
    LINEAGE_TRACKING_AVAILABLE = True
except ImportError:
    LINEAGE_TRACKING_AVAILABLE = False
    FeatureLineageTracker = None
    get_lineage_tracker = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """特征元数据"""
    feature_name: str
    feature_type: str
    params: Dict[str, Any]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime
    data_shape: Tuple[int, int]
    data_size_mb: float
    checksum: str
    version: str = "1.0"
    description: str = ""
    tags: List[str] = None
    storage_type: str = "filesystem"
    feature_id: str = ""


@dataclass
class StoreConfig:
    """存储配置"""
    base_path: str = "./feature_cache"
    max_size_mb: int = 1024
    ttl_hours: int = 24
    compression: bool = True
    use_filesystem: bool = True
    max_workers: int = 4
    enable_postgresql: bool = True


class FeatureStore:
    """
    特征存储管理器
    
    实现PostgreSQL优先存储，数据库连接失败时自动降级到文件系统。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, config: StoreConfig = None):
        self.config_manager = get_config_integration_manager()
        store_config = self.config_manager.get_config(ConfigScope.PROCESSING)
        
        if store_config:
            config = config or StoreConfig()
            for key in ['base_path', 'max_size_mb', 'ttl_hours', 'compression', 
                       'use_filesystem', 'max_workers', 'enable_postgresql']:
                if key in store_config and store_config[key] is not None:
                    setattr(config, key, store_config[key])
        
        self.config = config or StoreConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata"
        self.data_path = self.base_path / "data"
        for path in [self.metadata_path, self.data_path]:
            path.mkdir(exist_ok=True)
        
        self._lock = threading.Lock()
        
        self._pg_config = None
        self._pg_available = False
        
        if self.config.enable_postgresql:
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self._ensure_tables_exist()
                logger.info("✅ FeatureStore: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ FeatureStore: PostgreSQL 不可用，使用文件系统存储")
        
        self.stats = {
            'total_stored': 0,
            'total_loaded': 0,
            'total_deleted': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'postgresql_stores': 0,
            'filesystem_stores': 0
        }
        
        # 初始化存储配额管理器
        self._quota_manager = None
        if STORAGE_QUOTA_AVAILABLE and get_storage_quota_manager:
            try:
                quota_config = StorageQuotaConfig(
                    max_storage_size=10 * 1024 * 1024 * 1024,  # 10GB
                    max_file_count=10000,
                    auto_cleanup_enabled=True,
                    cleanup_threshold_percent=80.0,
                    min_file_age_days=7,
                    max_file_age_days=90
                )
                self._quota_manager = get_storage_quota_manager(
                    str(self.storage_dir), 
                    quota_config
                )
                logger.info("✅ FeatureStore: 存储配额管理已启用")
            except Exception as e:
                logger.warning(f"⚠️ FeatureStore: 存储配额管理初始化失败: {e}")
        
        # 初始化版本管理器
        self._version_manager = None
        if VERSION_MANAGEMENT_AVAILABLE and FeatureVersionManager:
            try:
                self._version_manager = FeatureVersionManager(str(self.storage_dir / "versions"))
                logger.info("✅ FeatureStore: 版本管理已启用")
            except Exception as e:
                logger.warning(f"⚠️ FeatureStore: 版本管理初始化失败: {e}")

        # 初始化特征血缘追踪器
        self._lineage_tracker = None
        if LINEAGE_TRACKING_AVAILABLE and get_lineage_tracker:
            try:
                lineage_storage_dir = str(self.base_path / "lineage")
                self._lineage_tracker = get_lineage_tracker(lineage_storage_dir)
                logger.info("✅ FeatureStore: 特征血缘追踪已启用")
            except Exception as e:
                logger.warning(f"⚠️ FeatureStore: 特征血缘追踪初始化失败: {e}")

        self.config_manager.register_config_watcher(
            ConfigScope.PROCESSING, self._on_config_change
        )

    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """获取PostgreSQL配置（统一使用 database_config 模块）
        
        所有数据库配置统一从 database_config 模块获取，确保配置一致性。
        不再在各地硬编码配置或密码。
        """
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
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
                        feature_id VARCHAR(64) PRIMARY KEY,
                        feature_name VARCHAR(255) NOT NULL,
                        feature_type VARCHAR(50) NOT NULL,
                        params JSONB DEFAULT '{}',
                        dependencies JSONB DEFAULT '[]',
                        data BYTEA,
                        data_shape JSONB,
                        data_size_mb FLOAT DEFAULT 0,
                        checksum VARCHAR(64),
                        version VARCHAR(20) DEFAULT '1.0',
                        description TEXT,
                        tags JSONB DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_name 
                    ON feature_cache(feature_name)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_type 
                    ON feature_cache(feature_type)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_updated 
                    ON feature_cache(updated_at DESC)
                """)
            
            conn.commit()
            logger.debug("特征缓存表已确保存在")
            
        except Exception as e:
            logger.error(f"创建特征缓存表失败: {e}")
        finally:
            if conn:
                conn.close()

    def _on_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):
        if scope == ConfigScope.PROCESSING:
            if hasattr(self.config, key):
                setattr(self.config, key, new_value)
                if key == 'base_path':
                    self.base_path = Path(new_value)
                    self.base_path.mkdir(parents=True, exist_ok=True)
                    self.metadata_path = self.base_path / "metadata"
                    self.data_path = self.base_path / "data"
                    for path in [self.metadata_path, self.data_path]:
                        path.mkdir(exist_ok=True)

    def store_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        config: FeatureRegistrationConfig,
        description: str = "",
        tags: List[str] = None
    ) -> bool:
        """
        存储特征数据（PostgreSQL优先，降级到文件系统）
        
        Args:
            feature_name: 特征名称
            data: 特征数据
            config: 特征配置
            description: 特征描述
            tags: 特征标签
            
        Returns:
            存储是否成功
        """
        try:
            with self._lock:
                feature_id = self._generate_feature_id(feature_name, config.params)
                
                if self._feature_exists(feature_id):
                    logger.info(f"特征 {feature_name} 已存在，更新中...")
                    return self._update_feature(feature_id, data, config, description, tags)
                
                metadata = FeatureMetadata(
                    feature_id=feature_id,
                    feature_name=feature_name,
                    feature_type=config.feature_type.value,
                    params=config.params,
                    dependencies=config.dependencies,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    data_shape=data.shape,
                    data_size_mb=data.memory_usage(deep=True).sum() / 1024 / 1024,
                    checksum=self._calculate_checksum(data),
                    description=description,
                    tags=tags or [],
                    storage_type="filesystem"
                )
                
                if self._pg_available:
                    success = self._save_to_postgresql(feature_id, data, metadata)
                    if success:
                        metadata.storage_type = "postgresql"
                        self.stats['postgresql_stores'] += 1
                        self.stats['total_stored'] += 1
                        logger.info(f"✅ 特征 {feature_name} 已存储到PostgreSQL，ID: {feature_id}")

                        # 注册特征血缘关系
                        self._register_feature_lineage(feature_name, config, metadata)

                        return True
                    else:
                        logger.warning("PostgreSQL保存失败，降级到文件系统")
                
                success = self._save_feature_data(feature_id, data)
                if not success:
                    return False
                
                success = self._save_metadata(feature_id, metadata)
                if not success:
                    self._delete_feature_data(feature_id)
                    return False
                
                self.stats['filesystem_stores'] += 1
                self.stats['total_stored'] += 1
                logger.info(f"✅ 特征 {feature_name} 已存储到文件系统，ID: {feature_id}")

                # 注册特征血缘关系
                self._register_feature_lineage(feature_name, config, metadata)

                return True

        except Exception as e:
            logger.error(f"存储特征 {feature_name} 失败: {e}")
            return False

    def _save_to_postgresql(
        self, 
        feature_id: str,
        data: pd.DataFrame, 
        metadata: FeatureMetadata
    ) -> bool:
        """保存特征到PostgreSQL"""
        conn = None
        try:
            import io
            
            conn = self._get_db_connection()
            if not conn:
                return False
            
            buffer = io.BytesIO()
            if self.config.compression:
                data.to_pickle(buffer, compression='gzip')
            else:
                data.to_pickle(buffer)
            
            data_bytes = buffer.getvalue()
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_cache (
                        feature_id, feature_name, feature_type, params, dependencies,
                        data, data_shape, data_size_mb, checksum, version,
                        description, tags, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (feature_id) DO UPDATE SET
                        feature_name = EXCLUDED.feature_name,
                        feature_type = EXCLUDED.feature_type,
                        params = EXCLUDED.params,
                        dependencies = EXCLUDED.dependencies,
                        data = EXCLUDED.data,
                        data_shape = EXCLUDED.data_shape,
                        data_size_mb = EXCLUDED.data_size_mb,
                        checksum = EXCLUDED.checksum,
                        description = EXCLUDED.description,
                        tags = EXCLUDED.tags,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    feature_id,
                    metadata.feature_name,
                    metadata.feature_type,
                    json.dumps(metadata.params),
                    json.dumps(metadata.dependencies),
                    data_bytes,
                    json.dumps(list(metadata.data_shape)),
                    metadata.data_size_mb,
                    metadata.checksum,
                    metadata.version,
                    metadata.description,
                    json.dumps(metadata.tags),
                    metadata.created_at,
                    metadata.updated_at
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

    def load_feature(
        self,
        feature_name: str,
        params: Dict[str, Any] = None
    ) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """
        加载特征数据（优先从PostgreSQL，降级到文件系统）
        
        Args:
            feature_name: 特征名称
            params: 特征参数
            
        Returns:
            (特征数据, 元数据) 或 None
        """
        try:
            feature_id = self._generate_feature_id(feature_name, params or {})
            
            if self._pg_available:
                result = self._load_from_postgresql(feature_id)
                if result is not None:
                    data, metadata = result
                    if self._is_expired(metadata):
                        logger.info(f"特征 {feature_name} 已过期，删除中...")
                        self.delete_feature(feature_id)
                        self.stats['cache_misses'] += 1
                        return None
                    
                    self.stats['cache_hits'] += 1
                    self.stats['total_loaded'] += 1
                    logger.info(f"从PostgreSQL加载特征 {feature_name} 成功")
                    return data, metadata
            
            if not self._feature_exists(feature_id):
                self.stats['cache_misses'] += 1
                logger.info(f"特征 {feature_name} 不存在")
                return None
            
            metadata = self._load_metadata(feature_id)
            if metadata and self._is_expired(metadata):
                logger.info(f"特征 {feature_name} 已过期，删除中...")
                self.delete_feature(feature_id)
                self.stats['cache_misses'] += 1
                return None
            
            data = self._load_feature_data(feature_id)
            if data is None:
                return None
            
            self.stats['cache_hits'] += 1
            self.stats['total_loaded'] += 1
            
            logger.info(f"从文件系统加载特征 {feature_name} 成功")
            return data, metadata
            
        except Exception as e:
            logger.error(f"加载特征 {feature_name} 失败: {e}")
            return None

    def _load_from_postgresql(
        self, 
        feature_id: str
    ) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """从PostgreSQL加载特征"""
        conn = None
        try:
            import io
            
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feature_name, feature_type, params, dependencies,
                           data, data_shape, data_size_mb, checksum, version,
                           description, tags, created_at, updated_at
                    FROM feature_cache WHERE feature_id = %s
                """, (feature_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                (feature_name, feature_type, params, dependencies,
                 data_bytes, data_shape, data_size_mb, checksum, version,
                 description, tags, created_at, updated_at) = row
                
                buffer = io.BytesIO(data_bytes)
                if self.config.compression:
                    data = pd.read_pickle(buffer, compression='gzip')
                else:
                    data = pd.read_pickle(buffer)
                
                metadata = FeatureMetadata(
                    feature_id=feature_id,
                    feature_name=feature_name,
                    feature_type=feature_type,
                    params=json.loads(params) if params else {},
                    dependencies=json.loads(dependencies) if dependencies else [],
                    created_at=created_at,
                    updated_at=updated_at,
                    data_shape=tuple(json.loads(data_shape)) if data_shape else (0, 0),
                    data_size_mb=data_size_mb or 0,
                    checksum=checksum or "",
                    version=version or "1.0",
                    description=description or "",
                    tags=json.loads(tags) if tags else [],
                    storage_type="postgresql"
                )
                
                return data, metadata
                
        except Exception as e:
            logger.error(f"从PostgreSQL加载失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def delete_feature(self, feature_id: str) -> bool:
        """删除特征"""
        try:
            with self._lock:
                deleted = False
                
                if self._pg_available:
                    deleted = self._delete_from_postgresql(feature_id) or deleted
                
                deleted = self._delete_from_filesystem(feature_id) or deleted
                
                if deleted:
                    self.stats['total_deleted'] += 1
                    logger.info(f"特征 {feature_id} 删除成功")
                
                return deleted
                
        except Exception as e:
            logger.error(f"删除特征 {feature_id} 失败: {e}")
            return False

    def _delete_from_postgresql(self, feature_id: str) -> bool:
        """从PostgreSQL删除特征"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("DELETE FROM feature_cache WHERE feature_id = %s", (feature_id,))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"从PostgreSQL删除失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _delete_from_filesystem(self, feature_id: str) -> bool:
        """从文件系统删除特征"""
        try:
            deleted = False
            
            data_file = self.data_path / f"{feature_id}.pkl"
            if data_file.exists():
                data_file.unlink()
                deleted = True
            
            metadata_file = self.metadata_path / f"{feature_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"从文件系统删除失败: {e}")
            return False

    def list_features(
        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """列出特征"""
        features = []
        seen_ids = set()
        
        if self._pg_available:
            pg_features = self._list_from_postgresql(feature_type, tags)
            for f in pg_features:
                if f.feature_id not in seen_ids:
                    features.append(f)
                    seen_ids.add(f.feature_id)
        
        fs_features = self._list_features_filesystem(feature_type, tags)
        for f in fs_features:
            if f.feature_id not in seen_ids:
                features.append(f)
                seen_ids.add(f.feature_id)
        
        return features

    def _list_from_postgresql(
        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """从PostgreSQL列出特征"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                query = """
                    SELECT feature_id, feature_name, feature_type, params, dependencies,
                           data_shape, data_size_mb, checksum, version,
                           description, tags, created_at, updated_at
                    FROM feature_cache
                    WHERE 1=1
                """
                params = []
                
                if feature_type:
                    query += " AND feature_type = %s"
                    params.append(feature_type)
                
                if tags:
                    query += " AND tags ?| %s"
                    params.append(tags)
                
                query += " ORDER BY updated_at DESC"
                
                cur.execute(query, params)
                
                features = []
                for row in cur.fetchall():
                    (feature_id, feature_name, ft, p, deps,
                     data_shape, data_size_mb, checksum, version,
                     description, tags_json, created_at, updated_at) = row
                    
                    features.append(FeatureMetadata(
                        feature_id=feature_id,
                        feature_name=feature_name,
                        feature_type=ft,
                        params=json.loads(p) if p else {},
                        dependencies=json.loads(deps) if deps else [],
                        created_at=created_at,
                        updated_at=updated_at,
                        data_shape=tuple(json.loads(data_shape)) if data_shape else (0, 0),
                        data_size_mb=data_size_mb or 0,
                        checksum=checksum or "",
                        version=version or "1.0",
                        description=description or "",
                        tags=json.loads(tags_json) if tags_json else [],
                        storage_type="postgresql"
                    ))
                
                return features
                
        except Exception as e:
            logger.error(f"从PostgreSQL列出特征失败: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def cleanup_expired(self) -> int:
        """清理过期特征"""
        try:
            expired_count = 0
            features = self.list_features()
            
            for metadata in features:
                if self._is_expired(metadata):
                    if self.delete_feature(metadata.feature_id):
                        expired_count += 1
            
            logger.info(f"清理了 {expired_count} 个过期特征")
            return expired_count
            
        except Exception as e:
            logger.error(f"清理过期特征失败: {e}")
            return 0

    def get_store_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = self.stats.copy()
        
        total_requests = stats['cache_hits'] + stats['cache_misses']
        stats['hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        stats['postgresql_available'] = self._pg_available
        
        total_size = 0
        feature_count = 0
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*), COALESCE(SUM(data_size_mb), 0) FROM feature_cache")
                        row = cur.fetchone()
                        stats['postgresql_count'] = row[0]
                        stats['postgresql_size_mb'] = row[1]
                        total_size += row[1]
                        feature_count += row[0]
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        total_size += metadata.get('data_size_mb', 0)
                        feature_count += 1
                except Exception:
                    continue
        except Exception:
            pass
        
        stats['total_size_mb'] = total_size
        stats['feature_count'] = feature_count
        
        return stats

    def sync_to_postgresql(self) -> int:
        """将文件系统中的数据同步到PostgreSQL"""
        if not self._pg_available:
            logger.warning("PostgreSQL不可用，无法同步")
            return 0
        
        synced = 0
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                feature_id = data.get('feature_id', metadata_file.stem)
                
                feature_data = self._load_feature_data(feature_id)
                if feature_data is not None:
                    metadata = self._dict_to_metadata(data)
                    metadata.feature_id = feature_id
                    
                    if self._save_to_postgresql(feature_id, feature_data, metadata):
                        synced += 1
                        
            except Exception as e:
                logger.error(f"同步特征失败 {metadata_file}: {e}")
        
        logger.info(f"已同步 {synced} 个特征到PostgreSQL")
        return synced

    def _dict_to_metadata(self, data: Dict[str, Any]) -> FeatureMetadata:
        """将字典转换为FeatureMetadata"""
        return FeatureMetadata(
            feature_id=data.get('feature_id', ''),
            feature_name=data['feature_name'],
            feature_type=data['feature_type'],
            params=data.get('params', {}),
            dependencies=data.get('dependencies', []),
            created_at=self._parse_datetime(data.get('created_at')),
            updated_at=self._parse_datetime(data.get('updated_at')),
            data_shape=tuple(data.get('data_shape', (0, 0))),
            data_size_mb=data.get('data_size_mb', 0),
            checksum=data.get('checksum', ''),
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            storage_type=data.get('storage_type', 'filesystem')
        )

    def _parse_datetime(self, dt: Any) -> datetime:
        """解析日期时间"""
        if isinstance(dt, datetime):
            return dt
        if isinstance(dt, str):
            try:
                return datetime.fromisoformat(dt)
            except Exception:
                pass
        return datetime.now()

    def _generate_feature_id(self, feature_name: str, params: Dict[str, Any]) -> str:
        """生成特征ID"""
        param_str = json.dumps(params, sort_keys=True)
        content = f"{feature_name}:{param_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _feature_exists(self, feature_id: str) -> bool:
        """检查特征是否存在"""
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1 FROM feature_cache WHERE feature_id = %s", (feature_id,))
                        return cur.fetchone() is not None
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        metadata_file = self.metadata_path / f"{feature_id}.json"
        return metadata_file.exists()

    def _save_feature_data(self, feature_id: str, data: pd.DataFrame) -> bool:
        """保存特征数据到文件系统"""
        try:
            data_file = self.data_path / f"{feature_id}.pkl"
            
            if self.config.compression:
                data.to_pickle(data_file, compression='gzip')
            else:
                data.to_pickle(data_file)
            
            return True
        except Exception as e:
            logger.error(f"保存特征数据失败: {e}")
            return False

    def _load_feature_data(self, feature_id: str) -> Optional[pd.DataFrame]:
        """从文件系统加载特征数据"""
        try:
            data_file = self.data_path / f"{feature_id}.pkl"
            
            if not data_file.exists():
                return None
            
            if self.config.compression:
                return pd.read_pickle(data_file, compression='gzip')
            else:
                return pd.read_pickle(data_file)
            
        except Exception as e:
            logger.error(f"加载特征数据失败: {e}")
            return None

    def _delete_feature_data(self, feature_id: str):
        """删除特征数据文件"""
        try:
            data_file = self.data_path / f"{feature_id}.pkl"
            if data_file.exists():
                data_file.unlink()
        except Exception as e:
            logger.error(f"删除特征数据失败: {e}")

    def _save_metadata(self, feature_id: str, metadata: FeatureMetadata) -> bool:
        """保存元数据到文件系统"""
        try:
            metadata_file = self.metadata_path / f"{feature_id}.json"
            metadata.feature_id = feature_id
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            return True
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            return False

    def _load_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """从文件系统加载元数据"""
        try:
            metadata_file = self.metadata_path / f"{feature_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return self._dict_to_metadata(data)
            
            return None
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return None

    def _delete_metadata(self, feature_id: str):
        """删除元数据文件"""
        try:
            metadata_file = self.metadata_path / f"{feature_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
        except Exception as e:
            logger.error(f"删除元数据失败: {e}")

    def _update_feature(
        self,
        feature_id: str,
        data: pd.DataFrame,
        config: FeatureRegistrationConfig,
        description: str,
        tags: List[str]
    ) -> bool:
        """更新特征"""
        try:
            metadata = self._load_metadata(feature_id)
            if not metadata:
                metadata = FeatureMetadata(
                    feature_id=feature_id,
                    feature_name=config.name if hasattr(config, 'name') else feature_id,
                    feature_type=config.feature_type.value,
                    params=config.params,
                    dependencies=config.dependencies,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    data_shape=data.shape,
                    data_size_mb=data.memory_usage(deep=True).sum() / 1024 / 1024,
                    checksum=self._calculate_checksum(data),
                    description=description,
                    tags=tags or []
                )
            
            metadata.updated_at = datetime.now()
            metadata.data_shape = data.shape
            metadata.data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            metadata.checksum = self._calculate_checksum(data)
            metadata.description = description
            metadata.tags = tags or []
            
            if self._pg_available:
                if self._save_to_postgresql(feature_id, data, metadata):
                    metadata.storage_type = "postgresql"
                    return True
            
            success = self._save_feature_data(feature_id, data)
            if not success:
                return False
            
            return self._save_metadata(feature_id, metadata)
            
        except Exception as e:
            logger.error(f"更新特征失败: {e}")
            return False

    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """计算数据校验和"""
        try:
            content = f"{data.shape}:{data.columns.tolist()}:{data.sum().sum()}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception:
            return ""

    def _is_expired(self, metadata: FeatureMetadata) -> bool:
        """检查特征是否过期"""
        if self.config.ttl_hours <= 0:
            return False
        
        expiry_time = metadata.updated_at + timedelta(hours=self.config.ttl_hours)
        return datetime.now() > expiry_time

    def _list_features_filesystem(
        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """从文件系统列出特征"""
        features = []
        
        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metadata = self._dict_to_metadata(data)
                    
                    if feature_type and metadata.feature_type != feature_type:
                        continue
                    
                    if tags and not any(tag in metadata.tags for tag in tags):
                        continue
                    
                    features.append(metadata)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"从文件系统列出特征失败: {e}")
        
        return features

    def close(self):
        """关闭存储管理器"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ============ 存储配额管理相关方法 ============

    def check_storage_quota(self) -> Dict[str, Any]:
        """
        检查存储配额状态

        Returns:
            Dict: 配额检查结果
        """
        if not self._quota_manager:
            return {
                "enabled": False,
                "message": "存储配额管理未启用"
            }

        try:
            return self._quota_manager.check_quota()
        except Exception as e:
            logger.error(f"检查存储配额失败: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

    def cleanup_storage(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        清理存储空间

        Args:
            dry_run: 是否为试运行模式

        Returns:
            Dict: 清理结果
        """
        if not self._quota_manager:
            return {
                "enabled": False,
                "message": "存储配额管理未启用"
            }

        try:
            # 首先清理过期的特征
            expired_count = self.cleanup_expired()

            # 然后执行配额管理器的清理
            quota_result = self._quota_manager.cleanup_old_files(dry_run=dry_run)

            return {
                "enabled": True,
                "dry_run": dry_run,
                "expired_features_cleaned": expired_count,
                "quota_cleanup": quota_result
            }
        except Exception as e:
            logger.error(f"清理存储失败: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

    def get_storage_quota_report(self) -> Dict[str, Any]:
        """
        获取存储配额报告

        Returns:
            Dict: 配额报告
        """
        if not self._quota_manager:
            return {
                "enabled": False,
                "message": "存储配额管理未启用"
            }

        try:
            return self._quota_manager.get_quota_report()
        except Exception as e:
            logger.error(f"获取存储配额报告失败: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

    def get_storage_stats_extended(self) -> Dict[str, Any]:
        """
        获取扩展存储统计信息（包含配额信息）

        Returns:
            Dict: 扩展统计信息
        """
        # 获取基本统计
        stats = self.get_store_stats()

        # 添加配额信息
        quota_check = self.check_storage_quota()
        quota_report = self.get_storage_quota_report()

        return {
            **stats,
            "quota": {
                "enabled": quota_check.get("enabled", False),
                "status": quota_check.get("status", "unknown"),
                "usage_percent": quota_check.get("usage_percent", 0),
                "needs_cleanup": quota_check.get("needs_cleanup", False)
            },
            "quota_report": quota_report if quota_report.get("enabled") else None
        }

    # ============ 版本管理相关方法 ============

    def create_feature_version(
        self,
        feature_name: str,
        data: pd.DataFrame,
        config: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        创建特征版本

        Args:
            feature_name: 特征名称
            data: 特征数据
            config: 特征配置
            description: 版本描述
            tags: 标签列表

        Returns:
            str: 版本ID，失败返回None
        """
        if not self._version_manager:
            logger.warning("版本管理未启用")
            return None

        try:
            version_id = self._version_manager.create_version(
                feature_name=feature_name,
                data=data,
                config=config,
                description=description,
                tags=tags
            )
            logger.info(f"✅ 特征版本已创建: {feature_name}@{version_id}")
            return version_id
        except Exception as e:
            logger.error(f"❌ 创建特征版本失败: {e}")
            return None

    def get_feature_versions(
        self,
        feature_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取特征版本列表

        Args:
            feature_name: 特征名称
            limit: 返回数量限制

        Returns:
            List: 版本信息列表
        """
        if not self._version_manager:
            return []

        try:
            return self._version_manager.list_versions(feature_name, limit)
        except Exception as e:
            logger.error(f"❌ 获取特征版本列表失败: {e}")
            return []

    def rollback_to_version(
        self,
        feature_name: str,
        version_id: str
    ) -> bool:
        """
        回滚到指定版本

        Args:
            feature_name: 特征名称
            version_id: 版本ID

        Returns:
            bool: 是否成功
        """
        if not self._version_manager:
            logger.warning("版本管理未启用")
            return False

        try:
            success = self._version_manager.rollback_to_version(feature_name, version_id)
            if success:
                logger.info(f"✅ 已回滚到版本: {feature_name}@{version_id}")
            else:
                logger.error(f"❌ 回滚失败: {feature_name}@{version_id}")
            return success
        except Exception as e:
            logger.error(f"❌ 回滚到版本失败: {e}")
            return False

    def compare_feature_versions(
        self,
        feature_name: str,
        version_id1: str,
        version_id2: str
    ) -> Optional[Dict[str, Any]]:
        """
        比较两个特征版本

        Args:
            feature_name: 特征名称
            version_id1: 版本1 ID
            version_id2: 版本2 ID

        Returns:
            Dict: 比较结果
        """
        if not self._version_manager:
            logger.warning("版本管理未启用")
            return None

        try:
            return self._version_manager.compare_versions(feature_name, version_id1, version_id2)
        except Exception as e:
            logger.error(f"❌ 比较特征版本失败: {e}")
            return None

    def get_version_info(
        self,
        feature_name: str,
        version_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取版本详细信息

        Args:
            feature_name: 特征名称
            version_id: 版本ID

        Returns:
            Dict: 版本信息
        """
        if not self._version_manager:
            return None

        try:
            return self._version_manager.get_version_info(feature_name, version_id)
        except Exception as e:
            logger.error(f"❌ 获取版本信息失败: {e}")
            return None

    def get_version_manager_stats(self) -> Dict[str, Any]:
        """
        获取版本管理器统计信息

        Returns:
            Dict: 统计信息
        """
        if not self._version_manager:
            return {
                "enabled": False,
                "message": "版本管理未启用"
            }

        try:
            return {
                "enabled": True,
                "version_dir": self._version_manager.version_dir,
                "features_count": len(self._version_manager.version_index)
            }
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    # ============ 特征血缘追踪相关方法 ============

    def _register_feature_lineage(
        self,
        feature_name: str,
        config: FeatureRegistrationConfig,
        metadata: FeatureMetadata
    ) -> bool:
        """
        注册特征血缘关系

        Args:
            feature_name: 特征名称
            config: 特征配置
            metadata: 特征元数据

        Returns:
            bool: 是否成功
        """
        if not self._lineage_tracker:
            return False

        try:
            # 确定特征类型
            feature_type = self._determine_feature_type(config)

            # 注册血缘关系
            success = self._lineage_tracker.register_feature(
                feature_name=feature_name,
                feature_type=feature_type,
                source_features=config.dependencies,
                transformation_logic=getattr(config, 'transformation_logic', ''),
                parameters=config.params,
                tags=metadata.tags,
                metadata={
                    'feature_id': metadata.feature_id,
                    'data_shape': metadata.data_shape,
                    'data_size_mb': metadata.data_size_mb,
                    'version': metadata.version
                }
            )

            if success:
                logger.debug(f"✅ 特征血缘已注册: {feature_name}")
            return success

        except Exception as e:
            logger.warning(f"⚠️ 注册特征血缘失败: {e}")
            return False

    def _determine_feature_type(self, config: FeatureRegistrationConfig) -> str:
        """
        根据配置确定特征类型

        Args:
            config: 特征配置

        Returns:
            str: 特征类型
        """
        # 根据依赖关系判断特征类型
        if not config.dependencies:
            return 'raw'

        # 根据参数判断
        params = config.params
        if 'window' in params or 'period' in params:
            if any(op in str(params) for op in ['mean', 'std', 'min', 'max', 'sum']):
                return 'aggregated'

        if 'transform' in str(params).lower() or 'operation' in params:
            return 'transformed'

        return 'derived'

    def get_feature_lineage(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        获取特征血缘信息

        Args:
            feature_name: 特征名称

        Returns:
            Dict: 血缘信息
        """
        if not self._lineage_tracker:
            return None

        try:
            node = self._lineage_tracker.get_lineage(feature_name)
            if node:
                return {
                    'feature_name': node.feature_name,
                    'feature_type': node.feature_type,
                    'source_features': node.source_features,
                    'derived_features': node.derived_features,
                    'transformation_logic': node.transformation_logic,
                    'parameters': node.parameters,
                    'version': node.version,
                    'tags': node.tags,
                    'created_at': node.created_at.isoformat() if node.created_at else None,
                    'metadata': node.metadata
                }
            return None
        except Exception as e:
            logger.error(f"❌ 获取特征血缘失败: {e}")
            return None

    def get_upstream_features(self, feature_name: str, depth: int = -1) -> List[str]:
        """
        获取上游特征（依赖的特征）

        Args:
            feature_name: 特征名称
            depth: 查询深度，-1表示无限

        Returns:
            List[str]: 上游特征列表
        """
        if not self._lineage_tracker:
            return []

        try:
            return self._lineage_tracker.get_upstream_features(feature_name, depth)
        except Exception as e:
            logger.error(f"❌ 获取上游特征失败: {e}")
            return []

    def get_downstream_features(self, feature_name: str, depth: int = -1) -> List[str]:
        """
        获取下游特征（被依赖的特征）

        Args:
            feature_name: 特征名称
            depth: 查询深度，-1表示无限

        Returns:
            List[str]: 下游特征列表
        """
        if not self._lineage_tracker:
            return []

        try:
            return self._lineage_tracker.get_downstream_features(feature_name, depth)
        except Exception as e:
            logger.error(f"❌ 获取下游特征失败: {e}")
            return []

    def get_lineage_graph(self, feature_name: str, depth: int = 2) -> Optional[Dict[str, Any]]:
        """
        获取特征血缘图谱

        Args:
            feature_name: 特征名称
            depth: 查询深度

        Returns:
            Dict: 血缘图谱数据
        """
        if not self._lineage_tracker:
            return None

        try:
            return self._lineage_tracker.get_lineage_graph(feature_name, depth)
        except Exception as e:
            logger.error(f"❌ 获取血缘图谱失败: {e}")
            return None

    def analyze_feature_impact(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        分析特征变更影响范围

        Args:
            feature_name: 特征名称

        Returns:
            Dict: 影响分析结果
        """
        if not self._lineage_tracker:
            return None

        try:
            return self._lineage_tracker.analyze_impact(feature_name)
        except Exception as e:
            logger.error(f"❌ 分析特征影响失败: {e}")
            return None

    def get_lineage_summary(self) -> Optional[Dict[str, Any]]:
        """
        获取血缘统计摘要

        Returns:
            Dict: 统计信息
        """
        if not self._lineage_tracker:
            return None

        try:
            return self._lineage_tracker.get_lineage_summary()
        except Exception as e:
            logger.error(f"❌ 获取血缘摘要失败: {e}")
            return None

    def export_lineage(self, format: str = "json") -> Optional[str]:
        """
        导出血缘数据

        Args:
            format: 导出格式 ('json', 'dot')

        Returns:
            str: 导出的数据
        """
        if not self._lineage_tracker:
            return None

        try:
            return self._lineage_tracker.export_lineage(format)
        except Exception as e:
            logger.error(f"❌ 导出血缘数据失败: {e}")
            return None

    def get_lineage_tracker_stats(self) -> Dict[str, Any]:
        """
        获取血缘追踪器统计信息

        Returns:
            Dict: 统计信息
        """
        if not self._lineage_tracker:
            return {
                "enabled": False,
                "message": "特征血缘追踪未启用"
            }

        try:
            summary = self._lineage_tracker.get_lineage_summary()
            return {
                "enabled": True,
                **summary
            }
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }


_feature_store_instance: Optional[FeatureStore] = None


def get_feature_store(**kwargs) -> FeatureStore:
    """获取全局FeatureStore实例"""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore(**kwargs)
    return _feature_store_instance
