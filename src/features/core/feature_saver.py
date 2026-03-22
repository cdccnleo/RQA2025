#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征结果持久化工具

负责保存/加载特征数据及元数据，支持PostgreSQL优先存储策略：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统 (parquet/csv/pickle)

使用方式:
    saver = FeatureSaver()
    saver.save_features(features, "feature_set_001", metadata={"source": "stock_data"})
    loaded = saver.load_features("feature_set_001")
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import threading

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """特征元数据"""
    feature_id: str
    feature_name: str
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    format: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    checksum: str = ""
    size_bytes: int = 0
    storage_type: str = "filesystem"  # postgresql / filesystem


class FeatureSaver:
    """
    特征结果持久化工具
    
    实现PostgreSQL优先存储，数据库连接失败时自动降级到文件系统。
    支持多种存储格式：parquet、csv、pickle。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(
        self,
        base_path: Union[str, Path] = "./feature_outputs",
        metadata_name: str = "metadata.json",
        enable_postgresql: bool = True
    ):
        """
        初始化特征保存器
        
        Args:
            base_path: 文件系统存储的基础路径
            metadata_name: 元数据文件名
            enable_postgresql: 是否启用PostgreSQL存储
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / metadata_name
        self.last_save_info: Optional[Dict[str, Any]] = None
        self.enable_postgresql = enable_postgresql
        
        self._lock = threading.Lock()
        
        self._pg_config = None
        self._pg_available = False
        if enable_postgresql:
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self._ensure_tables_exist()
                logger.info("✅ FeatureSaver: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ FeatureSaver: PostgreSQL 不可用，使用文件系统存储")
        
        self._metadata_cache: Dict[str, FeatureMetadata] = {}
        self._load_metadata_cache()
    
    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """
        获取PostgreSQL配置（统一使用 database_config 模块）
        
        所有数据库配置统一从 database_config 模块获取，确保配置一致性。
        不再在各地硬编码配置或密码。
        
        Returns:
            数据库配置字典，获取失败返回None
        """
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
        except Exception as e:
            logger.error(f"获取PostgreSQL配置失败: {e}")
            return None
    
    def _get_db_connection(self):
        """
        获取数据库连接（带重试机制）
        
        Returns:
            数据库连接对象或None
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
                    CREATE TABLE IF NOT EXISTS feature_store (
                        feature_id VARCHAR(64) PRIMARY KEY,
                        feature_name VARCHAR(255) NOT NULL,
                        shape JSONB NOT NULL,
                        columns JSONB NOT NULL,
                        dtypes JSONB NOT NULL,
                        format VARCHAR(20) NOT NULL DEFAULT 'parquet',
                        data BYTEA,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}',
                        checksum VARCHAR(64),
                        size_bytes BIGINT DEFAULT 0
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_store_name 
                    ON feature_store(feature_name)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_store_created 
                    ON feature_store(created_at DESC)
                """)
            
            conn.commit()
            logger.debug("特征存储表已确保存在")
            
        except Exception as e:
            logger.error(f"创建特征存储表失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def save_features(
        self,
        features: pd.DataFrame,
        feature_name: str,
        format: str = "parquet",
        metadata: Optional[Dict[str, Any]] = None,
        feature_id: Optional[str] = None
    ) -> bool:
        """
        保存特征数据（PostgreSQL优先，降级到文件系统）
        
        Args:
            features: 特征数据DataFrame
            feature_name: 特征集名称
            format: 存储格式 (parquet/csv/pickle)
            metadata: 附加元数据
            feature_id: 特征ID（可选，默认自动生成）
            
        Returns:
            保存是否成功
        """
        if features.empty:
            logger.warning("特征数据为空，跳过保存")
            return False
        
        with self._lock:
            try:
                if not feature_id:
                    feature_id = self._generate_feature_id(feature_name, metadata)
                
                feature_metadata = self._create_metadata(
                    feature_id, feature_name, features, format, metadata
                )
                
                if self._pg_available:
                    success = self._save_to_postgresql(features, feature_metadata)
                    if success:
                        feature_metadata.storage_type = "postgresql"
                        self._metadata_cache[feature_id] = feature_metadata
                        self.last_save_info = asdict(feature_metadata)
                        logger.info(f"✅ 特征已保存到PostgreSQL: {feature_name} (ID: {feature_id})")
                        return True
                    else:
                        logger.warning("PostgreSQL保存失败，降级到文件系统")
                
                success = self._save_to_filesystem(features, feature_metadata)
                if success:
                    feature_metadata.storage_type = "filesystem"
                    self._metadata_cache[feature_id] = feature_metadata
                    self.last_save_info = asdict(feature_metadata)
                    logger.info(f"✅ 特征已保存到文件系统: {feature_name} (ID: {feature_id})")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"保存特征失败: {e}", exc_info=True)
                return False
    
    def _save_to_postgresql(
        self, 
        features: pd.DataFrame, 
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
            if metadata.format == "parquet":
                features.to_parquet(buffer, index=False, compression='snappy')
            elif metadata.format == "csv":
                features.to_csv(buffer, index=False, encoding='utf-8')
            elif metadata.format == "pickle":
                features.to_pickle(buffer, compression='gzip')
            else:
                raise ValueError(f"不支持的格式: {metadata.format}")
            
            data_bytes = buffer.getvalue()
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_store (
                        feature_id, feature_name, shape, columns, dtypes, format,
                        data, created_at, updated_at, metadata, checksum, size_bytes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (feature_id) DO UPDATE SET
                        feature_name = EXCLUDED.feature_name,
                        shape = EXCLUDED.shape,
                        columns = EXCLUDED.columns,
                        dtypes = EXCLUDED.dtypes,
                        format = EXCLUDED.format,
                        data = EXCLUDED.data,
                        updated_at = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata,
                        checksum = EXCLUDED.checksum,
                        size_bytes = EXCLUDED.size_bytes
                """, (
                    metadata.feature_id,
                    metadata.feature_name,
                    json.dumps(list(metadata.shape)),
                    json.dumps(metadata.columns),
                    json.dumps(metadata.dtypes),
                    metadata.format,
                    data_bytes,
                    datetime.now(),
                    datetime.now(),
                    json.dumps(metadata.metadata),
                    metadata.checksum,
                    metadata.size_bytes
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
    
    def _save_to_filesystem(
        self, 
        features: pd.DataFrame, 
        metadata: FeatureMetadata
    ) -> bool:
        """保存特征到文件系统"""
        try:
            output_path = self.base_path / f"{metadata.feature_id}.{metadata.format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if metadata.format == "parquet":
                features.to_parquet(output_path, index=False)
            elif metadata.format == "csv":
                features.to_csv(output_path, index=False)
            elif metadata.format == "pickle":
                features.to_pickle(output_path)
            else:
                raise ValueError(f"不支持的格式: {metadata.format}")
            
            metadata_path = self.base_path / f"{metadata.feature_id}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, ensure_ascii=False, indent=2, default=str)
            
            self._update_metadata_file(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"保存到文件系统失败: {e}")
            return False
    
    def load_features(
        self, 
        feature_id: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        加载特征数据（优先从PostgreSQL，降级到文件系统）
        
        Args:
            feature_id: 特征ID
            format: 存储格式
            
        Returns:
            特征数据DataFrame，加载失败返回None
        """
        with self._lock:
            if self._pg_available:
                features = self._load_from_postgresql(feature_id)
                if features is not None:
                    logger.debug(f"从PostgreSQL加载特征: {feature_id}")
                    return features
            
            features = self._load_from_filesystem(feature_id, format)
            if features is not None:
                logger.debug(f"从文件系统加载特征: {feature_id}")
                return features
            
            logger.warning(f"特征加载失败，未找到: {feature_id}")
            return None
    
    def _load_from_postgresql(self, feature_id: str) -> Optional[pd.DataFrame]:
        """从PostgreSQL加载特征"""
        conn = None
        try:
            import io
            
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT data, format, columns FROM feature_store 
                    WHERE feature_id = %s
                """, (feature_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                data_bytes, format_type, columns = row
                buffer = io.BytesIO(data_bytes)
                
                if format_type == "parquet":
                    return pd.read_parquet(buffer)
                elif format_type == "csv":
                    return pd.read_csv(buffer)
                elif format_type == "pickle":
                    return pd.read_pickle(buffer)
                
                return None
                
        except Exception as e:
            logger.error(f"从PostgreSQL加载失败: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def _load_from_filesystem(
        self, 
        feature_id: str, 
        format: str
    ) -> Optional[pd.DataFrame]:
        """从文件系统加载特征"""
        try:
            output_path = self.base_path / f"{feature_id}.{format}"
            
            if not output_path.exists():
                for fmt in ['parquet', 'csv', 'pickle']:
                    alt_path = self.base_path / f"{feature_id}.{fmt}"
                    if alt_path.exists():
                        output_path = alt_path
                        format = fmt
                        break
            
            if not output_path.exists():
                return None
            
            if format == "parquet":
                return pd.read_parquet(output_path)
            elif format == "csv":
                return pd.read_csv(output_path)
            elif format == "pickle":
                return pd.read_pickle(output_path)
            
            return None
            
        except Exception as e:
            logger.error(f"从文件系统加载失败: {e}")
            return None
    
    def delete_features(self, feature_id: str) -> bool:
        """
        删除特征数据
        
        Args:
            feature_id: 特征ID
            
        Returns:
            删除是否成功
        """
        with self._lock:
            success = False
            
            if self._pg_available:
                success = self._delete_from_postgresql(feature_id) or success
            
            success = self._delete_from_filesystem(feature_id) or success
            
            if feature_id in self._metadata_cache:
                del self._metadata_cache[feature_id]
            
            return success
    
    def _delete_from_postgresql(self, feature_id: str) -> bool:
        """从PostgreSQL删除特征"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("DELETE FROM feature_store WHERE feature_id = %s", (feature_id,))
            
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
            for fmt in ['parquet', 'csv', 'pickle']:
                path = self.base_path / f"{feature_id}.{fmt}"
                if path.exists():
                    path.unlink()
                    deleted = True
            
            metadata_path = self.base_path / f"{feature_id}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"从文件系统删除失败: {e}")
            return False
    
    def list_features(
        self, 
        feature_name: Optional[str] = None,
        limit: int = 100
    ) -> List[FeatureMetadata]:
        """
        列出所有特征
        
        Args:
            feature_name: 过滤特征名称
            limit: 返回数量限制
            
        Returns:
            特征元数据列表
        """
        features = []
        
        if self._pg_available:
            features = self._list_from_postgresql(feature_name, limit)
        
        fs_features = self._list_from_filesystem(feature_name, limit)
        
        pg_ids = {f.feature_id for f in features}
        for f in fs_features:
            if f.feature_id not in pg_ids:
                features.append(f)
        
        return features[:limit]
    
    def _list_from_postgresql(
        self, 
        feature_name: Optional[str], 
        limit: int
    ) -> List[FeatureMetadata]:
        """从PostgreSQL列出特征"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                if feature_name:
                    cur.execute("""
                        SELECT feature_id, feature_name, shape, columns, dtypes, format,
                               created_at, updated_at, metadata, checksum, size_bytes
                        FROM feature_store 
                        WHERE feature_name = %s
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (feature_name, limit))
                else:
                    cur.execute("""
                        SELECT feature_id, feature_name, shape, columns, dtypes, format,
                               created_at, updated_at, metadata, checksum, size_bytes
                        FROM feature_store 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (limit,))
                
                features = []
                for row in cur.fetchall():
                    features.append(FeatureMetadata(
                        feature_id=row[0],
                        feature_name=row[1],
                        shape=tuple(json.loads(row[2])),
                        columns=json.loads(row[3]),
                        dtypes=json.loads(row[4]),
                        format=row[5],
                        created_at=str(row[6]),
                        updated_at=str(row[7]),
                        metadata=json.loads(row[8]) if row[8] else {},
                        checksum=row[9] or "",
                        size_bytes=row[10] or 0,
                        storage_type="postgresql"
                    ))
                
                return features
                
        except Exception as e:
            logger.error(f"从PostgreSQL列出特征失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _list_from_filesystem(
        self, 
        feature_name: Optional[str], 
        limit: int
    ) -> List[FeatureMetadata]:
        """从文件系统列出特征"""
        features = []
        try:
            for metadata_file in self.base_path.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if feature_name and data.get('feature_name') != feature_name:
                        continue
                    
                    features.append(FeatureMetadata(
                        feature_id=data['feature_id'],
                        feature_name=data['feature_name'],
                        shape=tuple(data['shape']),
                        columns=data['columns'],
                        dtypes=data['dtypes'],
                        format=data['format'],
                        created_at=data['created_at'],
                        updated_at=data['updated_at'],
                        metadata=data.get('metadata', {}),
                        checksum=data.get('checksum', ''),
                        size_bytes=data.get('size_bytes', 0),
                        storage_type="filesystem"
                    ))
                    
                    if len(features) >= limit:
                        break
                        
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"从文件系统列出特征失败: {e}")
        
        return features
    
    def get_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """
        获取特征元数据
        
        Args:
            feature_id: 特征ID
            
        Returns:
            特征元数据，未找到返回None
        """
        if feature_id in self._metadata_cache:
            return self._metadata_cache[feature_id]
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT feature_id, feature_name, shape, columns, dtypes, format,
                                   created_at, updated_at, metadata, checksum, size_bytes
                            FROM feature_store WHERE feature_id = %s
                        """, (feature_id,))
                        
                        row = cur.fetchone()
                        if row:
                            return FeatureMetadata(
                                feature_id=row[0],
                                feature_name=row[1],
                                shape=tuple(json.loads(row[2])),
                                columns=json.loads(row[3]),
                                dtypes=json.loads(row[4]),
                                format=row[5],
                                created_at=str(row[6]),
                                updated_at=str(row[7]),
                                metadata=json.loads(row[8]) if row[8] else {},
                                checksum=row[9] or "",
                                size_bytes=row[10] or 0,
                                storage_type="postgresql"
                            )
            except Exception as e:
                logger.error(f"从PostgreSQL获取元数据失败: {e}")
            finally:
                if conn:
                    conn.close()
        
        metadata_path = self.base_path / f"{feature_id}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return FeatureMetadata(**data)
            except Exception as e:
                logger.error(f"从文件系统获取元数据失败: {e}")
        
        return None
    
    def get_last_metadata(self) -> Optional[Dict[str, Any]]:
        """返回最近一次保存的元数据内容"""
        return self.last_save_info
    
    def sync_to_postgresql(self) -> int:
        """
        将文件系统中的数据同步到PostgreSQL
        
        Returns:
            同步的记录数量
        """
        if not self._pg_available:
            logger.warning("PostgreSQL不可用，无法同步")
            return 0
        
        synced = 0
        for metadata_file in self.base_path.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                feature_id = data['feature_id']
                format_type = data['format']
                
                features = self._load_from_filesystem(feature_id, format_type)
                if features is not None:
                    metadata = FeatureMetadata(**data)
                    if self._save_to_postgresql(features, metadata):
                        synced += 1
                        
            except Exception as e:
                logger.error(f"同步特征失败 {metadata_file}: {e}")
        
        logger.info(f"已同步 {synced} 个特征到PostgreSQL")
        return synced
    
    def _generate_feature_id(
        self, 
        feature_name: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """生成特征ID"""
        content = f"{feature_name}:{json.dumps(metadata or {}, sort_keys=True)}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _create_metadata(
        self,
        feature_id: str,
        feature_name: str,
        features: pd.DataFrame,
        format: str,
        metadata: Optional[Dict[str, Any]]
    ) -> FeatureMetadata:
        """创建特征元数据"""
        now = datetime.now().isoformat()
        
        dtypes = {col: str(dtype) for col, dtype in features.dtypes.items()}
        
        checksum = self._calculate_checksum(features)
        
        size_bytes = features.memory_usage(deep=True).sum()
        
        return FeatureMetadata(
            feature_id=feature_id,
            feature_name=feature_name,
            shape=features.shape,
            columns=features.columns.tolist(),
            dtypes=dtypes,
            format=format,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            checksum=checksum,
            size_bytes=int(size_bytes)
        )
    
    def _calculate_checksum(self, features: pd.DataFrame) -> str:
        """计算数据校验和"""
        try:
            content = f"{features.shape}:{features.columns.tolist()}:{features.sum().sum()}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception:
            return ""
    
    def _load_metadata_cache(self):
        """加载元数据缓存"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for fid, meta in data.items():
                            try:
                                self._metadata_cache[fid] = FeatureMetadata(**meta)
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"加载元数据缓存失败: {e}")
    
    def _update_metadata_file(self, metadata: FeatureMetadata):
        """更新元数据文件"""
        try:
            cache_data = {fid: asdict(meta) for fid, meta in self._metadata_cache.items()}
            cache_data[metadata.feature_id] = asdict(metadata)
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"更新元数据文件失败: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            "postgresql_available": self._pg_available,
            "total_features": 0,
            "postgresql_count": 0,
            "filesystem_count": 0,
            "total_size_bytes": 0
        }
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM feature_store")
                        row = cur.fetchone()
                        stats["postgresql_count"] = row[0]
                        stats["total_size_bytes"] += row[1]
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        fs_count = 0
        fs_size = 0
        for f in self.base_path.glob("*.parquet"):
            fs_count += 1
            fs_size += f.stat().st_size
        for f in self.base_path.glob("*.csv"):
            fs_count += 1
            fs_size += f.stat().st_size
        for f in self.base_path.glob("*.pickle"):
            fs_count += 1
            fs_size += f.stat().st_size
        
        stats["filesystem_count"] = fs_count
        stats["total_size_bytes"] += fs_size
        stats["total_features"] = stats["postgresql_count"] + fs_count
        
        return stats


_feature_saver_instance: Optional[FeatureSaver] = None


def get_feature_saver(**kwargs) -> FeatureSaver:
    """
    获取全局FeatureSaver实例
    
    Args:
        **kwargs: FeatureSaver初始化参数
        
    Returns:
        FeatureSaver实例
    """
    global _feature_saver_instance
    if _feature_saver_instance is None:
        _feature_saver_instance = FeatureSaver(**kwargs)
    return _feature_saver_instance
