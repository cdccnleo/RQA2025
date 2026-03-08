"""
特征存储模块

提供特征的版本管理、存储和查询功能，支持SQLite和文件系统存储后端。
支持特征版本控制、依赖追踪、元数据管理和缓存机制。
"""

import logging
import sqlite3
import json
import hashlib
import pickle
import gzip
from datetime import datetime
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from threading import Lock
from contextlib import contextmanager

import pandas as pd
import numpy as np

from ..exceptions import PipelineException, PipelineErrorCode

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """存储后端类型"""
    SQLITE = "sqlite"
    FILESYSTEM = "filesystem"


class FeatureStatus(Enum):
    """特征状态"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class FeatureVersion:
    """
    特征版本信息
    
    Attributes:
        version_id: 版本唯一标识
        feature_name: 特征名称
        version: 语义化版本号
        created_at: 创建时间
        updated_at: 更新时间
        author: 创建者
        description: 版本描述
        tags: 标签列表
        dependencies: 依赖特征列表
        status: 特征状态
    """
    version_id: str
    feature_name: str
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: FeatureStatus = FeatureStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'feature_name': self.feature_name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author,
            'description': self.description,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVersion':
        """从字典创建"""
        return cls(
            version_id=data['version_id'],
            feature_name=data['feature_name'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            author=data.get('author'),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
            status=FeatureStatus(data.get('status', 'active'))
        )


@dataclass
class FeatureMetadata:
    """
    特征元数据
    
    Attributes:
        feature_name: 特征名称
        feature_type: 特征类型
        data_shape: 数据形状 (rows, cols)
        data_size_bytes: 数据大小（字节）
        checksum: 数据校验和
        column_names: 列名列表
        column_types: 列类型字典
        statistics: 统计信息字典
        params: 特征参数
    """
    feature_name: str
    feature_type: str
    data_shape: Tuple[int, int]
    data_size_bytes: int
    checksum: str
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'data_shape': list(self.data_shape),
            'data_size_bytes': self.data_size_bytes,
            'checksum': self.checksum,
            'column_names': self.column_names,
            'column_types': self.column_types,
            'statistics': self.statistics,
            'params': self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """从字典创建"""
        return cls(
            feature_name=data['feature_name'],
            feature_type=data['feature_type'],
            data_shape=tuple(data.get('data_shape', [0, 0])),
            data_size_bytes=data.get('data_size_bytes', 0),
            checksum=data.get('checksum', ''),
            column_names=data.get('column_names', []),
            column_types=data.get('column_types', {}),
            statistics=data.get('statistics', {}),
            params=data.get('params', {})
        )


@dataclass
class FeatureStoreConfig:
    """
    特征存储配置
    
    Attributes:
        storage_path: 存储路径
        backend: 存储后端类型
        compression: 是否启用压缩
        max_cache_size: 最大缓存条目数
        ttl_hours: 特征生存时间（小时）
        enable_versioning: 是否启用版本控制
    """
    storage_path: str = "./feature_store"
    backend: StorageBackend = StorageBackend.FILESYSTEM
    compression: bool = True
    max_cache_size: int = 100
    ttl_hours: int = 168  # 7天
    enable_versioning: bool = True


class FeatureStore:
    """
    特征存储管理器
    
    提供特征的版本管理、存储、查询功能。
    支持SQLite和文件系统两种存储后端，具备缓存机制和依赖追踪功能。
    
    Attributes:
        config: 存储配置
        _cache: 内存缓存
        _lock: 线程锁
    """
    
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        """
        初始化特征存储
        
        Args:
            config: 存储配置，为None时使用默认配置
        """
        self.config = config or FeatureStoreConfig()
        self._cache: Dict[str, Tuple[pd.DataFrame, FeatureMetadata]] = {}
        self._cache_access_time: Dict[str, datetime] = {}
        self._lock = Lock()
        
        # 初始化存储路径
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # 根据后端类型初始化
        if self.config.backend == StorageBackend.SQLITE:
            self._init_sqlite_backend()
        else:
            self._init_filesystem_backend()
        
        logger.info(f"特征存储初始化完成，后端: {self.config.backend.value}")
    
    def _init_sqlite_backend(self) -> None:
        """初始化SQLite后端"""
        self._db_path = self._storage_path / "feature_store.db"
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 创建版本表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_versions (
                    version_id TEXT PRIMARY KEY,
                    feature_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    author TEXT,
                    description TEXT,
                    tags TEXT,
                    dependencies TEXT,
                    status TEXT DEFAULT 'active',
                    UNIQUE(feature_name, version)
                )
            ''')
            
            # 创建元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    feature_type TEXT,
                    data_shape TEXT,
                    data_size_bytes INTEGER,
                    checksum TEXT,
                    column_names TEXT,
                    column_types TEXT,
                    statistics TEXT,
                    params TEXT,
                    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id)
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_feature_name 
                ON feature_versions(feature_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status 
                ON feature_versions(status)
            ''')
            
            conn.commit()
    
    def _init_filesystem_backend(self) -> None:
        """初始化文件系统后端"""
        self._data_path = self._storage_path / "data"
        self._metadata_path = self._storage_path / "metadata"
        self._version_path = self._storage_path / "versions"
        
        for path in [self._data_path, self._metadata_path, self._version_path]:
            path.mkdir(exist_ok=True)
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_version_id(self, feature_name: str, version: str) -> str:
        """
        生成版本唯一标识
        
        Args:
            feature_name: 特征名称
            version: 版本号
            
        Returns:
            版本ID
        """
        content = f"{feature_name}:{version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """
        计算数据校验和
        
        Args:
            data: 特征数据
            
        Returns:
            校验和字符串
        """
        try:
            # 使用数据的values和columns生成哈希
            content = str(data.values.tobytes()) + str(list(data.columns))
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"计算校验和失败: {e}")
            return ""
    
    def _extract_metadata(self, data: pd.DataFrame, feature_name: str, 
                         feature_type: str, params: Dict[str, Any]) -> FeatureMetadata:
        """
        从数据中提取元数据
        
        Args:
            data: 特征数据
            feature_name: 特征名称
            feature_type: 特征类型
            params: 特征参数
            
        Returns:
            特征元数据
        """
        # 计算统计信息
        statistics = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            statistics['mean'] = data[numeric_cols].mean().to_dict()
            statistics['std'] = data[numeric_cols].std().to_dict()
            statistics['min'] = data[numeric_cols].min().to_dict()
            statistics['max'] = data[numeric_cols].max().to_dict()
        
        return FeatureMetadata(
            feature_name=feature_name,
            feature_type=feature_type,
            data_shape=data.shape,
            data_size_bytes=data.memory_usage(deep=True).sum(),
            checksum=self._calculate_checksum(data),
            column_names=list(data.columns),
            column_types={col: str(dtype) for col, dtype in data.dtypes.items()},
            statistics=statistics,
            params=params
        )
    
    def store_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        version: str = "1.0.0",
        feature_type: str = "generic",
        author: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储特征
        
        Args:
            feature_name: 特征名称
            data: 特征数据
            version: 版本号
            feature_type: 特征类型
            author: 创建者
            description: 描述
            tags: 标签列表
            dependencies: 依赖特征列表
            params: 特征参数
            
        Returns:
            版本ID
            
        Raises:
            PipelineException: 存储失败时抛出
        """
        try:
            with self._lock:
                version_id = self._generate_version_id(feature_name, version)
                
                # 检查版本是否已存在
                if self._version_exists(version_id):
                    logger.warning(f"特征版本已存在: {feature_name}@{version}")
                    # 更新版本
                    return self._update_feature(version_id, data, description)
                
                # 提取元数据
                metadata = self._extract_metadata(
                    data, feature_name, feature_type, params or {}
                )
                
                # 创建版本信息
                version_info = FeatureVersion(
                    version_id=version_id,
                    feature_name=feature_name,
                    version=version,
                    author=author,
                    description=description,
                    tags=tags or [],
                    dependencies=dependencies or []
                )
                
                # 存储数据
                if self.config.backend == StorageBackend.SQLITE:
                    self._store_to_sqlite(version_id, data, version_info, metadata)
                else:
                    self._store_to_filesystem(version_id, data, version_info, metadata)
                
                # 更新缓存
                self._update_cache(version_id, data, metadata)
                
                logger.info(f"特征存储成功: {feature_name}@{version}")
                return version_id
                
        except Exception as e:
            logger.error(f"存储特征失败: {e}")
            raise PipelineException(
                message=f"存储特征失败: {e}",
                error_code=PipelineErrorCode.DATA_QUALITY_ERROR,
                context={'feature_name': feature_name, 'version': version}
            )
    
    def _version_exists(self, version_id: str) -> bool:
        """检查版本是否存在"""
        if self.config.backend == StorageBackend.SQLITE:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM feature_versions WHERE version_id = ?",
                    (version_id,)
                )
                return cursor.fetchone() is not None
        else:
            version_file = self._version_path / f"{version_id}.json"
            return version_file.exists()
    
    def _store_to_sqlite(
        self,
        version_id: str,
        data: pd.DataFrame,
        version_info: FeatureVersion,
        metadata: FeatureMetadata
    ) -> None:
        """存储到SQLite"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 插入版本信息
            cursor.execute('''
                INSERT INTO feature_versions 
                (version_id, feature_name, version, author, description, tags, 
                 dependencies, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version_info.version_id,
                version_info.feature_name,
                version_info.version,
                version_info.author,
                version_info.description,
                json.dumps(version_info.tags),
                json.dumps(version_info.dependencies),
                version_info.status.value
            ))
            
            # 插入元数据
            cursor.execute('''
                INSERT INTO feature_metadata
                (version_id, feature_type, data_shape, data_size_bytes, checksum,
                 column_names, column_types, statistics, params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version_id,
                metadata.feature_type,
                json.dumps(metadata.data_shape),
                metadata.data_size_bytes,
                metadata.checksum,
                json.dumps(metadata.column_names),
                json.dumps(metadata.column_types),
                json.dumps(metadata.statistics),
                json.dumps(metadata.params)
            ))
            
            # 存储数据（序列化到BLOB）
            data_bytes = pickle.dumps(data)
            if self.config.compression:
                data_bytes = gzip.compress(data_bytes)
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_data (
                    version_id TEXT PRIMARY KEY,
                    data BLOB,
                    compressed BOOLEAN
                )
            ''')
            cursor.execute('''
                INSERT INTO feature_data (version_id, data, compressed)
                VALUES (?, ?, ?)
            ''', (version_id, data_bytes, self.config.compression))
            
            conn.commit()
    
    def _store_to_filesystem(
        self,
        version_id: str,
        data: pd.DataFrame,
        version_info: FeatureVersion,
        metadata: FeatureMetadata
    ) -> None:
        """存储到文件系统"""
        # 存储数据
        data_file = self._data_path / f"{version_id}.pkl"
        if self.config.compression:
            data.to_pickle(data_file, compression='gzip')
        else:
            data.to_pickle(data_file)
        
        # 存储版本信息
        version_file = self._version_path / f"{version_id}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info.to_dict(), f, indent=2)
        
        # 存储元数据
        metadata_file = self._metadata_path / f"{version_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _update_feature(
        self,
        version_id: str,
        data: pd.DataFrame,
        description: str
    ) -> str:
        """更新特征"""
        # 提取新元数据
        old_metadata = self.get_metadata(version_id)
        if old_metadata:
            metadata = self._extract_metadata(
                data,
                old_metadata.feature_name,
                old_metadata.feature_type,
                old_metadata.params
            )
        else:
            metadata = self._extract_metadata(data, "unknown", "generic", {})
        
        # 更新数据
        if self.config.backend == StorageBackend.SQLITE:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 更新数据
                data_bytes = pickle.dumps(data)
                if self.config.compression:
                    data_bytes = gzip.compress(data_bytes)
                cursor.execute('''
                    UPDATE feature_data SET data = ?, compressed = ?
                    WHERE version_id = ?
                ''', (data_bytes, self.config.compression, version_id))
                
                # 更新元数据
                cursor.execute('''
                    UPDATE feature_metadata SET
                    data_shape = ?, data_size_bytes = ?, checksum = ?,
                    column_names = ?, column_types = ?, statistics = ?
                    WHERE version_id = ?
                ''', (
                    json.dumps(metadata.data_shape),
                    metadata.data_size_bytes,
                    metadata.checksum,
                    json.dumps(metadata.column_names),
                    json.dumps(metadata.column_types),
                    json.dumps(metadata.statistics),
                    version_id
                ))
                
                # 更新版本信息
                cursor.execute('''
                    UPDATE feature_versions SET
                    updated_at = CURRENT_TIMESTAMP, description = ?
                    WHERE version_id = ?
                ''', (description, version_id))
                
                conn.commit()
        else:
            # 文件系统更新
            data_file = self._data_path / f"{version_id}.pkl"
            if self.config.compression:
                data.to_pickle(data_file, compression='gzip')
            else:
                data.to_pickle(data_file)
            
            # 更新元数据
            metadata_file = self._metadata_path / f"{version_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # 更新版本信息
            version_file = self._version_path / f"{version_id}.json"
            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    version_info = FeatureVersion.from_dict(json.load(f))
                version_info.updated_at = datetime.now()
                version_info.description = description
                with open(version_file, 'w', encoding='utf-8') as f:
                    json.dump(version_info.to_dict(), f, indent=2)
        
        # 更新缓存
        self._update_cache(version_id, data, metadata)
        
        return version_id
    
    def get_feature(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """
        获取特征
        
        Args:
            feature_name: 特征名称
            version: 版本号，为None时获取最新版本
            
        Returns:
            (特征数据, 元数据) 或 None
        """
        try:
            # 确定版本ID
            if version:
                version_id = self._generate_version_id(feature_name, version)
            else:
                version_id = self._get_latest_version_id(feature_name)
                if not version_id:
                    return None
            
            # 检查缓存
            if version_id in self._cache:
                self._cache_access_time[version_id] = datetime.now()
                logger.debug(f"缓存命中: {feature_name}")
                return self._cache[version_id]
            
            # 从存储加载
            result = self._load_feature(version_id)
            if result:
                data, metadata = result
                # 更新缓存
                self._update_cache(version_id, data, metadata)
                logger.info(f"特征加载成功: {feature_name}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"获取特征失败: {e}")
            return None
    
    def _get_latest_version_id(self, feature_name: str) -> Optional[str]:
        """获取最新版本ID"""
        if self.config.backend == StorageBackend.SQLITE:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT version_id FROM feature_versions
                    WHERE feature_name = ? AND status = 'active'
                    ORDER BY created_at DESC LIMIT 1
                ''', (feature_name,))
                row = cursor.fetchone()
                return row[0] if row else None
        else:
            # 文件系统实现
            versions = []
            for version_file in self._version_path.glob("*.json"):
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        info = FeatureVersion.from_dict(json.load(f))
                        if info.feature_name == feature_name and info.status == FeatureStatus.ACTIVE:
                            versions.append((info.created_at, info.version_id))
                except Exception:
                    continue
            
            if versions:
                versions.sort(reverse=True)
                return versions[0][1]
            return None
    
    def _load_feature(self, version_id: str) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """加载特征数据"""
        try:
            if self.config.backend == StorageBackend.SQLITE:
                return self._load_from_sqlite(version_id)
            else:
                return self._load_from_filesystem(version_id)
        except Exception as e:
            logger.error(f"加载特征数据失败: {e}")
            return None
    
    def _load_from_sqlite(self, version_id: str) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """从SQLite加载"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 加载数据
            cursor.execute('''
                SELECT data, compressed FROM feature_data WHERE version_id = ?
            ''', (version_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            data_bytes, compressed = row
            if compressed:
                data_bytes = gzip.decompress(data_bytes)
            data = pickle.loads(data_bytes)
            
            # 加载元数据
            cursor.execute('''
                SELECT * FROM feature_metadata WHERE version_id = ?
            ''', (version_id,))
            row = cursor.fetchone()
            if row:
                metadata = FeatureMetadata(
                    feature_name=row[2],
                    feature_type=row[3],
                    data_shape=tuple(json.loads(row[4])),
                    data_size_bytes=row[5],
                    checksum=row[6],
                    column_names=json.loads(row[7]),
                    column_types=json.loads(row[8]),
                    statistics=json.loads(row[9]),
                    params=json.loads(row[10])
                )
            else:
                metadata = None
            
            return data, metadata
    
    def _load_from_filesystem(self, version_id: str) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """从文件系统加载"""
        # 加载数据
        data_file = self._data_path / f"{version_id}.pkl"
        if not data_file.exists():
            return None
        
        if self.config.compression:
            data = pd.read_pickle(data_file, compression='gzip')
        else:
            data = pd.read_pickle(data_file)
        
        # 加载元数据
        metadata_file = self._metadata_path / f"{version_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = FeatureMetadata.from_dict(json.load(f))
        else:
            metadata = None
        
        return data, metadata
    
    def get_metadata(self, version_id: str) -> Optional[FeatureMetadata]:
        """
        获取特征元数据
        
        Args:
            version_id: 版本ID
            
        Returns:
            特征元数据或None
        """
        try:
            if self.config.backend == StorageBackend.SQLITE:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT feature_type, data_shape, data_size_bytes, checksum,
                               column_names, column_types, statistics, params
                        FROM feature_metadata WHERE version_id = ?
                    ''', (version_id,))
                    row = cursor.fetchone()
                    if row:
                        return FeatureMetadata(
                            feature_name="",
                            feature_type=row[0],
                            data_shape=tuple(json.loads(row[1])),
                            data_size_bytes=row[2],
                            checksum=row[3],
                            column_names=json.loads(row[4]),
                            column_types=json.loads(row[5]),
                            statistics=json.loads(row[6]),
                            params=json.loads(row[7])
                        )
            else:
                metadata_file = self._metadata_path / f"{version_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        return FeatureMetadata.from_dict(json.load(f))
            return None
        except Exception as e:
            logger.error(f"获取元数据失败: {e}")
            return None
    
    def list_features(
        self,
        feature_name: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureVersion]:
        """
        列出特征版本
        
        Args:
            feature_name: 特征名称过滤
            status: 状态过滤
            tags: 标签过滤
            
        Returns:
            特征版本列表
        """
        try:
            if self.config.backend == StorageBackend.SQLITE:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    query = "SELECT * FROM feature_versions WHERE 1=1"
                    params = []
                    
                    if feature_name:
                        query += " AND feature_name = ?"
                        params.append(feature_name)
                    if status:
                        query += " AND status = ?"
                        params.append(status.value)
                    
                    query += " ORDER BY created_at DESC"
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    versions = []
                    for row in rows:
                        version = FeatureVersion(
                            version_id=row[0],
                            feature_name=row[1],
                            version=row[2],
                            created_at=datetime.fromisoformat(row[3]),
                            updated_at=datetime.fromisoformat(row[4]),
                            author=row[5],
                            description=row[6],
                            tags=json.loads(row[7]),
                            dependencies=json.loads(row[8]),
                            status=FeatureStatus(row[9])
                        )
                        
                        # 标签过滤
                        if tags and not any(tag in version.tags for tag in tags):
                            continue
                        
                        versions.append(version)
                    
                    return versions
            else:
                versions = []
                for version_file in self._version_path.glob("*.json"):
                    try:
                        with open(version_file, 'r', encoding='utf-8') as f:
                            version = FeatureVersion.from_dict(json.load(f))
                            
                            if feature_name and version.feature_name != feature_name:
                                continue
                            if status and version.status != status:
                                continue
                            if tags and not any(tag in version.tags for tag in tags):
                                continue
                            
                            versions.append(version)
                    except Exception:
                        continue
                
                versions.sort(key=lambda x: x.created_at, reverse=True)
                return versions
                
        except Exception as e:
            logger.error(f"列出特征失败: {e}")
            return []
    
    def delete_feature(self, version_id: str, soft_delete: bool = True) -> bool:
        """
        删除特征
        
        Args:
            version_id: 版本ID
            soft_delete: 是否软删除
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                if soft_delete:
                    # 软删除：更新状态
                    if self.config.backend == StorageBackend.SQLITE:
                        with self._get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE feature_versions SET status = 'deleted'
                                WHERE version_id = ?
                            ''', (version_id,))
                            conn.commit()
                    else:
                        version_file = self._version_path / f"{version_id}.json"
                        if version_file.exists():
                            with open(version_file, 'r', encoding='utf-8') as f:
                                version = FeatureVersion.from_dict(json.load(f))
                            version.status = FeatureStatus.DELETED
                            with open(version_file, 'w', encoding='utf-8') as f:
                                json.dump(version.to_dict(), f, indent=2)
                else:
                    # 硬删除
                    if self.config.backend == StorageBackend.SQLITE:
                        with self._get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('DELETE FROM feature_data WHERE version_id = ?', (version_id,))
                            cursor.execute('DELETE FROM feature_metadata WHERE version_id = ?', (version_id,))
                            cursor.execute('DELETE FROM feature_versions WHERE version_id = ?', (version_id,))
                            conn.commit()
                    else:
                        # 删除文件
                        for path in [self._data_path, self._metadata_path, self._version_path]:
                            for ext in ['.pkl', '.json']:
                                file_path = path / f"{version_id}{ext}"
                                if file_path.exists():
                                    file_path.unlink()
                
                # 清理缓存
                if version_id in self._cache:
                    del self._cache[version_id]
                    del self._cache_access_time[version_id]
                
                logger.info(f"特征删除成功: {version_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除特征失败: {e}")
            return False
    
    def _update_cache(self, version_id: str, data: pd.DataFrame, metadata: FeatureMetadata) -> None:
        """更新缓存"""
        # 检查缓存大小
        if len(self._cache) >= self.config.max_cache_size:
            # LRU淘汰
            oldest_id = min(self._cache_access_time, key=self._cache_access_time.get)
            del self._cache[oldest_id]
            del self._cache_access_time[oldest_id]
        
        self._cache[version_id] = (data, metadata)
        self._cache_access_time[version_id] = datetime.now()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_access_time.clear()
            logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.config.max_cache_size,
            'cache_entries': list(self._cache.keys())
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        try:
            features = self.list_features(status=FeatureStatus.ACTIVE)
            total_size = sum(
                (self.get_metadata(v.version_id) or FeatureMetadata("", "", (0, 0), 0, "")).data_size_bytes
                for v in features
            )
            
            return {
                'total_features': len(features),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'storage_path': str(self._storage_path),
                'backend': self.config.backend.value
            }
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}
    
    def close(self) -> None:
        """关闭存储"""
        self.clear_cache()
        logger.info("特征存储已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
