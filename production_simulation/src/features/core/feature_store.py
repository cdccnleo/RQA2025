import logging
"""特征存储管理器"""
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta
import threading

from .config import FeatureRegistrationConfig
from .config_integration import get_config_integration_manager, ConfigScope

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


@dataclass
class StoreConfig:

    """存储配置"""
    base_path: str = "./feature_cache"
    max_size_mb: int = 1024  # 最大存储大小(MB)
    ttl_hours: int = 24  # 特征生存时间(小时)
    compression: bool = True  # 是否启用压缩
    use_filesystem: bool = True  # 使用文件系统存储
    max_workers: int = 4  # 最大工作线程数


class FeatureStore:

    """特征存储管理器"""

    def __init__(self, config: StoreConfig = None):

        # 配置集成
        self.config_manager = get_config_integration_manager()
        store_config = self.config_manager.get_config(ConfigScope.PROCESSING)
        if store_config:
            base_path = store_config.get('base_path', None)
            max_size_mb = store_config.get('max_size_mb', None)
            ttl_hours = store_config.get('ttl_hours', None)
            compression = store_config.get('compression', None)
            use_filesystem = store_config.get('use_filesystem', None)
            max_workers = store_config.get('max_workers', None)
            # 合并配置
            config = config or StoreConfig()
            if base_path is not None:
                config.base_path = base_path
            if max_size_mb is not None:
                config.max_size_mb = max_size_mb
            if ttl_hours is not None:
                config.ttl_hours = ttl_hours
            if compression is not None:
                config.compression = compression
            if use_filesystem is not None:
                config.use_filesystem = use_filesystem
            if max_workers is not None:
                config.max_workers = max_workers
        self.config = config or StoreConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        # 初始化存储结构
        self.metadata_path = self.base_path / "metadata"
        self.data_path = self.base_path / "data"
        for path in [self.metadata_path, self.data_path]:
            path.mkdir(exist_ok=True)
        # 线程锁
        self._lock = threading.Lock()
        # 性能统计
        self.stats = {
            'total_stored': 0,
            'total_loaded': 0,
            'total_deleted': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        # 注册配置变更监听器
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

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
        存储特征数据

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
                # 生成特征ID
                feature_id = self._generate_feature_id(feature_name, config.params)

                # 检查是否已存在
                if self._feature_exists(feature_id):
                    logger.info(f"特征 {feature_name} 已存在，更新中...")
                    return self._update_feature(feature_id, data, config, description, tags)

                # 创建元数据
                metadata = FeatureMetadata(
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
                    tags=tags or []
                )

                # 保存数据
                success = self._save_feature_data(feature_id, data)
                if not success:
                    return False

                # 保存元数据
                success = self._save_metadata(feature_id, metadata)
                if not success:
                    # 清理已保存的数据
                    self._delete_feature_data(feature_id)
                    return False

                self.stats['total_stored'] += 1
                logger.info(f"特征 {feature_name} 存储成功，ID: {feature_id}")

                return True

        except Exception as e:
            logger.error(f"存储特征 {feature_name} 失败: {e}")
            return False

    def load_feature(


        self,
        feature_name: str,
        params: Dict[str, Any] = None
    ) -> Optional[Tuple[pd.DataFrame, FeatureMetadata]]:
        """
        加载特征数据

        Args:
            feature_name: 特征名称
            params: 特征参数

        Returns:
            (特征数据, 元数据) 或 None
        """
        try:
            # 生成特征ID
            feature_id = self._generate_feature_id(feature_name, params or {})

            # 检查是否存在
            if not self._feature_exists(feature_id):
                self.stats['cache_misses'] += 1
                logger.info(f"特征 {feature_name} 不存在")
                return None

            # 检查是否过期
            metadata = self._load_metadata(feature_id)
            if metadata and self._is_expired(metadata):
                logger.info(f"特征 {feature_name} 已过期，删除中...")
                self.delete_feature(feature_id)
                self.stats['cache_misses'] += 1
                return None

            # 加载数据
            data = self._load_feature_data(feature_id)
            if data is None:
                return None

            self.stats['cache_hits'] += 1
            self.stats['total_loaded'] += 1

            logger.info(f"特征 {feature_name} 加载成功")
            return data, metadata

        except Exception as e:
            logger.error(f"加载特征 {feature_name} 失败: {e}")
            return None

    def delete_feature(self, feature_id: str) -> bool:
        """删除特征"""
        try:
            with self._lock:
                # 删除数据文件
                self._delete_feature_data(feature_id)

                # 删除元数据
                self._delete_metadata(feature_id)

                self.stats['total_deleted'] += 1
                logger.info(f"特征 {feature_id} 删除成功")
                return True

        except Exception as e:
            logger.error(f"删除特征 {feature_id} 失败: {e}")
            return False

    def list_features(


        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """列出特征"""
        try:
            return self._list_features_filesystem(feature_type, tags)
        except Exception as e:
            logger.error(f"列出特征失败: {e}")
            return []

    def cleanup_expired(self) -> int:
        """清理过期特征"""
        try:
            expired_count = 0
            features = self.list_features()

            for metadata in features:
                if self._is_expired(metadata):
                    feature_id = self._generate_feature_id(
                        metadata.feature_name, metadata.params
                    )
                    if self.delete_feature(feature_id):
                        expired_count += 1

            logger.info(f"清理了 {expired_count} 个过期特征")
            return expired_count

        except Exception as e:
            logger.error(f"清理过期特征失败: {e}")
            return 0

    def get_store_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = self.stats.copy()

        # 计算缓存命中率
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['hit_rate'] = 0.0

        # 计算存储使用情况
        total_size = 0
        feature_count = 0

        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf - 8') as f:
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

    def _generate_feature_id(self, feature_name: str, params: Dict[str, Any]) -> str:
        """生成特征ID"""
        # 创建参数字符串
        param_str = json.dumps(params, sort_keys=True)

        # 生成哈希
        content = f"{feature_name}:{param_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _feature_exists(self, feature_id: str) -> bool:
        """检查特征是否存在"""
        metadata_file = self.metadata_path / f"{feature_id}.json"
        return metadata_file.exists()

    def _save_feature_data(self, feature_id: str, data: pd.DataFrame) -> bool:
        """保存特征数据"""
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
        """加载特征数据"""
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
        """删除特征数据"""
        try:
            data_file = self.data_path / f"{feature_id}.pkl"
            if data_file.exists():
                data_file.unlink()
        except Exception as e:
            logger.error(f"删除特征数据失败: {e}")

    def _save_metadata(self, feature_id: str, metadata: FeatureMetadata) -> bool:
        """保存元数据"""
        try:
            metadata_file = self.metadata_path / f"{feature_id}.json"
            with open(metadata_file, 'w', encoding='utf - 8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            return False

    def _load_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """加载元数据"""
        try:
            metadata_file = self.metadata_path / f"{feature_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    return FeatureMetadata(**self._normalize_metadata_dict(data))

            return None
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return None

    def _delete_metadata(self, feature_id: str):
        """删除元数据"""
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
            # 更新数据
            success = self._save_feature_data(feature_id, data)
            if not success:
                return False

            # 更新元数据
            metadata = self._load_metadata(feature_id)
            if metadata:
                metadata.updated_at = datetime.now()
                metadata.data_shape = data.shape
                metadata.data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                metadata.checksum = self._calculate_checksum(data)
                metadata.description = description
                metadata.tags = tags or []

                return self._save_metadata(feature_id, metadata)

            return False
        except Exception as e:
            logger.error(f"更新特征失败: {e}")
            return False

    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """计算数据校验和"""
        try:
            # 使用数据的哈希值作为校验和
            content = data.to_string()
            return hashlib.md5(content.encode()).hexdigest()
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
                    with open(metadata_file, 'r', encoding='utf - 8') as f:
                        data = self._normalize_metadata_dict(json.load(f))
                        metadata = FeatureMetadata(**data)

                        # 过滤
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

    def _normalize_metadata_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """确保从磁盘载入的元数据字段类型正确"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'data_shape' in data and isinstance(data['data_shape'], list):
            data['data_shape'] = tuple(data['data_shape'])
        return data

    def close(self):
        """关闭存储管理器"""
        # 文件系统存储不需要特殊关闭操作

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
