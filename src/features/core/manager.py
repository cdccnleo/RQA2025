import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征管理器模块"""
import pandas as pd
from src.infrastructure.logging.core.unified_logger import get_unified_logger
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import hashlib
from pathlib import Path

from .feature_config import FeatureType
# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
    logger = logging.getLogger(__name__)
except ImportError:
    # 降级到直接导入
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger('__name__')


@dataclass
class FeatureMetadata:

    """特征元数据"""
    name: str
    feature_type: FeatureType
    description: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'feature_type': self.feature_type.value,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """从字典创建"""
        if 'feature_type' in data and isinstance(data['feature_type'], str):
            data['feature_type'] = FeatureType(data['feature_type'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)


class FeatureManager:

    """特征管理器"""

    def __init__(self, cache_dir: str = "./feature_cache"):
        """
        初始化特征管理器

        Args:
            cache_dir: 缓存目录
        """
        # 初始化logger
        self.logger = get_unified_logger('__name__')

        self.cache_dir = Path(os.path.join(os.path.dirname(__file__), cache_dir))  # 修正缓存目录路径
        self.cache_dir = Path(self.cache_dir).resolve()  # 确保是绝对路径
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 特征注册表
        self.features: Dict[str, FeatureMetadata] = {}

        # 缓存统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'deletes': 0
        }

        # 加载现有特征
        self._load_feature_registry()

    def _load_feature_registry(self):
        """加载特征注册表"""
        registry_file = self.cache_dir / "feature_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf - 8') as f:
                    data = json.load(f)
                    for feature_data in data.get('features', []):
                        metadata = FeatureMetadata.from_dict(feature_data)
                        self.features[metadata.name] = metadata
                self.logger.info(f"加载了 {len(self.features)} 个特征")
            except Exception as e:
                self.logger.error(f"加载特征注册表失败: {e}")

    def _save_feature_registry(self):
        """保存特征注册表"""
        registry_file = self.cache_dir / "feature_registry.json"
        try:
            data = {
                'features': [feature.to_dict() for feature in self.features.values()],
                'last_updated': datetime.now().isoformat()
            }
            with open(registry_file, 'w', encoding='utf - 8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存特征注册表失败: {e}")

    def register_feature(self, metadata: FeatureMetadata) -> bool:
        """
        注册特征

        Args:
            metadata: 特征元数据

        Returns:
            是否成功
        """
        try:
            # 检查特征是否已存在
            if metadata.name in self.features:
                existing = self.features[metadata.name]
                if existing.version == metadata.version:
                    self.logger.warning(f"特征 {metadata.name} 版本 {metadata.version} 已存在")
                    return False
                else:
                    self.logger.info(
                        f"更新特征 {metadata.name} 从版本 {existing.version} 到 {metadata.version}")

            # 注册特征
            metadata.updated_at = datetime.now()
            self.features[metadata.name] = metadata

            # 保存注册表
            self._save_feature_registry()

            self.logger.info(f"注册特征: {metadata.name} (版本 {metadata.version})")
            return True

        except Exception as e:
            self.logger.error(f"注册特征失败: {e}")
            return False

    def unregister_feature(self, name: str) -> bool:
        """
        注销特征

        Args:
            name: 特征名称

        Returns:
            是否成功
        """
        try:
            if name in self.features:
                del self.features[name]
                self._save_feature_registry()
                self.logger.info(f"注销特征: {name}")
                return True
            else:
                self.logger.warning(f"特征 {name} 不存在")
                return False
        except Exception as e:
            self.logger.error(f"注销特征失败: {e}")
            return False

    def get_feature(self, name: str) -> Optional[FeatureMetadata]:
        """
        获取特征元数据

        Args:
            name: 特征名称

        Returns:
            特征元数据
        """
        return self.features.get(name)

    def list_features(self, feature_type: Optional[FeatureType] = None,


                      enabled_only: bool = True) -> List[FeatureMetadata]:
        """
        列出特征

        Args:
            feature_type: 特征类型过滤
            enabled_only: 是否只返回启用的特征

        Returns:
            特征列表
        """
        features = []
        for feature in self.features.values():
            if enabled_only and not feature.enabled:
                continue
            if feature_type and feature.feature_type != feature_type:
                continue
            features.append(feature)

        return features

    def enable_feature(self, name: str) -> bool:
        """
        启用特征

        Args:
            name: 特征名称

        Returns:
            是否成功
        """
        feature = self.get_feature(name)
        if feature:
            feature.enabled = True
            feature.updated_at = datetime.now()
            self._save_feature_registry()
            self.logger.info(f"启用特征: {name}")
            return True
        else:
            self.logger.warning(f"特征 {name} 不存在")
            return False

    def disable_feature(self, name: str) -> bool:
        """
        禁用特征

        Args:
            name: 特征名称

        Returns:
            是否成功
        """
        feature = self.get_feature(name)
        if feature:
            feature.enabled = False
            feature.updated_at = datetime.now()
            self._save_feature_registry()
            self.logger.info(f"禁用特征: {name}")
            return True
        else:
            self.logger.warning(f"特征 {name} 不存在")
            return False

    def _get_cache_key(self, feature_name: str, data_hash: str) -> str:
        """生成缓存键"""
        return f"{feature_name}_{data_hash}"

    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希值"""
        # 使用数据的前几行和最后几行计算哈希
        sample_data = pd.concat([data.head(5), data.tail(5)])
        data_str = sample_data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def get_cached_features(self, data: pd.DataFrame, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        获取缓存的特征

        Args:
            data: 输入数据
            feature_names: 特征名称列表

        Returns:
            缓存的特征数据
        """
        data_hash = self._get_data_hash(data)

        for feature_name in feature_names:
            cache_key = self._get_cache_key(feature_name, data_hash)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                try:
                    cached_data = pd.read_pickle(cache_file)
                    self.cache_stats['hits'] += 1
                    self.logger.debug(f"缓存命中: {feature_name}")
                    return cached_data
                except Exception as e:
                    self.logger.warning(f"读取缓存失败: {e}")

        self.cache_stats['misses'] += 1
        return None

    def cache_features(self, data: pd.DataFrame, feature_names: List[str],


                       features_data: pd.DataFrame) -> bool:
        """
        缓存特征数据

        Args:
            data: 输入数据
            feature_names: 特征名称列表
            features_data: 特征数据

        Returns:
            是否成功
        """
        try:
            data_hash = self._get_data_hash(data)

            for feature_name in feature_names:
                cache_key = self._get_cache_key(feature_name, data_hash)
                cache_file = self.cache_dir / f"{cache_key}.pkl"

                # 只保存相关的特征列
                if feature_name in features_data.columns:
                    feature_data = features_data[[feature_name]]
                    feature_data.to_pickle(cache_file)

            self.cache_stats['saves'] += 1
            self.logger.debug(f"缓存特征: {feature_names}")
            return True

        except Exception as e:
            self.logger.error(f"缓存特征失败: {e}")
            return False

    def clear_cache(self, feature_name: Optional[str] = None) -> int:
        """
        清理缓存

        Args:
            feature_name: 特征名称，如果为None则清理所有缓存

        Returns:
            清理的文件数量
        """
        try:
            if feature_name:
                # 清理特定特征的缓存
                pattern = f"{feature_name}_*.pkl"
                cache_files = list(self.cache_dir.glob(pattern))
            else:
                # 清理所有缓存
                cache_files = list(self.cache_dir.glob("*.pkl"))
                # 排除注册表文件
                cache_files = [f for f in cache_files if f.name != "feature_registry.json"]

            deleted_count = 0
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"删除缓存文件失败 {cache_file}: {e}")

            self.cache_stats['deletes'] += deleted_count
            self.logger.info(f"清理了 {deleted_count} 个缓存文件")
            return deleted_count

        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'saves': self.cache_stats['saves'],
            'deletes': self.cache_stats['deletes'],
            'hit_rate': self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0.0,
            'cache_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024)
        }

    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """
        获取特征依赖

        Args:
            feature_name: 特征名称

        Returns:
            依赖列表
        """
        feature = self.get_feature(feature_name)
        if feature:
            return feature.dependencies
        return []

    def validate_feature_dependencies(self, feature_name: str) -> bool:
        """
        验证特征依赖

        Args:
            feature_name: 特征名称

        Returns:
            依赖是否完整
        """
        dependencies = self.get_feature_dependencies(feature_name)
        for dep in dependencies:
            if dep not in self.features:
                self.logger.warning(f"特征 {feature_name} 的依赖 {dep} 不存在")
                return False
        return True

    def export_feature_registry(self, file_path: str) -> bool:
        """
        导出特征注册表

        Args:
            file_path: 导出文件路径

        Returns:
            是否成功
        """
        try:
            data = {
                'features': [feature.to_dict() for feature in self.features.values()],
                'exported_at': datetime.now().isoformat(),
                'total_features': len(self.features)
            }

            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"导出特征注册表到: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"导出特征注册表失败: {e}")
            return False

    def import_feature_registry(self, file_path: str, overwrite: bool = False) -> int:
        """
        导入特征注册表

        Args:
            file_path: 导入文件路径
            overwrite: 是否覆盖现有特征

        Returns:
            导入的特征数量
        """
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                data = json.load(f)

            imported_count = 0
            for feature_data in data.get('features', []):
                metadata = FeatureMetadata.from_dict(feature_data)

                if metadata.name in self.features and not overwrite:
                    self.logger.warning(f"跳过已存在的特征: {metadata.name}")
                    continue

                self.features[metadata.name] = metadata
                imported_count += 1

            self._save_feature_registry()
            self.logger.info(f"导入了 {imported_count} 个特征")
            return imported_count

        except Exception as e:
            self.logger.error(f"导入特征注册表失败: {e}")
            return 0


# 导出主要类
__all__ = ['FeatureMetadata', 'FeatureManager']
