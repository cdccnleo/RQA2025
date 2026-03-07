#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征版本管理模块
提供特征版本控制、回滚和追踪功能
"""

import json
import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):

    """自定义JSON编码器，处理datetime对象"""

    def default(self, obj):

        if hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat', None)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # 处理pandas dtype对象
            return str(obj)
        return super().default(obj)


@dataclass
class FeatureVersion:

    """特征版本信息"""
    version_id: str
    timestamp: datetime
    feature_names: List[str]
    feature_count: int
    checksum: str
    description: str
    creator: str
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = None
    status: str = "active"  # active, deprecated, deleted


@dataclass
class FeatureChange:

    """特征变更信息"""
    change_id: str
    version_id: str
    change_type: str  # added, removed, modified
    feature_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: datetime = None


class FeatureVersionManager:

    """特征版本管理器"""

    def __init__(self, storage_dir: str = "./feature_versions"):
        """
        初始化特征版本管理器

        Args:
            storage_dir: 存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.versions: Dict[str, FeatureVersion] = {}
        self.changes: Dict[str, List[FeatureChange]] = {}
        self.lock = threading.RLock()

        self._load_version_index()
        logger.info(f"特征版本管理器初始化完成，存储目录: {self.storage_dir}")

    def _load_version_index(self):
        """加载版本索引"""
        index_file = self.storage_dir / "version_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf - 8') as f:
                    data = json.load(f)

                    # 加载版本信息
                    for version_data in data.get('versions', []):
                        version = FeatureVersion(
                            version_id=version_data['version_id'],
                            timestamp=datetime.fromisoformat(version_data['timestamp']),
                            feature_names=version_data['feature_names'],
                            feature_count=version_data['feature_count'],
                            checksum=version_data['checksum'],
                            description=version_data['description'],
                            creator=version_data['creator'],
                            parent_version=version_data.get('parent_version'),
                            metadata=version_data.get('metadata', {}),
                            status=version_data.get('status', 'active')
                        )
                        self.versions[version.version_id] = version

                    # 加载变更信息
                    for change_data in data.get('changes', []):
                        change = FeatureChange(
                            change_id=change_data['change_id'],
                            version_id=change_data['version_id'],
                            change_type=change_data['change_type'],
                            feature_name=change_data['feature_name'],
                            old_value=change_data.get('old_value'),
                            new_value=change_data.get('new_value'),
                            timestamp=datetime.fromisoformat(change_data['timestamp'])
                        )
                        if change.version_id not in self.changes:
                            self.changes[change.version_id] = []
                        self.changes[change.version_id].append(change)

                logger.info(f"加载了 {len(self.versions)} 个特征版本")
            except Exception as e:
                logger.error(f"加载版本索引失败: {e}")

    def _save_version_index(self):
        """保存版本索引"""
        index_file = self.storage_dir / "version_index.json"
        try:
            with self.lock:
                data = {
                    'versions': [
                        {
                            'version_id': version.version_id,
                            'timestamp': version.timestamp.isoformat(),
                            'feature_names': version.feature_names,
                            'feature_count': version.feature_count,
                            'checksum': version.checksum,
                            'description': version.description,
                            'creator': version.creator,
                            'parent_version': version.parent_version,
                            'metadata': version.metadata or {},
                            'status': version.status
                        }
                        for version in self.versions.values()
                    ],
                    'changes': [
                        {
                            'change_id': change.change_id,
                            'version_id': change.version_id,
                            'change_type': change.change_type,
                            'feature_name': change.feature_name,
                            'old_value': change.old_value,
                            'new_value': change.new_value,
                            'timestamp': change.timestamp.isoformat()
                        }
                        for changes in self.changes.values()
                        for change in changes
                    ]
                }
                with open(index_file, 'w', encoding='utf - 8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存版本索引失败: {e}")

    def create_version(self, features: pd.DataFrame, description: str = "",


                       creator: str = "system", parent_version: Optional[str] = None) -> str:
        """
        创建特征版本

        Args:
            features: 特征数据框
            description: 版本描述
            creator: 创建者
            parent_version: 父版本ID

        Returns:
            str: 版本ID
        """
        version_id = f"feature_v{datetime.now().strftime('%Y % m % d_ % H % M % S')}_{int(time.time() * 1000) % 1000}"
        version_path = self.storage_dir / version_id

        try:
            with self.lock:
                # 创建版本目录
                version_path.mkdir(exist_ok=True)

                # 保存特征数据
                features_file = version_path / "features.parquet"
                features.to_parquet(features_file, index=True)

                # 计算特征信息
                feature_names = features.columns.tolist()
                feature_count = len(feature_names)
                checksum = self._calculate_features_checksum(features)

                # 创建版本信息
                version = FeatureVersion(
                    version_id=version_id,
                    timestamp=datetime.now(),
                    feature_names=feature_names,
                    feature_count=feature_count,
                    checksum=checksum,
                    description=description,
                    creator=creator,
                    parent_version=parent_version,
                    metadata={
                        'shape': list(features.shape),  # 转换为list
                        'dtypes': {col: str(dtype) for col, dtype in features.dtypes.items()},
                        'memory_usage': int(features.memory_usage(deep=True).sum())  # 转换为int
                    }
                )

                self.versions[version_id] = version

                # 记录变更（如果是基于父版本）
                if parent_version and parent_version in self.versions:
                    self._record_changes(version_id, parent_version, features)

                # 保存版本信息
                self._save_version_info(version)

                # 保存索引
                self._save_version_index()

                logger.info(f"特征版本创建成功: {version_id}, 特征数量: {feature_count}")
                return version_id

        except Exception as e:
            logger.error(f"创建特征版本失败: {e}")
            if version_path.exists():
                import shutil
                shutil.rmtree(version_path)
            raise

    def get_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """
        获取特征版本

        Args:
            version_id: 版本ID

        Returns:
            Optional[pd.DataFrame]: 特征数据框
        """
        if version_id not in self.versions:
            logger.warning(f"特征版本不存在: {version_id}")
            return None

        version_path = self.storage_dir / version_id
        features_file = version_path / "features.parquet"

        if not features_file.exists():
            logger.error(f"特征文件不存在: {version_id}")
            return None

        try:
            features = pd.read_parquet(features_file)
            logger.info(f"加载特征版本: {version_id}")
            return features
        except Exception as e:
            logger.error(f"加载特征版本失败: {e}")
            return None

    def list_versions(self, status: Optional[str] = None, creator: Optional[str] = None) -> List[FeatureVersion]:
        """
        列出特征版本

        Args:
            status: 状态过滤
            creator: 创建者过滤

        Returns:
            List[FeatureVersion]: 版本列表
        """
        versions = list(self.versions.values())

        if status:
            versions = [v for v in versions if v.status == status]

        if creator:
            versions = [v for v in versions if v.creator == creator]

        return sorted(versions, key=lambda x: x.timestamp, reverse=True)

    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        比较两个版本

        Args:
            version1_id: 版本1 ID
            version2_id: 版本2 ID

        Returns:
            Dict[str, Any]: 比较结果
        """
        if version1_id not in self.versions or version2_id not in self.versions:
            raise ValueError("版本不存在")

        version1 = self.versions[version1_id]
        version2 = self.versions[version2_id]

        # 获取特征数据
        features1 = self.get_version(version1_id)
        features2 = self.get_version(version2_id)

        if features1 is None or features2 is None:
            raise ValueError("无法加载特征数据")

        # 比较特征
        common_features = set(version1.feature_names) & set(version2.feature_names)
        only_in_v1 = set(version1.feature_names) - set(version2.feature_names)
        only_in_v2 = set(version2.feature_names) - set(version1.feature_names)

        # 比较共同特征的值
        value_changes = {}
        for feature in common_features:
            if not features1[feature].equals(features2[feature]):
                value_changes[feature] = {
                    'v1_mean': features1[feature].mean(),
                    'v2_mean': features2[feature].mean(),
                    'v1_std': features1[feature].std(),
                    'v2_std': features2[feature].std()
                }

        return {
            'version1': version1_id,
            'version2': version2_id,
            'common_features': list(common_features),
            'only_in_v1': list(only_in_v1),
            'only_in_v2': list(only_in_v2),
            'value_changes': value_changes,
            'feature_count_diff': version2.feature_count - version1.feature_count
        }

    def rollback_version(self, target_version_id: str, description: str = "") -> str:
        """
        回滚到指定版本

        Args:
            target_version_id: 目标版本ID
            description: 回滚描述

        Returns:
            str: 新版本ID
        """
        if target_version_id not in self.versions:
            raise ValueError(f"目标版本不存在: {target_version_id}")

        # 获取目标版本的特征
        target_features = self.get_version(target_version_id)
        if target_features is None:
            raise ValueError("无法加载目标版本特征")

        # 创建新版本
        rollback_description = f"回滚到版本 {target_version_id}: {description}"
        new_version_id = self.create_version(
            features=target_features,
            description=rollback_description,
            creator="system",
            parent_version=target_version_id
        )

        logger.info(f"版本回滚成功: {target_version_id} -> {new_version_id}")
        return new_version_id

    def delete_version(self, version_id: str) -> bool:
        """
        删除版本（软删除）

        Args:
            version_id: 版本ID

        Returns:
            bool: 是否删除成功
        """
        if version_id not in self.versions:
            logger.warning(f"特征版本不存在: {version_id}")
            return False

        try:
            with self.lock:
                # 标记为删除
                self.versions[version_id].status = "deleted"
                self._save_version_index()

                logger.info(f"特征版本删除成功: {version_id}")
                return True

        except Exception as e:
            logger.error(f"删除特征版本失败: {e}")
            return False

    def get_version_lineage(self, version_id: str) -> Dict[str, Any]:
        """
        获取版本血缘关系

        Args:
            version_id: 版本ID

        Returns:
            Dict[str, Any]: 血缘关系信息
        """
        if version_id not in self.versions:
            raise ValueError(f"版本不存在: {version_id}")

        lineage = {
            'version_id': version_id,
            'ancestors': [],
            'descendants': [],
            'changes': []
        }

        # 查找祖先版本
        current_version = self.versions[version_id]
        while current_version.parent_version:
            parent_id = current_version.parent_version
            if parent_id in self.versions:
                lineage['ancestors'].append({
                    'version_id': parent_id,
                    'description': self.versions[parent_id].description,
                    'timestamp': self.versions[parent_id].timestamp.isoformat()
                })
                current_version = self.versions[parent_id]
            else:
                break

        # 查找后代版本
        for v_id, version in self.versions.items():
            if version.parent_version == version_id:
                lineage['descendants'].append({
                    'version_id': v_id,
                    'description': version.description,
                    'timestamp': version.timestamp.isoformat()
                })

        # 获取变更信息
        if version_id in self.changes:
            lineage['changes'] = [
                {
                    'change_type': change.change_type,
                    'feature_name': change.feature_name,
                    'timestamp': change.timestamp.isoformat()
                }
                for change in self.changes[version_id]
            ]

        return lineage

    def _calculate_features_checksum(self, features: pd.DataFrame) -> str:
        """计算特征校验和"""
        # 使用特征名称和数据的哈希值
        feature_str = "|".join(sorted(features.columns))
        data_hash = hashlib.md5(features.values.tobytes()).hexdigest()
        combined = f"{feature_str}|{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _save_version_info(self, version: FeatureVersion):
        """保存版本信息"""
        version_path = self.storage_dir / version.version_id
        info_file = version_path / "version_info.json"

        with open(info_file, 'w', encoding='utf - 8') as f:
            json.dump(asdict(version), f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

    def _record_changes(self, new_version_id: str, parent_version_id: str, new_features: pd.DataFrame):
        """记录版本变更"""
        parent_features = self.get_version(parent_version_id)
        if parent_features is None:
            return

        changes = []

        # 比较特征
        old_features = set(parent_features.columns)
        new_features_set = set(new_features.columns)

        # 新增的特征
        for feature in new_features_set - old_features:
            change = FeatureChange(
                change_id=f"change_{int(time.time() * 1000)}",
                version_id=new_version_id,
                change_type="added",
                feature_name=feature,
                new_value=new_features[feature].describe().to_dict(),
                timestamp=datetime.now()
            )
            changes.append(change)

        # 删除的特征
        for feature in old_features - new_features_set:
            change = FeatureChange(
                change_id=f"change_{int(time.time() * 1000)}",
                version_id=new_version_id,
                change_type="removed",
                feature_name=feature,
                old_value=parent_features[feature].describe().to_dict(),
                timestamp=datetime.now()
            )
            changes.append(change)

        # 修改的特征
        for feature in old_features & new_features_set:
            if not parent_features[feature].equals(new_features[feature]):
                change = FeatureChange(
                    change_id=f"change_{int(time.time() * 1000)}",
                    version_id=new_version_id,
                    change_type="modified",
                    feature_name=feature,
                    old_value=parent_features[feature].describe().to_dict(),
                    new_value=new_features[feature].describe().to_dict(),
                    timestamp=datetime.now()
                )
                changes.append(change)

        if changes:
            self.changes[new_version_id] = changes

    def get_version_stats(self) -> Dict[str, Any]:
        """获取版本统计信息"""
        total_versions = len(self.versions)
        active_versions = len([v for v in self.versions.values() if v.status == 'active'])
        total_features = sum(v.feature_count for v in self.versions.values())

        return {
            'total_versions': total_versions,
            'active_versions': active_versions,
            'deleted_versions': total_versions - active_versions,
            'total_features': total_features,
            'avg_features_per_version': total_features / total_versions if total_versions > 0 else 0,
            'oldest_version': min(v.timestamp for v in self.versions.values()) if self.versions else None,
            'newest_version': max(v.timestamp for v in self.versions.values()) if self.versions else None
        }
