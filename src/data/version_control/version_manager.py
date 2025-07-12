"""
版本管理器，用于管理数据版本
"""
from typing import Dict, List, Any, Optional, Union, Set
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import hashlib
import pandas as pd
import threading
import copy

from src.data.interfaces import IDataModel
from src.infrastructure.utils.exceptions import DataVersionError, DataLoaderError

logger = logging.getLogger(__name__)


class DataVersionManager:
    """
    版本管理器，用于管理数据版本
    """
    def __init__(self, version_dir: str):
        """
        初始化版本管理器
        
        Args:
            version_dir: 版本目录
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # 版本元数据文件
        self.metadata_file = self.version_dir / 'version_metadata.json'
        self.metadata = self._load_metadata()
        
        # 版本历史记录文件
        self.history_file = self.version_dir / 'version_history.json'
        self.history = self._load_history()
        
        # 版本血缘关系文件
        self.lineage_file = self.version_dir / 'version_lineage.json'
        self.lineage = self._load_lineage()
        
        # 当前版本
        self.current_version = self.metadata.get('latest_version')
        
        # 用于并发控制的锁
        self._lock = threading.Lock()
        
        logger.info(f"VersionManager initialized with directory: {version_dir}")
        logger.info(f"Current version: {self.current_version}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载版本元数据"""
        if not self.metadata_file.exists():
            # 创建默认元数据
            default_metadata = {
                'versions': {},
                'latest_version': None,
                'branches': {'main': None}
            }
            self._save_metadata(default_metadata)
            return default_metadata
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version metadata: {e}")
            # 元数据损坏时使用默认值
            default_metadata = {
                'versions': {},
                'latest_version': None,
                'branches': {'main': None}
            }
            return default_metadata
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """保存版本元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """加载版本历史记录"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            return []
    
    def _save_history(self) -> None:
        """保存版本历史记录"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
    
    def _load_lineage(self) -> Dict[str, List[str]]:
        """加载版本血缘关系"""
        if not self.lineage_file.exists():
            return {}
        
        try:
            with open(self.lineage_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version lineage: {e}")
            return {}
    
    def _update_lineage(self, version: str, parent_version: Optional[str] = None) -> None:
        """
        更新版本血缘关系
        
        Args:
            version: 当前版本
            parent_version: 父版本
        """
        if parent_version:
            if parent_version not in self.lineage:
                self.lineage[parent_version] = []
            
            if version not in self.lineage[parent_version]:
                self.lineage[parent_version].append(version)
        
        # 确保当前版本在血缘关系中有一个条目
        if version not in self.lineage:
            self.lineage[version] = []
        
        # 保存血缘关系
        try:
            with open(self.lineage_file, 'w') as f:
                json.dump(self.lineage, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version lineage: {e}")
    
    def _get_latest_version(self) -> Optional[str]:
        """获取最新版本"""
        return self.metadata.get('latest_version')
    
    def _generate_version(self) -> str:
        """生成版本号"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        if not self.history:
            return f"v_{timestamp}_1"
        
        last_version = self.history[-1].get('version_id', self.history[-1].get('version', ''))
        if last_version.startswith(f"v_{timestamp}"):
            # 同一时间戳，增加序号
            seq = int(last_version.split('_')[-1]) + 1
            return f"v_{timestamp}_{seq}"
        
        return f"v_{timestamp}_1"
    
    def _calculate_hash(self, data_model: IDataModel) -> str:
        """计算数据模型的哈希值"""
        data_str = data_model.data.to_json()
        metadata_str = json.dumps(data_model.get_metadata(), sort_keys=True)
        combined = data_str + metadata_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_ancestors(self, version: str, ancestors: Optional[Set[str]] = None) -> Set[str]:
        """
        获取版本的所有祖先版本
        
        Args:
            version: 版本号
            ancestors: 已收集的祖先版本集合
            
        Returns:
            Set[str]: 祖先版本集合
        """
        if ancestors is None:
            ancestors = set()
        
        # 遍历所有版本的血缘关系
        for parent, children in self.lineage.items():
            if version in children:
                ancestors.add(parent)
                # 递归获取父版本的祖先
                ancestors.update(self._get_ancestors(parent, ancestors))
        
        return ancestors
    
    def create_version(
        self,
        data_model: IDataModel,
        description: str,
        tags: Optional[List[str]] = None,
        creator: Optional[str] = None,
        branch: Optional[str] = None
    ) -> str:
        """
        创建新版本
        
        Args:
            data_model: 数据模型
            description: 版本描述
            tags: 版本标签
            creator: 创建者
            branch: 分支名称
            
        Returns:
            str: 版本号
            
        Raises:
            DataVersionError: 如果创建版本失败
        """
        with self._lock:
            # 生成版本号
            version = self._generate_version()
            
            # 保存数据为parquet文件
            data_file = self.version_dir / f"{version}.parquet"
            
            try:
                # 保存数据
                data_model.data.to_parquet(data_file)
                
                # 计算哈希值
                hash_value = self._calculate_hash(data_model)
                
                # 记录版本信息
                version_info = {
                    'version_id': version,
                    'timestamp': datetime.now().isoformat(),
                    'description': description,
                    'tags': tags or [],
                    'hash': hash_value,
                    'metadata': data_model.get_metadata(),
                    'data_shape': data_model.data.shape,
                    'data_columns': data_model.data.columns.tolist(),
                    'creator': creator or 'system',
                    'branch': branch or 'main'
                }
                
                # 更新元数据
                self.metadata['versions'][version] = version_info
                self.metadata['latest_version'] = version
                
                # 更新分支信息
                branch_name = branch or 'main'
                if branch_name not in self.metadata['branches']:
                    self.metadata['branches'][branch_name] = version
                else:
                    self.metadata['branches'][branch_name] = version
                
                # 保存元数据
                self._save_metadata(self.metadata)
                
                # 更新历史记录
                self.history.append(version_info)
                self._save_history()
                
                # 更新当前版本
                self.current_version = version
                
                # 更新版本血缘关系
                parent_version = self.current_version if self.current_version != version else None
                self._update_lineage(version, parent_version)
                
                logger.info(f"Created new version: {version}")
                return version
            except Exception as e:
                # 清理失败的版本文件
                if data_file.exists():
                    os.remove(data_file)
                
                logger.error(f"Failed to create version: {e}")
                raise DataVersionError(f"Failed to create version: {e}")
    
    def get_version(self, version: Optional[str] = None) -> Optional[IDataModel]:
        """
        获取指定版本的数据模型
        
        Args:
            version: 版本号，如果为None则获取当前版本
            
        Returns:
            IDataModel: 数据模型，如果版本不存在则返回None
        """
        if version is None:
            version = self.current_version
        
        if version is None:
            return None
        
        # 检查版本是否存在
        data_file = self.version_dir / f"{version}.parquet"
        if not data_file.exists():
            return None
        
        try:
            # 加载数据
            data = pd.read_parquet(data_file)
            
            # 获取元数据
            version_info = self.metadata['versions'].get(version)
            if not version_info:
                # 尝试从历史记录中获取
                version_info = next((v for v in self.history if v.get('version_id') == version), None)
                if not version_info:
                    return None
            
            metadata = version_info.get('metadata', {})
            
            # 创建数据模型
            from src.data.data_manager import DataModel
            data_model = DataModel(data, metadata)
            
            logger.info(f"Loaded version: {version}")
            return data_model
        except Exception as e:
            logger.error(f"Failed to load version {version}: {e}")
            return None
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        获取版本信息
        
        Args:
            version: 版本号
            
        Returns:
            Dict[str, Any]: 版本信息，如果版本不存在则返回None
        """
        # 从元数据中获取版本信息
        version_info = self.metadata['versions'].get(version)
        if not version_info:
            # 尝试从历史记录中获取
            version_info = next((v for v in self.history if v.get('version_id') == version), None)
        
        return version_info
    
    def get_lineage(self, version: str) -> Dict[str, Any]:
        """
        获取版本的血缘关系
        
        Args:
            version: 版本号
            
        Returns:
            Dict[str, Any]: 血缘关系信息
        """
        ancestors = self._get_ancestors(version)
        ancestors_info = []
        for ancestor in ancestors:
            info = self.get_version_info(ancestor)
            if info:
                ancestors_info.append({
                    'version_id': ancestor,
                    'timestamp': info.get('timestamp'),
                    'description': info.get('description')
                })
        
        return {
            'version_id': version,
            'ancestors': ancestors_info
        }
    
    def list_versions(
        self,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        creator: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        列出版本
        
        Args:
            limit: 返回的版本数量限制
            tags: 按标签筛选
            creator: 按创建者筛选
            branch: 按分支筛选
            
        Returns:
            List[Dict[str, Any]]: 版本列表
        """
        versions = list(self.metadata['versions'].values())
        
        # 按标签筛选
        if tags:
            versions = [
                v for v in versions
                if any(tag in v.get('tags', []) for tag in tags)
            ]
        
        # 按创建者筛选
        if creator:
            versions = [
                v for v in versions
                if v.get('creator') == creator
            ]
        
        # 按分支筛选
        if branch:
            versions = [
                v for v in versions
                if v.get('branch') == branch
            ]
        
        # 按时间戳排序
        versions.sort(key=lambda x: x.get('timestamp', ''))
        
        # 限制数量
        if limit is not None:
            return versions[-limit:]
        
        return versions
        
    def delete_version(self, version: str) -> bool:
        """
        删除版本
        
        Args:
            version: 版本号
            
        Returns:
            bool: 是否成功删除
            
        Raises:
            DataVersionError: 如果删除版本失败
        """
        with self._lock:
            # 检查版本是否存在
            data_file = self.version_dir / f"{version}.parquet"
            if not data_file.exists():
                raise DataVersionError(f"Version {version} does not exist")
            
            # 检查是否为当前版本
            if version == self.current_version:
                raise DataVersionError(f"Cannot delete current version {version}")
            
            try:
                # 删除版本文件
                os.remove(data_file)
                
                # 更新元数据
                if version in self.metadata['versions']:
                    del self.metadata['versions'][version]
                
                # 更新分支信息
                for branch, branch_version in list(self.metadata['branches'].items()):
                    if branch_version == version:
                        # 找到该分支的前一个版本
                        branch_versions = [v for v in self.history if v.get('branch') == branch]
                        prev_version = None
                        for v in reversed(branch_versions):
                            if v.get('version_id') != version:
                                prev_version = v.get('version_id')
                                break
                        self.metadata['branches'][branch] = prev_version
                
                # 保存元数据
                self._save_metadata(self.metadata)
                
                # 更新历史记录
                self.history = [v for v in self.history if v.get('version_id') != version]
                self._save_history()
                
                # 更新血缘关系
                if version in self.lineage:
                    # 获取子版本
                    children = self.lineage[version]
                    # 获取父版本
                    parents = set()
                    for parent, parent_children in self.lineage.items():
                        if version in parent_children:
                            parents.add(parent)
                    
                    # 删除当前版本的血缘关系
                    del self.lineage[version]
                    
                    # 更新子版本的血缘关系
                    for parent in parents:
                        if parent in self.lineage:
                            self.lineage[parent].extend(children)
                            self.lineage[parent] = list(set(self.lineage[parent]))
                            self.lineage[parent].remove(version)
                
                # 保存血缘关系
                with open(self.lineage_file, 'w') as f:
                    json.dump(self.lineage, f, indent=2)
                
                logger.info(f"Deleted version: {version}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete version {version}: {e}")
                raise DataVersionError(f"Failed to delete version {version}: {e}")
    
    def rollback(self, version: str) -> str:
        """
        回滚到指定版本
        
        Args:
            version: 目标版本号
            
        Returns:
            str: 新创建的版本号
            
        Raises:
            DataVersionError: 如果回滚失败
        """
        # 获取目标版本
        target_model = self.get_version(version)
        if target_model is None:
            raise DataVersionError(f"Version {version} not found")
        
        try:
            # 获取目标版本信息
            target_info = self.get_version_info(version)
            
            # 创建新版本
            new_version = self.create_version(
                target_model,
                description=f"Rollback to version {version}",
                tags=['rollback', f'from_{version}'],
                creator=target_info.get('creator'),
                branch=target_info.get('branch')
            )
            
            logger.info(f"Rolled back to version {version}, created new version {new_version}")
            return new_version
        except Exception as e:
            logger.error(f"Failed to rollback to version {version}: {e}")
            raise DataVersionError(f"Failed to rollback to version {version}: {e}")
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            version1: 版本1
            version2: 版本2
            
        Returns:
            Dict[str, Any]: 比较结果，包含metadata_diff和data_diff
            
        Raises:
            DataVersionError: 如果比较版本失败
        """
        # 获取两个版本的数据模型
        model1 = self.get_version(version1)
        model2 = self.get_version(version2)
        
        if model1 is None or model2 is None:
            raise DataVersionError(f"One or both versions do not exist: {version1}, {version2}")
        
        try:
            # 比较元数据
            metadata1 = model1.get_metadata()
            metadata2 = model2.get_metadata()
            
            metadata_diff = {
                'added': {k: v for k, v in metadata2.items() if k not in metadata1},
                'removed': {k: v for k, v in metadata1.items() if k not in metadata2},
                'changed': {
                    k: {'from': metadata1[k], 'to': metadata2[k]}
                    for k in metadata1.keys() & metadata2.keys()
                    if metadata1[k] != metadata2[k]
                }
            }
            
            # 比较数据结构
            data_diff = {
                'shape_diff': {
                    'rows': model2.data.shape[0] - model1.data.shape[0],
                    'columns': model2.data.shape[1] - model1.data.shape[1]
                },
                'columns_diff': {
                    'added': list(set(model2.data.columns) - set(model1.data.columns)),
                    'removed': list(set(model1.data.columns) - set(model2.data.columns))
                }
            }
            
            # 比较数据值
            common_columns = set(model1.data.columns) & set(model2.data.columns)
            value_diff = {}
            
            for col in common_columns:
                if not model1.data[col].equals(model2.data[col]):
                    value_diff[col] = {
                        'changed_rows': (model1.data[col] != model2.data[col]).sum(),
                        'null_diff': model2.data[col].isnull().sum() - model1.data[col].isnull().sum()
                    }
            
            data_diff['value_diff'] = value_diff
            
            return {
                'metadata_diff': metadata_diff,
                'data_diff': data_diff
            }
        except Exception as e:
            logger.error(f"Failed to compare versions {version1} and {version2}: {e}")
            raise DataVersionError(f"Failed to compare versions: {e}")
