"""
版本管理器，用于管理数据版本
"""
# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    import logging

    def get_infrastructure_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import hashlib
import threading
import re

# 安全导入pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # 创建一个简单的pandas模拟

    class MockPandas:

        class DataFrame:

            def __init__(self, data=None):

                self.data = data or {}

            def to_dict(self):

                return self.data

            def __len__(self):

                return len(self.data) if isinstance(self.data, dict) else 0

    pd = MockPandas()

# 在文件顶部导入DataModel
try:
    from ...data_manager import DataModel
except ImportError:
    try:
        from ...models import SimpleDataModel as DataModel
    except ImportError:
        # 如果导入失败，创建一个完整的DataModel类

        class DataModel:

            def __init__(self, data=None, frequency='1d', metadata=None):

                self.data = data
                self._frequency = frequency
                self._user_metadata = dict(metadata) if metadata else {}
                self._metadata = dict(self._user_metadata)
                if not metadata or 'created_at' not in self._metadata:
                    self._metadata['created_at'] = datetime.now().isoformat()
                self._metadata.update({
                    'data_shape': data.shape if hasattr(data, 'shape') else None,
                    'data_columns': data.columns.tolist() if hasattr(data, 'columns') else None,
                })

            def validate(self):

                if self.data is None or (hasattr(self.data, 'empty') and self.data.empty):
                    return False
                return True

            def get_frequency(self):

                return self._frequency

            def get_metadata(self, user_only=False):

                if user_only:
                    return self._user_metadata
                return self._metadata

            def from_dict(self, data_dict):

                data = data_dict.get('data', {})
                frequency = data_dict.get('frequency', '1d')
                metadata = data_dict.get('metadata', {})
                return DataModel(data, frequency, metadata)

            def to_dict(self):

                return {
                    'data': self.data.to_dict() if hasattr(self.data, 'to_dict') else self.data,
                    'frequency': self._frequency,
                    'metadata': self._metadata
                }

# 在文件顶部导入DataVersionError
try:
    from src.infrastructure.utils.exceptions import DataVersionError
except ImportError:
    # 如果导入失败，创建一个简单的异常类

    class DataVersionError(Exception):

        pass

logger = get_infrastructure_logger('__name__')


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
        self._cache_dir = self.version_dir

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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not self.history:
            version = f"v_{timestamp}_1"
        else:
            last_version = self.history[-1].get('version_id', self.history[-1].get('version', ''))
            if last_version.startswith(f"v_{timestamp}"):
                # 同一时间戳，增加序号
                seq = int(last_version.split('_')[-1]) + 1
                version = f"v_{timestamp}_{seq}"
            else:
                version = f"v_{timestamp}_1"
        version = re.sub(r'[^a-zA-Z0-9_]', '_', version)
        return version

    def _calculate_hash(self, data_model: DataModel) -> str:
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
        data_model: DataModel,
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
                # 验证数据模型是否有效
                if data_model is None or data_model.data is None:
                    raise ValueError("DataModel or data is None")
                
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
                    'metadata': dict(getattr(data_model, '_user_metadata', {})),
                    'data_shape': data_model.data.shape if data_model.data is not None else (0, 0),
                    'data_columns': data_model.data.columns.tolist() if data_model.data is not None else [],
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

                # 记录更新前的当前版本用于血缘
                previous_current = self.current_version

                # 更新当前版本
                self.current_version = version

                # 更新版本血缘关系（父版本应是更新前的 current）
                parent_version = previous_current if previous_current and previous_current != version else None
                self._update_lineage(version, parent_version)

                logger.info(f"Created new version: {version}")
                return version
            except Exception as e:
                # 清理失败的版本文件
                if data_file.exists():
                    os.remove(data_file)

                logger.error(f"Failed to create version: {e}")
                raise DataVersionError(f"Failed to create version: {e}")

    def get_version(self, version: Optional[str] = None) -> Optional[DataModel]:
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

            # 验证数据是否有效
            if data is None:
                logger.error(f"Failed to load version {version}: data is None")
                return None

            # 获取元数据
            version_info = self.metadata['versions'].get(version)
            if not version_info:
                # 尝试从历史记录中获取
                version_info = next(
                    (v for v in self.history if v.get('version_id') == version), None)
                if not version_info:
                    return None

            metadata = version_info.get('metadata', {}) or {}

            # 兼容不同 DataModel 签名：优先关键字参数，其次回退多签
            data_model = None
            try:
                data_model = DataModel(data=data, metadata=metadata)  # 常见签名
            except TypeError:
                try:
                    data_model = DataModel(data, '1d', metadata)  # 兼容 data, frequency, metadata
                except TypeError:
                    try:
                        data_model = DataModel(data)  # 最小签名
                        if hasattr(data_model, 'metadata') and not getattr(data_model, 'metadata'):
                            setattr(data_model, 'metadata', dict(metadata))
                    except Exception:
                        data_model = None

            if data_model is None:
                logger.error(f"Failed to construct DataModel for version {version}")
                return None

            # 补丁：还原用户元数据，保证 get_metadata(user_only=True) 一致
            try:
                setattr(data_model, '_user_metadata', dict(metadata))
                setattr(data_model, '_metadata', dict(metadata))
            except Exception:
                pass

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

    def rollback_to_version(self, version_id: str) -> Optional[DataModel]:
        """
        回滚到指定版本

        Args:
            version_id: 要回滚到的版本ID

        Returns:
            DataModel: 回滚后的版本数据模型，如果版本不存在则返回None
        """
        if version_id not in self.metadata['versions']:
            logger.error(f"Version {version_id} does not exist")
            return None

        # 加载目标版本
        target_version = self.get_version(version_id)
        if target_version:
            # 创建新版本指向回滚的版本
            new_version_id = self.create_version(
                target_version,
                description=f"Rollback to version {version_id}",
                tags=['rollback']
            )

            # 更新当前版本
            self.current_version = new_version_id
            self.metadata['latest_version'] = new_version_id
            self._save_metadata(self.metadata)

            logger.info(f"Rolled back to version {version_id}, new version: {new_version_id}")
            return target_version

        return None

    def export_version(self, version_id: str, export_path: Union[str, Path]) -> bool:
        """
        导出版本到外部文件

        Args:
            version_id: 版本ID
            export_path: 导出路径

        Returns:
            bool: 导出是否成功
        """
        export_path = Path(export_path)
        version_file = self.version_dir / f"{version_id}.parquet"

        if not version_file.exists():
            logger.error(f"Version file {version_file} does not exist")
            return False

        try:
            # 复制版本文件到导出路径
            shutil.copy2(version_file, export_path)
            logger.info(f"Exported version {version_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export version {version_id}: {e}")
            return False

    def import_version(self, import_path: Union[str, Path]) -> Optional[str]:
        """
        从外部文件导入版本

        Args:
            import_path: 导入路径

        Returns:
            str: 新导入版本的ID，如果导入失败则返回None
        """
        import_path = Path(import_path)

        if not import_path.exists():
            logger.error(f"Import file {import_path} does not exist")
            return None

        try:
            # 读取导入的数据
            data = pd.read_parquet(import_path)

            # 验证数据是否有效
            if data is None:
                logger.error(f"Failed to import version: data is None from {import_path}")
                return None

            metadata = {
                'source': 'imported',
                'imported_from': str(import_path),
                'imported_at': datetime.now().isoformat()
            }

            # 创建新数据模型（优先使用关键字参数以兼容不同实现）
            try:
                data_model = DataModel(data=data, metadata=metadata)
            except TypeError:
                data_model = DataModel(data, '1d', metadata)

            # 创建新版本
            version_id = self.create_version(
                data_model,
                description=f"Imported from {import_path}",
                tags=['imported']
            )

            logger.info(f"Imported version {version_id} from {import_path}")
            return version_id
        except Exception as e:
            logger.error(f"Failed to import version: {e}")
            return None

    def update_metadata(self, version_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        更新版本元数据

        Args:
            version_id: 版本ID
            new_metadata: 新的元数据

        Returns:
            bool: 更新是否成功
        """
        if version_id not in self.metadata['versions']:
            logger.error(f"Version {version_id} does not exist")
            return False

        try:
            # 更新元数据
            self.metadata['versions'][version_id]['metadata'].update(new_metadata)
            self._save_metadata(self.metadata)

            # 更新历史记录
            for entry in self.history:
                if entry.get('version_id') == version_id:
                    entry['metadata'].update(new_metadata)
                    break
            self._save_history()

            logger.info(f"Updated metadata for version {version_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for version {version_id}: {e}")
            return False

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

        # 验证数据是否有效
        # 容错：若任一版本数据为 None，则以空DataFrame代替，保证比较返回结构化结果而非抛异常
        if model1.data is None:
            model1.data = pd.DataFrame()
        if model2.data is None:
            model2.data = pd.DataFrame()

        try:
            # 比较元数据
            metadata1 = model1.get_metadata(user_only=True)
            metadata2 = model2.get_metadata(user_only=True)

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
                s1 = model1.data[col]
                s2 = model2.data[col]
                # 对齐索引后再比较，避免ValueError
                try:
                    s1_aligned, s2_aligned = s1.align(s2, join='outer')
                    # 用fillna补齐缺失值，保证比较不抛异常
                    s1_aligned = s1_aligned.fillna(value=pd.NA)
                    s2_aligned = s2_aligned.fillna(value=pd.NA)
                    if not s1_aligned.equals(s2_aligned):
                        value_diff[col] = {
                            'changed_rows': (s1_aligned != s2_aligned).sum(),
                            'null_diff': s2_aligned.isnull().sum() - s1_aligned.isnull().sum()
                        }
                except Exception as e:
                    value_diff[col] = {'error': f'Failed to compare column {col}: {e}'}

            data_diff['value_diff'] = value_diff

            return {
                'metadata_diff': metadata_diff,
                'data_diff': data_diff
            }
        except Exception as e:
            logger.error(f"Failed to compare versions {version1} and {version2}: {e}")
            raise DataVersionError(f"Failed to compare versions: {e}")
