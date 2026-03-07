
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

from .configdiff import ConfigDiff
from .configversion import ConfigVersion

"""版本管理相关类"""
logger = logging.getLogger(__name__)


class ConfigVersionManager:
    """配置版本管理器"""

    def __init__(self, storage_path: str = "config/versions",
                 max_versions: int = 100,
                 auto_backup: bool = True):
        """
        初始化配置版本管理器

        Args:
            storage_path: 版本存储路径
            max_versions: 最大版本数量
            auto_backup: 是否自动备份
        """
        self.storage_path = Path(storage_path)
        self.max_versions = max_versions
        self.auto_backup = auto_backup

        # 版本存储
        self._versions: Dict[str, 'ConfigVersion'] = {}
        self._version_history: List[str] = []

        # 统计信息
        self.stats: Dict[str, Any] = {
            'total_versions': 0,
            'total_rollbacks': 0,
            'total_restores': 0,
            'storage_size': 0
        }

        # 初始化存储
        self._initialize_storage()

    def _initialize_storage(self):
        """初始化版本存储"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 加载现有版本
        self._load_versions()

    def _load_versions(self):
        """加载现有版本"""
        try:
            # 加载版本索引
            index_file = self.storage_path / "index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)

                self._version_history = index_data.get('history', [])
                self.stats.update(index_data.get('stats', {}))

                # 加载版本数据
                for version_id in self._version_history:
                    version_file = self.storage_path / f"{version_id}.json"
                    if version_file.exists():
                        with open(version_file, 'r', encoding='utf-8') as f:
                            version_data = json.load(f)

                        config_version = ConfigVersion(
                            version_id=version_data['version_id'],
                            timestamp=version_data['timestamp'],
                            config_data=version_data['config_data'],
                            checksum=version_data['checksum'],
                            author=version_data.get('author', 'system'),
                            description=version_data.get('description', ''),
                            tags=version_data.get('tags', []),
                            metadata=version_data.get('metadata', {})
                        )

                        self._versions[version_id] = config_version

        except Exception as e:
            logger.error(f"Failed to load versions: {e}")

    def _save_version_index(self):
        """保存版本索引"""
        try:
            index_data = {
                'history': self._version_history,
                'stats': self.stats,
                'last_updated': time.time()
            }

            index_file = self.storage_path / "index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save version index: {e}")

    def create_version(self, config_data: Dict[str, Any],
                       author: str = "system",
                       description: str = "",
                       tags: Optional[List[str]] = None) -> str:
        """
        创建新版本

        Args:
            config_data: 配置数据
            author: 作者
            description: 描述
            tags: 标签列表

        Returns:
            版本ID
        """
        # 生成版本ID
        timestamp = time.time()
        config_str = json.dumps(config_data, sort_keys=True)
        checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        version_id = f"v{int(timestamp)}_{checksum}"

        # 创建版本对象
        version = ConfigVersion(
            version_id=version_id,
            timestamp=timestamp,
            config_data=config_data.copy(),
            checksum=checksum,
            author=author,
            description=description,
            tags=tags or []
        )

        # 保存版本
        self._versions[version_id] = version
        self._version_history.append(version_id)

        # 限制版本数量
        if len(self._version_history) > self.max_versions:
            self._cleanup_old_versions()

        # 保存到文件
        self._save_version(version)
        self._save_version_index()

        self.stats['total_versions'] += 1

        logger.info(f"Created config version: {version_id}")
        return version_id

    def _save_version(self, version: ConfigVersion):
        """保存版本到文件"""
        try:
            version_file = self.storage_path / f"{version.version_id}.json"

            version_data = {
                'version_id': version.version_id,
                'timestamp': version.timestamp,
                'config_data': version.config_data,
                'checksum': version.checksum,
                'author': version.author,
                'description': version.description,
                'tags': version.tags,
                'metadata': version.metadata
            }

            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save version {version.version_id}: {e}")

    def _cleanup_old_versions(self):
        """清理旧版本"""
        while len(self._version_history) > self.max_versions:
            old_version_id = self._version_history.pop(0)

            # 删除版本文件
            version_file = self.storage_path / f"{old_version_id}.json"
            if version_file.exists():
                version_file.unlink()

            # 从内存中移除
            if old_version_id in self._versions:
                del self._versions[old_version_id]

        logger.info(f"Cleaned up old versions, kept {len(self._version_history)} versions")

    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """
        获取指定版本

        Args:
            version_id: 版本ID

        Returns:
            版本对象，如果不存在返回None
        """
        return self._versions.get(version_id)

    def get_latest_version(self) -> Optional[ConfigVersion]:
        """获取最新版本"""
        if self._version_history:
            latest_id = self._version_history[-1]
            return self._versions.get(latest_id)
        return None

    def list_versions(self, limit: Optional[int] = None,
                      author: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[ConfigVersion]:
        """
        列出版本

        Args:
            limit: 限制数量
            author: 作者过滤
            tags: 标签过滤

        Returns:
            版本列表
        """
        versions = []

        for version_id in reversed(self._version_history):  # 最新的在前
            version = self._versions.get(version_id)
            if not version:
                continue

            # 应用过滤器
            if author and version.author != author:
                continue

            if tags:
                if not any(tag in version.tags for tag in tags):
                    continue

            versions.append(version)

            if limit and len(versions) >= limit:
                break

        return versions

    def rollback_to_version(self, version_id: str) -> bool:
        """
        回滚到指定版本 - P0级别增强实现

        Args:
            version_id: 目标版本ID

        Returns:
            是否回滚成功
        """
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return False

        try:
            # 1. 验证版本数据完整性
            if not self._validate_version(version):
                logger.error(f"Version {version_id} validation failed")
                return False

            # 2. 创建备份点（当前状态）
            current_config = self._get_current_config()
            if current_config:
                backup_description = f"Pre-rollback backup to {version_id}"
                backup_version_id = self.create_version(
                    current_config,
                    author="system",
                    description=backup_description,
                    tags=["backup", "pre-rollback"]
                )
                logger.info(f"Created backup version: {backup_version_id}")

            # 3. 执行回滚操作
            rollback_result = self._execute_rollback(version)
            if not rollback_result:
                logger.error(f"Failed to execute rollback to {version_id}")
                return False

            # 4. 创建回滚记录
            rollback_description = f"Rollback to version {version_id}"
            new_version_id = self.create_version(
                version.config_data,
                author="system",
                description=rollback_description,
                tags=["rollback", f"from-{version_id}"]
            )

            # 5. 更新统计信息
            self.stats['total_rollbacks'] += 1
            self.stats['last_rollback'] = {
                'timestamp': time.time(),
                'target_version': version_id,
                'new_version': new_version_id
            }

            logger.info(
                f"Successfully rolled back to version {version_id} (new version: {new_version_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False

    def _validate_version(self, version: ConfigVersion) -> bool:
        """验证版本数据完整性和有效性"""
        try:
            # 1. 检查数据类型
            if not isinstance(version.config_data, dict):
                logger.error("Version config data is not a dictionary")
                return False

            # 2. 验证校验和
            config_str = json.dumps(version.config_data, sort_keys=True)
            calculated_checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]

            if calculated_checksum != version.checksum:
                logger.error(
                    f"Version checksum mismatch: expected {version.checksum}, got {calculated_checksum}")
                return False

            # 3. 检查配置的基本结构
            if not self._validate_config_structure(version.config_data):
                logger.error("Version config structure validation failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Version validation error: {e}")
            return False

    def _validate_config_structure(self, config_data: Dict[str, Any]) -> bool:
        """验证配置结构的基本有效性"""
        try:
            # 检查配置的深度（防止过度嵌套）
            max_depth = 10
            if self._get_dict_depth(config_data) > max_depth:
                logger.error(f"Config structure too deep (max: {max_depth})")
                return False

            # 检查配置大小（防止过大的配置）
            config_str = json.dumps(config_data)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(config_str.encode()) > max_size:
                logger.error(f"Config too large (max: {max_size} bytes)")
                return False

            # 检查关键配置项（可根据需要自定义）
            required_keys = []
            for key in required_keys:
                if key not in config_data:
                    logger.warning(f"Required config key '{key}' missing")

            return True

        except Exception as e:
            logger.error(f"Config structure validation error: {e}")
            return False

    def _get_dict_depth(self, d: Dict[str, Any], depth: int = 0) -> int:
        """获取字典的嵌套深度"""
        if not isinstance(d, dict):
            return depth
        
        if not d:  # 空字典
            return depth
        
        max_child_depth = depth + 1  # 至少是当前深度+1
        for v in d.values():
            if isinstance(v, dict):
                child_depth = self._get_dict_depth(v, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth

    def _get_current_config(self) -> Optional[Dict[str, Any]]:
        """获取当前配置状态（由子类实现）"""
        # 这里需要与具体的配置管理器集成
        # 暂时返回None，子类可以重写这个方法
        return None

    def _execute_rollback(self, version: ConfigVersion) -> bool:
        """执行回滚操作（由子类实现）"""
        # 这里需要与具体的配置管理器集成
        # 暂时返回True，子类可以重写这个方法
        return True

    def restore_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        恢复指定版本的配置

        Args:
            version_id: 版本ID

        Returns:
            配置数据，如果版本不存在返回None
        """
        version = self.get_version(version_id)
        if version:
            self.stats['total_restores'] += 1
            return version.config_data.copy()
        return None

    def compare_versions(self, version_id1: str, version_id2: str) -> Optional[ConfigDiff]:
        """
        比较两个版本的差异

        Args:
            version_id1: 版本1 ID
            version_id2: 版本2 ID

        Returns:
            版本差异对象
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)

        if not version1 or not version2:
            return None

        # 计算差异
        diff = self._calculate_diff(version1.config_data, version2.config_data)

        config_diff = ConfigDiff(
            version_from=version_id1,
            version_to=version_id2,
            added_keys=diff['added'],
            removed_keys=diff['removed'],
            modified_keys=diff['modified'],
            timestamp=time.time()
        )
        return config_diff

    def _calculate_diff(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算配置差异

        Args:
            config1: 配置1
            config2: 配置2

        Returns:
            差异信息
        """
        keys1 = set(self._flatten_keys(config1))
        keys2 = set(self._flatten_keys(config2))

        added = keys2 - keys1
        removed = keys1 - keys2
        common = keys1 & keys2

        modified = {}
        for key in common:
            value1 = self._get_nested_value(config1, key)
            value2 = self._get_nested_value(config2, key)

            if value1 != value2:
                modified[key] = {
                    'old_value': value1,
                    'new_value': value2
                }

        return {
            'added': list(added),
            'removed': list(removed),
            'modified': modified
        }

    def _flatten_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """扁平化嵌套字典的键"""
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)

            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))

        return keys

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """获取嵌套字典的值"""
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    def delete_version(self, version_id: str) -> bool:
        """
        删除指定版本

        Args:
            version_id: 版本ID

        Returns:
            是否删除成功
        """
        if version_id not in self._versions:
            return False

        try:
            # 删除文件
            version_file = self.storage_path / f"{version_id}.json"
            if version_file.exists():
                version_file.unlink()

            # 从内存中移除
            del self._versions[version_id]
            self._version_history.remove(version_id)

            # 更新索引
            self._save_version_index()

            logger.info(f"Deleted version: {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    def cleanup_old_versions(self, keep_count: int = 10):
        """
        清理旧版本，保留指定数量的最新版本

        Args:
            keep_count: 保留的版本数量
        """
        while len(self._version_history) > keep_count:
            old_version_id = self._version_history.pop(0)
            self.delete_version(old_version_id)

        logger.info(f"Cleaned up old versions, kept {len(self._version_history)} versions")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats.update({
            'current_versions': len(self._versions),
            'storage_path': str(self.storage_path),
            'max_versions': self.max_versions
        })
        return stats

    def export_versions(self, file_path: str) -> bool:
        """
        导出所有版本到文件

        Args:
            file_path: 导出文件路径

        Returns:
            是否导出成功
        """
        try:
            export_data = {
                'versions': {},
                'history': self._version_history,
                'stats': self.stats,
                'export_time': time.time()
            }

            for version_id, version in self._versions.items():
                export_data['versions'][version_id] = version.to_dict()

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to export versions: {e}")
            return False

    def import_versions(self, file_path: str) -> bool:
        """
        从文件导入版本

        Args:
            file_path: 导入文件路径

        Returns:
            是否导入成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # 导入版本
            for version_id, version_data in import_data.get('versions', {}).items():
                config_version = ConfigVersion(
                    version_id=version_data['version_id'],
                    timestamp=version_data['timestamp'],
                    config_data=version_data['config_data'],
                    checksum=version_data['checksum'],
                    author=version_data.get('author', 'system'),
                    description=version_data.get('description', ''),
                    tags=version_data.get('tags', []),
                    metadata=version_data.get('metadata', {})
                )

                self._versions[version_id] = config_version

            # 导入历史
            self._version_history = import_data.get('history', [])
            self.stats.update(import_data.get('stats', {}))

            # 保存索引
            self._save_version_index()

            logger.info(f"Imported {len(self._versions)} versions from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import versions: {e}")
            return False




