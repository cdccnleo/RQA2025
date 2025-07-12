import json
import os
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

from src.infrastructure.config.interfaces.version_storage import IVersionManager, IVersionStorage

logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """版本状态枚举"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

@dataclass
class ConfigVersion:
    """配置版本信息"""
    version_id: str
    config_hash: str
    created_at: datetime
    created_by: str
    description: str
    status: VersionStatus
    parent_version: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """从字典创建实例"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = VersionStatus(data['status'])
        return cls(**data)

class ConfigVersionManager:
    """配置版本管理器"""
    
    def __init__(self, storage: IVersionStorage):
        self.storage = storage
        self._lock = threading.Lock()
        self._cache: Dict[str, ConfigVersion] = {}
    
    def create_version(self, config: Dict[str, Any], created_by: str, 
                      description: str = "", tags: List[str] = None) -> ConfigVersion:
        """创建新版本"""
        with self._lock:
            # 生成配置哈希
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # 检查是否已存在相同配置
            existing = self.storage.find_by_hash(config_hash)
            if existing:
                logger.warning(f"Configuration already exists with hash {config_hash}")
                return existing
            
            # 创建新版本
            version_id = str(uuid.uuid4())
            version = ConfigVersion(
                version_id=version_id,
                config_hash=config_hash,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                status=VersionStatus.DRAFT,
                tags=tags or []
            )
            
            # 保存到存储
            self.storage.save_version(version, config)
            self._cache[version_id] = version
            
            logger.info(f"Created new config version {version_id}")
            return version
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """获取指定版本"""
        # 先从缓存获取
        if version_id in self._cache:
            return self._cache[version_id]
        
        # 从存储获取
        version = self.storage.get_version(version_id)
        if version:
            self._cache[version_id] = version
        return version
    
    def list_versions(self, status: Optional[VersionStatus] = None, 
                     tags: Optional[List[str]] = None) -> List[ConfigVersion]:
        """列出版本"""
        versions = self.storage.list_versions()
        
        # 过滤状态
        if status:
            versions = [v for v in versions if v.status == status]
        
        # 过滤标签
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        return versions
    
    def publish_version(self, version_id: str) -> bool:
        """发布版本"""
        with self._lock:
            version = self.get_version(version_id)
            if not version:
                logger.error(f"Version {version_id} not found")
                return False
            
            if version.status != VersionStatus.DRAFT:
                logger.error(f"Version {version_id} is not in draft status")
                return False
            
            # 更新状态
            version.status = VersionStatus.PUBLISHED
            self.storage.update_version(version)
            self._cache[version_id] = version
            
            logger.info(f"Published version {version_id}")
            return True
    
    def archive_version(self, version_id: str) -> bool:
        """归档版本"""
        with self._lock:
            version = self.get_version(version_id)
            if not version:
                return False
            
            version.status = VersionStatus.ARCHIVED
            self.storage.update_version(version)
            self._cache[version_id] = version
            
            logger.info(f"Archived version {version_id}")
            return True
    
    def get_latest_published(self) -> Optional[ConfigVersion]:
        """获取最新发布的版本"""
        versions = self.list_versions(status=VersionStatus.PUBLISHED)
        if not versions:
            return None
        
        return max(versions, key=lambda v: v.created_at)
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本"""
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            return {"error": "One or both versions not found"}
        
        config1 = self.storage.get_config(version_id1)
        config2 = self.storage.get_config(version_id2)
        
        return self._compare_configs(config1, config2)
    
    def _compare_configs(self, config1: Dict[str, Any], 
                        config2: Dict[str, Any]) -> Dict[str, Any]:
        """比较配置内容"""
        all_keys = set(config1.keys()) | set(config2.keys())
        
        changes = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        for key in all_keys:
            if key not in config1:
                changes["added"].append(key)
            elif key not in config2:
                changes["removed"].append(key)
            elif config1[key] != config2[key]:
                changes["modified"].append(key)
        
        return changes

class ConfigVersionStorage(IVersionStorage):
    """配置版本存储实现"""
    
    def __init__(self, storage_dir: str = "./config_versions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    def save_version(self, version: ConfigVersion, config: Dict[str, Any]):
        """保存版本"""
        with self._lock:
            version_file = self.storage_dir / f"{version.version_id}.json"
            config_file = self.storage_dir / f"{version.version_id}_config.json"
            
            # 保存版本信息
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)
            
            # 保存配置内容
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """获取版本信息"""
        version_file = self.storage_dir / f"{version_id}.json"
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConfigVersion.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading version {version_id}: {e}")
            return None
    
    def update_version(self, version: ConfigVersion):
        """更新版本信息"""
        version_file = self.storage_dir / f"{version.version_id}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)
    
    def list_versions(self) -> List[ConfigVersion]:
        """列出所有版本"""
        versions = []
        for file_path in self.storage_dir.glob("*.json"):
            if not file_path.name.endswith("_config.json"):
                version_id = file_path.stem
                version = self.get_version(version_id)
                if version:
                    versions.append(version)
        return versions
    
    def get_config(self, version_id: str) -> Optional[Dict[str, Any]]:
        """获取版本配置内容"""
        config_file = self.storage_dir / f"{version_id}_config.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config for version {version_id}: {e}")
            return None
    
    def find_by_hash(self, config_hash: str) -> Optional[ConfigVersion]:
        """根据配置哈希查找版本"""
        for version in self.list_versions():
            if version.config_hash == config_hash:
                return version
        return None
    
    def delete_version(self, version_id: str) -> bool:
        """删除版本"""
        try:
            version_file = self.storage_dir / f"{version_id}.json"
            config_file = self.storage_dir / f"{version_id}_config.json"
            
            if version_file.exists():
                version_file.unlink()
            if config_file.exists():
                config_file.unlink()
            
            logger.info(f"Deleted version {version_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting version {version_id}: {e}")
            return False

# 创建默认存储实例
default_storage = ConfigVersionStorage()
default_manager = ConfigVersionManager(default_storage)

def get_version_manager() -> ConfigVersionManager:
    """获取版本管理器实例"""
    return default_manager