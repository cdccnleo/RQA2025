from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ConfigVersion:
    """配置版本信息"""
    version_id: str
    config: Dict[str, Any]
    timestamp: float
    author: str
    comment: str

class IVersionStorage(ABC):
    """版本存储接口"""

    @abstractmethod
    def save_version(self, env: str, config: Dict, version_id: str) -> bool:
        """保存配置版本"""
        pass

    @abstractmethod
    def get_version(self, env: str, version_id: str) -> Optional[Dict]:
        """获取配置版本"""
        pass

    @abstractmethod
    def delete_version(self, env: str, version_id: str) -> bool:
        """删除配置版本"""
        pass

    @abstractmethod
    def list_versions(self, env: str) -> Dict[str, str]:
        """列出所有版本"""
        pass

class IVersionManager(ABC):
    """版本管理接口"""
    
    @abstractmethod
    def add_version(self, env: str, config: Dict[str, Any], author: str = "system", comment: str = "") -> str:
        """添加新版本"""
        pass
    
    @abstractmethod
    def get_versions(self, env: str, limit: int = 10) -> List[ConfigVersion]:
        """获取环境的所有版本"""
        pass
    
    @abstractmethod
    def get_version(self, env: str, version_id: str) -> Optional[ConfigVersion]:
        """获取特定版本"""
        pass
    
    @abstractmethod
    def rollback(self, env: str, version_id: str) -> bool:
        """回滚到指定版本"""
        pass
    
    @abstractmethod
    def diff_versions(self, env: str, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本的差异"""
        pass
