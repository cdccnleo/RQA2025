from typing import Dict, Optional
from abc import ABC, abstractmethod

class IVersionManager(ABC):
    """版本管理接口"""

    @abstractmethod
    def add_version(self, env: str, config: Dict) -> str:
        """添加新版本"""
        pass

    @abstractmethod
    def get_version(self, env: str, version: str) -> Optional[Dict]:
        """获取特定版本"""
        pass

    @abstractmethod
    def diff_versions(self, env: str, v1: str, v2: str) -> Dict:
        """比较版本差异"""
        pass

    @abstractmethod
    def rollback(self, env: str, version: str) -> bool:
        """回滚到指定版本"""
        pass
