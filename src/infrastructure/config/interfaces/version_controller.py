from abc import ABC, abstractmethod
from typing import Dict, Optional

class IVersionManager(ABC):
    """版本管理器接口
    
    功能:
    1. 版本添加和获取
    2. 版本差异比较
    3. 版本回滚支持
    """

    @abstractmethod
    def add_version(self, env: str, config: Dict) -> str:
        """添加配置版本

        Args:
            env: 环境名称
            config: 配置字典

        Returns:
            版本ID
        """
        pass

    @abstractmethod
    def get_version(self, env: str, version_id: str) -> Optional[Dict]:
        """获取指定版本配置

        Args:
            env: 环境名称
            version_id: 版本ID

        Returns:
            配置字典或None
        """
        pass

    @abstractmethod
    def diff_versions(self, env: str, v1: str, v2: str) -> Dict:
        """比较两个版本的差异

        Args:
            env: 环境名称
            v1: 版本1 ID
            v2: 版本2 ID

        Returns:
            差异字典
        """
        pass
