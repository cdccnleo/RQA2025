from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class IDiffService(ABC):
    """差异比较服务接口"""

    @abstractmethod
    def compare_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个字典的差异"""
        pass


class IVersionComparator(ABC):
    """版本比较器接口"""
    
    @abstractmethod
    def compare_versions(self, env: str, v1: str, v2: str) -> Dict[str, Any]:
        """比较两个版本的差异
        
        Args:
            env: 环境名称
            v1: 版本1标识符
            v2: 版本2标识符
            
        Returns:
            差异详情字典，包含变更字段和前后值
        """
        pass

    @abstractmethod
    def get_change_type(self, old_val: Any, new_val: Any) -> Optional[str]:
        """获取变更类型
        
        Args:
            old_val: 旧值
            new_val: 新值
            
        Returns:
            变更类型字符串(ADDED/MODIFIED/DELETED)或None
        """
        pass
