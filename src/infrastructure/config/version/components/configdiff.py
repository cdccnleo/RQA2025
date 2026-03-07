"""
configdiff 模块

提供 configdiff 相关功能和接口。
"""

from typing import Dict, Any, List
from dataclasses import dataclass

"""版本管理相关类"""


@dataclass
class ConfigDiff:
    """配置差异信息"""
    version_from: str
    version_to: str
    added_keys: List[str]
    removed_keys: List[str]
    modified_keys: Dict[str, Dict[str, Any]]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'version_from': self.version_from,
            'version_to': self.version_to,
            'added_keys': self.added_keys,
            'removed_keys': self.removed_keys,
            'modified_keys': self.modified_keys,
            'timestamp': self.timestamp
        }




