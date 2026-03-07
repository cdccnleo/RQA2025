"""
configversion 模块

提供 configversion 相关功能和接口。
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

"""版本管理相关类"""


@dataclass
class ConfigVersion:
    """配置版本信息"""
    version_id: str
    timestamp: float
    config_data: Dict[str, Any]
    checksum: str
    author: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        """获取版本创建时间"""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp,
            'checksum': self.checksum,
            'author': self.author,
            'description': self.description,
            'tags': self.tags,
            'metadata': self.metadata,
            'config_size': len(json.dumps(self.config_data, sort_keys=True))
        }




