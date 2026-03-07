
"""配置文件存储相关类"""

from typing import Dict, Any
from dataclasses import dataclass, field
from .configscope import ConfigScope


@dataclass
class ConfigItem:
    """配置项"""
    key: str
    value: Any
    scope: ConfigScope
    timestamp: float
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)




