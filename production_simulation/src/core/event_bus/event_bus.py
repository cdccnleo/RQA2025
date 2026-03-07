"""
事件总线模块（别名模块）
提供向后兼容的导入路径

实际实现在 core.py 中
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# 从types模块导入EventPriority
from .types import EventPriority

# 定义事件类型枚举
class EventType(Enum):
    """事件类型"""
    SYSTEM = "system"
    BUSINESS = "business"
    DATA = "data"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CUSTOM = "custom"

# 定义事件数据类
@dataclass
class Event:
    """事件数据类"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    source: str = "unknown"
    timestamp: Optional[float] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

try:
    from .core import EventBus, EventBusConfig
except ImportError:
    # 提供基础实现
    class EventBus:
        pass
    
    class EventBusConfig:
        pass

__all__ = ['EventBus', 'EventBusConfig', 'EventType', 'Event', 'EventPriority']

