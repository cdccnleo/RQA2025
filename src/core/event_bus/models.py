"""
事件总线数据模型
包含事件和处理器的数据结构定义
"""

from typing import Dict, Callable, Any, Union, Optional, TYPE_CHECKING
from dataclasses import dataclass
import time

from ..foundation.base import generate_id
from .types import EventType, EventPriority

@dataclass
class Event:

    """事件数据类 - 增强版"""
    event_type: Union[EventType, str]
    data: Dict[str, Any] = None
    timestamp: float = None
    source: str = "system"
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    event_id: Optional[str] = None
    correlation_id: Optional[str] = None  # 用于关联相关事件

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = time.time()
        if self.event_id is None:
            self.event_id = generate_id(f"event_{self.event_type}")
        if self.data is None:
            self.data = {}
        if self.correlation_id is None:
            self.correlation_id = self.event_id


@dataclass
class EventHandler:

    """事件处理器 - 优化版"""
    handler: Callable
    priority: EventPriority = EventPriority.NORMAL
    async_handler: bool = False
    retry_on_failure: bool = True
    max_retries: int = 3
    batch_size: int = 1  # 批处理大小
    timeout: float = 30.0  # 超时时间


# HandlerExecutionContext从event_bus.__init__或core导入
# 这里不直接导出，避免循环导入
__all__ = ['Event', 'EventHandler']