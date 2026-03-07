"""
事件持久化模块
提供事件存储和检索功能
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class PersistenceMode(Enum):
    """持久化模式"""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EventPersistence:
    """事件持久化管理器"""

    def __init__(self, mode: PersistenceMode = PersistenceMode.MEMORY, config: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.config = config or {}
        self._events: Dict[str, Dict[str, Any]] = {}
        self._event_status: Dict[str, EventStatus] = {}

    def save_event(self, event: Any) -> bool:
        """保存事件"""
        try:
            event_id = getattr(event, 'event_id', None)
            if not event_id:
                return False

            event_data = {
                'event_id': event_id,
                'event_type': str(getattr(event, 'event_type', 'unknown')),
                'data': getattr(event, 'data', {}),
                'source': getattr(event, 'source', 'unknown'),
                'timestamp': getattr(event, 'timestamp', datetime.now().timestamp()),
                'correlation_id': getattr(event, 'correlation_id', None)
            }

            self._events[event_id] = event_data
            self._event_status[event_id] = EventStatus.PENDING

            return True
        except Exception:
            return False

    def update_event_status(self, event_id: str, status: EventStatus) -> bool:
        """更新事件状态"""
        if event_id in self._event_status:
            self._event_status[event_id] = status
            return True
        return False

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """获取事件"""
        return self._events.get(event_id)

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """按类型获取事件"""
        return [event for event in self._events.values()
                if event['event_type'] == event_type]

    def get_stats(self) -> Dict[str, Any]:
        """获取持久化统计"""
        return {
            'total_events': len(self._events),
            'events_by_status': {
                status.value: sum(1 for s in self._event_status.values() if s == status)
                for status in EventStatus
            }
        }

    def shutdown(self) -> None:
        """关闭持久化"""
        self._events.clear()
        self._event_status.clear()


# 持久化事件数据类
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class PersistedEvent:
    """持久化事件数据类"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    source: str = "unknown"
    timestamp: float = None
    correlation_id: Optional[str] = None
    status: EventStatus = EventStatus.PENDING
    
    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().timestamp()


# 导出PersistedEvent
__all__ = ['PersistenceMode', 'EventStatus', 'EventPersistence', 'PersistedEvent']