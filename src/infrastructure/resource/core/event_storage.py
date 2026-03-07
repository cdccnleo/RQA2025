
import threading

from .shared_interfaces import ILogger, StandardLogger
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
"""
事件存储器

职责：存储和管理事件历史记录
"""


class EventStorage:
    """
    事件存储器

    职责：存储和管理事件的历史记录
    """

    def __init__(self, max_events: int = 1000, retention_hours: int = 24,
                 logger: Optional[ILogger] = None):
        self.max_events = max_events
        self.retention_period = timedelta(hours=retention_hours)
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

        self._events = deque(maxlen=max_events)
        self._lock = threading.RLock()

    def store_event(self, event_type: str, event_data: Any, timestamp: Optional[datetime] = None) -> None:
        """存储事件"""
        if timestamp is None:
            timestamp = datetime.now()

        event_record = {
            'type': event_type,
            'data': event_data,
            'timestamp': timestamp,
            'id': f"{event_type}_{timestamp.timestamp()}"
        }

        with self._lock:
            self._events.append(event_record)
            self.logger.log_debug(f"已存储事件: {event_type}")

    def get_events(self, event_type: Optional[str] = None,
                   since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取事件记录"""
        with self._lock:
            events = list(self._events)

            # 过滤事件类型
            if event_type:
                events = [e for e in events if e['type'] == event_type]

            # 过滤时间
            if since:
                events = [e for e in events if e['timestamp'] >= since]

            # 清理过期事件
            self._cleanup_expired_events()

            # 限制数量
            if limit:
                events = events[-limit:]

            return events

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        """获取事件数量"""
        with self._lock:
            if event_type:
                return sum(1 for e in self._events if e['type'] == event_type)
            return len(self._events)

    def clear_events(self, event_type: Optional[str] = None) -> None:
        """清除事件记录"""
        with self._lock:
            if event_type:
                self._events = deque(
                    (e for e in self._events if e['type'] != event_type),
                    maxlen=self.max_events
                )
                self.logger.log_info(f"已清除事件: {event_type}")
            else:
                self._events.clear()
                self.logger.log_info("已清除所有事件记录")

    def _cleanup_expired_events(self) -> None:
        """清理过期事件"""
        cutoff_time = datetime.now() - self.retention_period
        expired_count = 0

        while self._events and self._events[0]['timestamp'] < cutoff_time:
            self._events.popleft()
            expired_count += 1

        if expired_count > 0:
            self.logger.log_debug(f"已清理过期事件: {expired_count}个")

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            now = datetime.now()
            oldest_timestamp = self._events[0]['timestamp'] if self._events else now

            return {
                'total_events': len(self._events),
                'max_capacity': self.max_events,
                'retention_hours': self.retention_period.total_seconds() / 3600,
                'oldest_event_age_hours': (now - oldest_timestamp).total_seconds() / 3600,
                'utilization_percent': (len(self._events) / self.max_events) * 100
            }
