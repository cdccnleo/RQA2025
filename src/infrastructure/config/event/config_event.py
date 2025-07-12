"""配置变更事件定义"""
import time
from dataclasses import dataclass
from typing import Any, Optional, Callable, Dict, List
from abc import ABC, abstractmethod

class EventFilter(ABC):
    """事件过滤器抽象基类"""
    @abstractmethod
    def filter(self, event: 'ConfigEvent') -> bool:
        """过滤事件"""
        pass

import warnings

class ConfigEventBus:
    """配置事件总线(主实现)
    
    功能:
    1. 事件发布订阅
    2. 多级过滤
    3. 死信队列处理
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._filters: List[EventFilter] = []
        self._dead_letters: List[Dict] = []

    def subscribe(self, event_type: str, callback: Callable) -> str:
        """订阅事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        return f"sub-{len(self._subscribers[event_type])}"

    def publish(self, event_type: str, event_data: Dict) -> bool:
        """发布事件
        Args:
            event_type: 事件类型字符串
            event_data: 事件数据字典
        Returns:
            bool: 发布是否成功
        """
        try:
            # 创建ConfigEvent对象以保持向后兼容
            event = ConfigEvent(
                event_type=event_type,
                key=event_data.get('key', ''),
                old_value=event_data.get('old_value'),
                new_value=event_data.get('new_value', event_data)
            )

            if all(f.filter(event) for f in self._filters):
                for callback in self._subscribers.get(event_type, []):
                    callback(event)
                return True
            return False
        except Exception as e:
            self._dead_letters.append({
                'event_type': event_type,
                'event_data': event_data,
                'error': str(e),
                'timestamp': time.time()
            })
            return False

    def add_filter(self, filter: EventFilter):
        """添加事件过滤器"""
        self._filters.append(filter)

    def get_dead_letters(self) -> List[Dict]:
        """获取死信队列"""
        return self._dead_letters

    def clear_dead_letters(self):
        """清空死信队列"""
        self._dead_letters.clear()

@dataclass
class ConfigEvent:
    """配置变更事件

    属性:
        event_type: 事件类型 (config_updated/config_loaded)
        key: 配置键
        old_value: 旧值 (更新事件才有)
        new_value: 新值
        timestamp: 事件时间戳
    """
    __slots__ = ('event_type', 'key', 'old_value', 'new_value', 'timestamp')

    event_type: str
    key: str
    old_value: Optional[Any]
    new_value: Any

    def __post_init__(self):
        """初始化后设置时间戳"""
        import time
        self.timestamp = time.time()
