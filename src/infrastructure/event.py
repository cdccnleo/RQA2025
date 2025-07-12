"""
轻量级事件系统实现
提供配置变更通知的核心机制
"""
import os
import threading
import uuid
from typing import Callable, Dict, List, Any, Optional

# 标准事件类型定义
class ConfigEvents:
    """配置管理相关事件类型"""
    CONFIG_CHANGED = "config_changed"
    CONFIG_RELOADED = "config_reloaded"
    CONFIG_ERROR = "config_error"
    
class EventSystem:
    """线程安全的事件发布-订阅系统"""

    _instance = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        # 测试环境专用
        self._events: Dict[str, List[Any]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> str:
        """
        订阅指定类型事件并返回订阅ID
        :param event_type: 事件类型标识符
        :param callback: 事件处理函数，接受event_data参数
        :return: 唯一订阅ID字符串
        """
        import uuid
        sub_id = str(uuid.uuid4())
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = {}
            self._subscribers[event_type][sub_id] = callback
        return sub_id

    def unsubscribe(self, event_type: str, callback_or_id: Any) -> None:
        """
        取消事件订阅
        :param event_type: 事件类型标识符
        :param callback_or_id: 可以是回调函数或订阅ID
        """
        with self._lock:
            if event_type in self._subscribers:
                if isinstance(callback_or_id, str):  # 按订阅ID取消
                    self._subscribers[event_type].pop(callback_or_id, None)
                else:  # 按回调函数取消
                    to_remove = [sub_id for sub_id, cb in self._subscribers[event_type].items() 
                                if cb == callback_or_id]
                    for sub_id in to_remove:
                        self._subscribers[event_type].pop(sub_id, None)

    def publish(self, event_type: str, event_data: Any = None) -> None:
        """
        发布事件
        :param event_type: 事件类型标识符
        :param event_data: 事件相关数据
        """
        with self._lock:
            # 测试环境记录事件
            if os.environ.get('TESTING') == 'true':
                if event_type not in self._events:
                    self._events[event_type] = []
                self._events[event_type].append(event_data)
                
            if event_type in self._subscribers:
                # 复制列表避免回调中修改订阅者
                for callback in list(self._subscribers[event_type]):
                    try:
                        callback(event_data)
                    except Exception as e:
                        print(f"Event callback error: {e}")

    @classmethod
    def get_default(cls) -> 'EventSystem':
        """获取默认事件系统实例(单例)"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_events(self, event_type: str) -> List[Any]:
        """
        获取指定类型的事件(仅用于测试)
        :param event_type: 要查询的事件类型
        :return: 事件列表的副本
        """
        with self._lock:
            return list(self._events.get(event_type, []))

    def clear_events(self, event_type: Optional[str] = None) -> None:
        """
        清除记录的事件(仅用于测试)
        :param event_type: 要清除的事件类型，None表示清除所有
        """
        with self._lock:
            if event_type is None:
                self._events.clear()
            elif event_type in self._events:
                self._events[event_type].clear()


