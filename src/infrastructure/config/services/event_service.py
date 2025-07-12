from typing import Dict, List, Callable, Optional
from ..error.exceptions import EventError, EventDeliveryError
from ..interfaces.event_system import IConfigEventSystem, IEventSubscriber
from ..interfaces.version_controller import IVersionManager

class ConfigEventBus(IConfigEventSystem):
    """配置事件总线实现"""
    
    def __init__(self, version_manager: IVersionManager):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._dead_letters: List[Dict] = []
        self._version_manager = version_manager

    def publish(self, event_type: str, payload: Dict) -> None:
        """发布配置变更事件"""
        try:
            # 添加版本上下文
            # 获取最新版本号
            latest_version = None
            if hasattr(self._version_manager, 'get_latest_version'):
                latest_version = self._version_manager.get_latest_version()
            elif hasattr(self._version_manager, '_versions') and self._version_manager._versions:
                latest_env = next(iter(self._version_manager._versions))
                if self._version_manager._versions[latest_env]:
                    latest_version = self._version_manager._versions[latest_env][-1]['id']
            
            context = {
                **payload,
                "version": latest_version or "unknown"
            }
            
            for handler in self._subscribers.get(event_type, []):
                try:
                    handler(context)
                except Exception as e:
                    self._dead_letters.append({
                        "event": event_type,
                        "payload": payload,
                        "error": str(e)
                    })
                    raise EventDeliveryError(
                        f"事件处理失败: {event_type}",
                        event_type=event_type
                    )
                    
        except Exception as e:
            raise EventError(f"事件发布失败: {str(e)}")

    def subscribe(self, 
                 event_type: str, 
                 handler: Callable[[Dict], None],
                 filter_func: Optional[Callable[[Dict], bool]] = None) -> str:
        """订阅配置事件
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            filter_func: 可选过滤器函数，返回True时才会调用handler
        Returns:
            订阅ID
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if filter_func:
            def wrapped_handler(event):
                if filter_func(event):
                    return handler(event)
            self._subscribers[event_type].append(wrapped_handler)
        else:
            self._subscribers[event_type].append(handler)
            
        return f"sub-{len(self._subscribers[event_type])}"

    def get_dead_letters(self) -> List[Dict]:
        """获取死信队列"""
        return self._dead_letters

    def clear_dead_letters(self) -> None:
        """清空死信队列"""
        self._dead_letters.clear()


class EventSubscriber(IEventSubscriber):
    """基础事件订阅者"""
    
    def __init__(self, event_bus: IConfigEventSystem):
        self._event_bus = event_bus
        
    def handle_event(self, event: Dict) -> bool:
        """默认事件处理（需被子类覆盖）"""
        raise NotImplementedError
