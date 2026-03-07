
# from ..core.unified_interface import IConfigEventBus, IConfigVersionManager as IVersionManager, IEventSubscriber
# 暂时注释掉不存在的接口导入

from ..config_exceptions import ConfigLoadError
import time
from typing import Dict, List, Callable, Optional, Any
"""
基础设施层 - 工具组件组件

event_service 模块

通用工具组件
提供工具组件相关的功能实现。
"""


class ConfigEventBus:

    """
event_service - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator
""" """配置事件总线实现"""

    def __init__(self, version_manager=None):

        self._subscribers: Dict[str, List[Callable]] = {}
        self._subscription_ids: Dict[str, Dict[str, Callable]] = {}  # event -> {id: handler}
        self._dead_letters: List[Dict] = []
        self._version_manager = version_manager
        self.event_history: List = []  # 事件历史记录
        self._max_history = 100  # 最大历史记录数

    @property
    def max_history_size(self):
        """获取最大历史记录数"""
        return self._max_history

    @max_history_size.setter
    def max_history_size(self, value: int):
        """设置最大历史记录数"""
        if value < 0:
            raise ValueError("max_history_size cannot be negative")
        self._max_history = value

    def publish(self, event_type, payload=None) -> None:
        """发布事件，支持事件对象或事件类型字符串"""

        # 处理事件对象
        if hasattr(event_type, 'event_type') and hasattr(event_type, 'data'):
            # 这是一个ConfigEvent对象
            event_obj = event_type
            event_type_str = event_obj.event_type
            payload = event_obj.data
        else:
            # 这是一个字符串事件类型
            event_type_str = event_type
            if payload is None:
                payload = {}

        print(f"事件发布: {event_type_str}, payload={payload}")
        try:
            # 添加版本上下文
            # 获取最新版本号
            latest_version = None
            if self._version_manager is not None and hasattr(self._version_manager, 'get_latest_version'):
                latest_version = self._version_manager.get_latest_version()
            elif self._version_manager is not None and hasattr(self._version_manager, '_versions') and self._version_manager._versions:
                latest_env = next(iter(self._version_manager._versions))
                if self._version_manager._versions[latest_env]:
                    latest_version = self._version_manager._versions[latest_env][-1]['id']

            # 处理None payload的情况
            if payload is None:
                payload = {}

            context = {
                **payload,
                "version": latest_version or "unknown"
            }

            handler_exceptions = []
            for handler in self._subscribers.get(event_type_str, []):
                try:
                    handler(context)
                except Exception as e:
                    self._dead_letters.append({
                        "event": event_type_str,
                        "payload": payload,
                        "error": str(e)
                    })
                    handler_exceptions.append(e)

            # 根据事件类型决定是否抛出异常
            # 对于字符串事件类型，应该抛出ConfigLoadError异常
            # 对于ConfigEvent对象，不应该抛出异常
            if not hasattr(event_type, 'event_type') and handler_exceptions:
                # 字符串事件类型，抛出第一个异常
                raise ConfigLoadError(f"事件处理失败: {str(handler_exceptions[0])}", details={
                    'event_type': event_type_str,
                    'payload': payload,
                    'original_error': str(handler_exceptions[0])
                })

            # 记录事件历史 (如果是事件对象)
            if hasattr(event_type, 'event_type'):
                self.event_history.append(event_type)
                if len(self.event_history) > self._max_history:
                    self.event_history.pop(0)

        except Exception as e:
            if isinstance(e, ConfigLoadError):
                raise
            else:
                # 只有在发布事件本身失败时才抛出ConfigLoadError
                raise ConfigLoadError(f"事件发布失败: {str(e)}", details={'original_error': str(e)})

    def subscribe(self, event_type: str, handler: Callable[[Dict], None],

                  filter_func: Optional[Callable[[Dict], bool]] = None):
        """订阅配置事件"
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            filter_func: 可选过滤器函数，返回True时才会调用handler
        Returns:
            订阅ID
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
            self._subscription_ids[event_type] = {}

        if filter_func:

            def wrapped_handler(event):

                if filter_func(event):
                    return handler(event)
            self._subscribers[event_type].append(wrapped_handler)
        else:
            self._subscribers[event_type].append(handler)

        subscription_id = f"sub-{len(self._subscribers[event_type])}"
        self._subscription_ids[event_type][subscription_id] = handler
        return subscription_id

    def unsubscribe(self, event: str, subscription_id_or_handler) -> bool:
        """取消订阅"""
        if not isinstance(subscription_id_or_handler, str):
            # 如果传递的是handler对象，直接从订阅者列表中移除
            handler = subscription_id_or_handler
            if event in self._subscribers:
                try:
                    self._subscribers[event].remove(handler)
                    # 同时清理subscription_ids中的对应项
                    if event in self._subscription_ids:
                        ids_to_remove = []
                        for sub_id, h in self._subscription_ids[event].items():
                            if h == handler:
                                ids_to_remove.append(sub_id)
                        for sub_id in ids_to_remove:
                            del self._subscription_ids[event][sub_id]
                    return True
                except ValueError:
                    pass
            return False
        else:
            # 如果传递的是subscription_id字符串，使用原有逻辑
            subscription_id = subscription_id_or_handler
            if event in self._subscription_ids and subscription_id in self._subscription_ids[event]:
                handler = self._subscription_ids[event][subscription_id]
                if event in self._subscribers:
                    try:
                        self._subscribers[event].remove(handler)
                        del self._subscription_ids[event][subscription_id]
                        return True
                    except ValueError:
                        pass
            return False

    def get_subscribers(self, event: str) -> Dict[str, Callable]:
        """获取事件订阅者"""
        return self._subscription_ids.get(event, {})

    def notify_config_updated(self, key: str, old_value: Any, new_value: Any) -> None:
        """通知配置更新"""
        self.publish("config_updated", {
            "key": key,
            "old_value": old_value,
            "new_value": new_value
        })

    def notify_config_error(self, error: str, details: Dict[str, Any]) -> None:
        """通知配置错误"""
        self.publish("config_error", {
            "error": error,
            "details": details
        })

    def notify_config_loaded(self, source: str, config: Dict[str, Any]) -> None:
        """通知配置加载"""
        self.publish("config_loaded", {
            "source": source,
            "config": config
        })

    def get_dead_letters(self) -> List[Dict]:
        """获取死信队列"""
        return self._dead_letters

    def clear_dead_letters(self) -> None:
        """清空死信队列"""
        self._dead_letters.clear()

    def get_recent_events(self, event_type: Optional[str] = None, limit: Optional[int] = None) -> List:
        """获取最近的事件"""
        if event_type is not None:
            # 如果提供了event_type，使用过滤方法
            return self.get_recent_events_filtered(event_type, limit)
        else:
            # 否则返回所有事件，可能限制数量
            if limit is None or limit >= len(self.event_history):
                return self.event_history.copy()
            else:
                return self.event_history[-limit:].copy()

    def get_recent_events_filtered(self, event_type: str, limit: Optional[int] = None) -> List:
        """获取指定类型的最近事件"""
        filtered_events = [event for event in self.event_history if getattr(
            event, 'event_type', None) == event_type]
        if limit is None or limit >= len(filtered_events):
            return filtered_events.copy()
        else:
            return filtered_events[-limit:].copy()

    def clear_history(self) -> None:
        """清空事件历史"""
        self.event_history.clear()

    def emit_config_changed(self, key: str, old_value: Any, new_value: Any) -> None:
        """发送配置变更事件（IConfigEventService接口实现）"""
        self.publish("config_changed", {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": time.time()
        })

    def emit_config_loaded(self, source: str) -> None:
        """发送配置加载事件（IConfigEventService接口实现）"""
        self.publish("config_loaded", {
            "source": source,
            "timestamp": time.time()
        })


class EventSubscriber:

    """基础事件订阅者"""

    def __init__(self, event_bus):

        self._event_bus = event_bus

    def handle_event(self, event: Dict) -> bool:
        """默认事件处理（需被子类覆盖）"""
        raise NotImplementedError


# EventService别名，保持向后兼容性
EventService = ConfigEventBus


