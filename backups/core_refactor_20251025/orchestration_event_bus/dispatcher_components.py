from datetime import datetime
from typing import Dict, Any, List
import logging
from typing import Dict, Any, List, Callable
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)


class EventDispatcher(ABC):

    """事件分发器抽象基类"""

    @abstractmethod
    def dispatch_event(self, event) -> bool:
        """分发事件"""

    @abstractmethod
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""

    @abstractmethod
    def unregister_handler(self, event_type: str, handler: Callable):
        """取消注册事件处理器"""

    @abstractmethod
    def get_dispatch_stats(self) -> Dict[str, Any]:
        """获取分发统计信息"""


class BasicEventDispatcher(EventDispatcher):

    """基础事件分发器实现"""

    def __init__(self):

        self._handlers = {}
        self._dispatched_count = 0

    def dispatch_event(self, event) -> bool:
        """分发事件"""
        try:
            event_type = event.get('type', 'unknown')
            if event_type in self._handlers:
                for handler in self._handlers[event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"事件处理失败: {e}")
            self._dispatched_count += 1
            logger.info(f"事件分发成功: {event_type}")
            return True
        except Exception as e:
            logger.error(f"事件分发失败: {e}")
            return False

    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"注册事件处理器: {event_type}")

    def unregister_handler(self, event_type: str, handler: Callable):
        """取消注册事件处理器"""
        if event_type in self._handlers:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.info(f"取消注册事件处理器: {event_type}")

    def get_dispatch_stats(self) -> Dict[str, Any]:
        """获取分发统计信息"""
        return {
            'dispatched_count': self._dispatched_count,
            'handler_count': sum(len(handlers) for handlers in self._handlers.values()),
            'event_types': list(self._handlers.keys())
        }


class ComponentFactory:

    """组件工厂"""

    def __init__(self):

        self._components = {}

    def create_component(self, component_type: str, config: Dict[str, Any]):
        """创建组件"""
        try:
            component = self._create_component_instance(component_type, config)
            if component and component.initialize(config):
                return component
            return None
        except Exception as e:
            logger.error(f"创建组件失败: {e}")
            return None

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """创建组件实例"""
        return None


# -*- coding: utf-8 -*-
# #!/usr/bin/env python3
"""
统一Dispatcher组件工厂

合并所有dispatcher_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:18:35
"""


class IDispatcherComponent(ABC):

    """Dispatcher组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_dispatcher_id(self) -> int:
        """获取dispatcher ID"""


class DispatcherComponent(IDispatcherComponent):

    """统一Dispatcher组件实现"""

    def __init__(self, dispatcher_id: int, component_type: str = "Dispatcher"):
        """初始化组件"""
        self.dispatcher_id = dispatcher_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{dispatcher_id}"
        self.creation_time = datetime.now()

    def get_dispatcher_id(self) -> int:
        """获取dispatcher ID"""
        return self.dispatcher_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "dispatcher_id": self.dispatcher_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_core_event_bus_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "dispatcher_id": self.dispatcher_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_dispatcher_processing"
            }
            return result
        except Exception as e:
            return {
                "dispatcher_id": self.dispatcher_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "dispatcher_id": self.dispatcher_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class DispatcherComponentFactory:

    """Dispatcher组件工厂"""

    # 支持的dispatcher ID列表
    SUPPORTED_DISPATCHER_IDS = [5]

    @staticmethod
    def create_component(dispatcher_id: int) -> DispatcherComponent:
        """创建指定ID的dispatcher组件"""
        if dispatcher_id not in DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS:
            raise ValueError(
                f"不支持的dispatcher ID: {dispatcher_id}。支持的ID: {DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS}")

        return DispatcherComponent(dispatcher_id, "Dispatcher")

    @staticmethod
    def get_available_dispatchers() -> List[int]:
        """获取所有可用的dispatcher ID"""
        return sorted(list(DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS))

    @staticmethod
    def create_all_dispatchers() -> Dict[int, DispatcherComponent]:
        """创建所有可用dispatcher"""
        return {
            dispatcher_id: DispatcherComponent(dispatcher_id, "Dispatcher")
            for dispatcher_id in DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "DispatcherComponentFactory",
            "version": "2.0.0",
            "total_dispatchers": len(DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS),
            "supported_ids": sorted(list(DispatcherComponentFactory.SUPPORTED_DISPATCHER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_dispatcher_dispatcher_component_5(): return DispatcherComponentFactory.create_component(5)


__all__ = [
    "IDispatcherComponent",
    "DispatcherComponent",
    "DispatcherComponentFactory",
    "create_dispatcher_dispatcher_component_5",
]
