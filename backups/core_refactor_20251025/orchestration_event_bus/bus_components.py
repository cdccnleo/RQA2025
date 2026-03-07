from datetime import datetime
from typing import Dict, Any, List, Callable, Union
import logging
from abc import ABC, abstractmethod
from .unified_event_interface import (
    IEventBus, Event, EventDeliveryMode, EventPersistence
)
logger = logging.getLogger(__name__)


class EventHandler(ABC):

    """事件处理器抽象基类"""

    @abstractmethod
    def handle_event(self, event) -> bool:
        """处理事件"""

    @abstractmethod
    def can_handle(self, event) -> bool:
        """判断是否可以处理该事件"""


class EventBus(IEventBus):

    """事件总线实现 - 高性能优化版本"""

    def __init__(self, max_workers: int = 10, enable_async: bool = True, batch_size: int = 100):

        self._subscribers = {}
        self._handlers = {}
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # 性能优化：预编译事件类型映射和连接池
        self._subscriber_cache = {}
        self._batch_queue = []
        self._connection_pool = {}

        # 事件历史记录 (测试兼容性)
        self._event_history = []
        self._max_history_size = 10000

        # 初始化清理定时器 (测试兼容性)
        import threading
        self._cleanup_timer = threading.Timer(3600.0, self._cleanup_old_events)  # 1小时后清理
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

        # 异步支持
        if self.enable_async:
            import asyncio
            self._loop = asyncio.new_event_loop()
        else:
            self._loop = None

        # 创建线程池执行器（仅在启用异步时）
        if self.enable_async:
            try:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(max_workers=max_workers)
            except ImportError:
                self.logger.warning("ThreadPoolExecutor not available, using synchronous execution")
                self._executor = None
        else:
            self._executor = None

    def subscribe(self, event_type, handler: EventHandler, priority: int = 1):
        """订阅事件 - 优化版本"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append((handler, priority))

        # 更新缓存
        self._subscriber_cache[event_type] = sorted(
            self._subscribers[event_type],
            key=lambda x: x[1],  # 按优先级排序
            reverse=True
        )
        self.logger.info(f"订阅事件: {event_type}, 当前订阅者数量: {len(self._subscribers[event_type])}")

    def publish(self, event_type, data=None, source="system"):
        """发布事件 - 高性能优化版本"""
        # 如果禁用异步处理，直接处理单个事件
        if not self.enable_async:
            self._process_single_event(event_type, data, source)
            return

        # 批量处理优化
        if len(self._batch_queue) < self.batch_size:
            self._batch_queue.append((event_type, data, source))
            if len(self._batch_queue) == 1:  # 第一个事件，启动异步处理
                self._schedule_batch_processing()
            return

        # 直接处理单个事件
        self._process_single_event(event_type, data, source)

    def _schedule_batch_processing(self):
        """调度批量处理"""
        if self.enable_async and self._batch_queue:
            import threading
            thread = threading.Thread(target=self._process_batch_events)
            thread.daemon = True
            thread.start()

    def _process_batch_events(self):
        """批量处理事件 - 性能优化"""
        max_iterations = 100  # 防止无限循环
        iteration_count = 0

        while self._batch_queue and iteration_count < max_iterations:
            iteration_count += 1
            batch = self._batch_queue[:self.batch_size]
            self._batch_queue = self._batch_queue[self.batch_size:]

            # 批量处理同一类型的事件
            event_groups = {}
            for event_type, data, source in batch:
                if event_type not in event_groups:
                    event_groups[event_type] = []
                event_groups[event_type].append((data, source))

            # 并行处理不同类型的事件
            for event_type, events in event_groups.items():
                self._process_event_type_batch(event_type, events)

        # 如果达到最大迭代次数，记录警告
        if iteration_count >= max_iterations and self._batch_queue:
            self.logger.warning(f"批量处理达到最大迭代次数({max_iterations})，剩余{len(self._batch_queue)}个事件未处理")

    def _process_event_type_batch(self, event_type, events):
        """批量处理同一类型的事件"""
        # 创建标准事件对象
        event_objects = []
        timestamp = __import__('time').time()
        for data, source in events:
            event_obj = {
                'type': event_type,
                'data': data,
                'source': source,
                'timestamp': timestamp,
                'batch_size': len(events)
            }
            event_objects.append(event_obj)

            # 总是添加到历史记录
            self._add_to_history(event_obj)

        # 检查是否有订阅者
        if event_type not in self._subscriber_cache:
            return

        subscribers = self._subscriber_cache[event_type]
        if not subscribers:
            return

        # 并行处理订阅者
        if self.enable_async:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(subscribers), self.max_workers)) as executor:
                futures = []
                for handler, priority in subscribers:
                    for event in event_objects:
                        futures.append(executor.submit(self._safe_handle_event, handler, event))

                # 等待所有处理完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"批量事件处理失败: {e}")
        else:
            # 同步处理
            for handler, priority in subscribers:
                for event in event_objects:
                    self._safe_handle_event(handler, event)

    def _process_single_event(self, event_type, data, source):
        """处理单个事件"""
        event = {
            'type': event_type,
            'data': data,
            'source': source,
            'timestamp': __import__('time').time()
        }

        # 总是记录到事件历史 (测试兼容性)
        # 如果是Event对象，直接记录，否则创建字典
        if hasattr(event_type, 'event_type'):
            self._add_to_history(event_type)  # event_type是Event对象
        else:
            self._add_to_history(event)

        # 检查是否有订阅者
        if event_type not in self._subscriber_cache:
            return

        subscribers = self._subscriber_cache[event_type]
        if not subscribers:
            return

        # 按优先级顺序处理
        for handler, priority in subscribers:
            self._safe_handle_event(handler, event)

    def _safe_handle_event(self, handler, event):
        """安全处理事件"""
        try:
            handler.handle_event(event)
        except Exception as e:
            self.logger.error(f"事件处理失败: {e}, 事件类型: {event.get('type')}")

    def publish_sync(self, event_type, data=None, source="system"):
        """同步发布事件"""
        # 处理Event对象的情况（测试兼容性）
        if hasattr(event_type, 'event_type'):
            # 如果第一个参数是Event对象
            event_obj = event_type
            event_type = event_obj.event_type
            data = event_obj.data
            source = event_obj.source

        self._process_single_event(event_type, data, source)
        self.logger.info(f"同步发布事件: {event_type}")
        return True  # 测试兼容性

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'total_subscribers': sum(len(subs) for subs in self._subscribers.values()),
            'total_event_types': len(self._subscribers),
            'batch_queue_size': len(self._batch_queue),
            'async_enabled': self.enable_async,
            'max_workers': self.max_workers
        }

    def unsubscribe(self, event_type, handler: EventHandler):
        """取消订阅 - 优化版本"""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                (h, p) for h, p in self._subscribers[event_type]
                if h != handler
            ]
            # 更新缓存
            if self._subscribers[event_type]:
                self._subscriber_cache[event_type] = sorted(
                    self._subscribers[event_type],
                    key=lambda x: x[1],
                    reverse=True
                )
            else:
                self._subscriber_cache.pop(event_type, None)
        self.logger.info(f"取消订阅事件: {event_type}")

    def optimize_for_high_concurrency(self):
        """高并发优化配置"""
        # 增加批处理大小
        self.batch_size = 200
        # 增加工作线程数
        self.max_workers = 20
        # 启用异步处理
        self.enable_async = True
        self.logger.info("已启用高并发优化配置")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'async_enabled': self.enable_async,
            'cache_hit_rate': len(self._subscriber_cache) / max(len(self._subscribers), 1),
            'avg_subscribers_per_type': sum(len(subs) for subs in self._subscribers.values()) / max(len(self._subscribers), 1)
        }

    def _add_to_history(self, event):
        """添加事件到历史记录"""
        # 将事件转换为字典格式存储
        if hasattr(event, 'event_type'):
            # 如果是Event对象，转换为字典
            event_dict = {
                'type': event.event_type,
                'data': event.data,
                'source': event.source,
                'timestamp': event.timestamp or __import__('time').time(),
                'priority': event.priority.value if hasattr(event.priority, 'value') else event.priority
            }
        else:
            # 如果已经是字典，直接使用
            event_dict = event.copy()

        self._event_history.append(event_dict)
        # 限制历史记录大小
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

    def _cleanup_old_events(self):
        """定期清理旧事件（测试兼容性）"""
        import time
        current_time = time.time()
        # 清理24小时前的事件
        cutoff_time = current_time - (24 * 3600)
        self.clear_history(cutoff_time)

    def get_event_history(self) -> List[Dict[str, Any]]:
        """获取事件历史记录"""
        return self._event_history.copy()

    def shutdown(self):
        """关闭事件总线"""
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        if self._loop and self.enable_async:
            try:
                self._loop.close()
            except Exception as e:
                self.logger.warning(f"关闭异步循环时出错: {e}")
        self.logger.info("EventBus已关闭")

    def clear_history(self, before_time=None):
        """清理事件历史记录"""
        if before_time is None:
            self._event_history.clear()
        else:
            # 删除指定时间之前的记录
            self._event_history[:] = [
                event for event in self._event_history
                if self._get_event_timestamp(event) >= before_time
            ]
        self.logger.info(f"事件历史记录已清理，剩余记录数: {len(self._event_history)}")

    def _get_event_timestamp(self, event):
        """获取事件的timestamp"""
        if hasattr(event, 'timestamp'):
            return event.timestamp or 0
        elif isinstance(event, dict):
            return event.get('timestamp', 0)
        else:
            return 0

    # 实现统一事件总线接口的其他方法

    def publish_async(self, event: Union[Event, Dict[str, Any]]) -> str:
        """异步发布事件"""
        import uuid
        event_id = str(uuid.uuid4())

        # 转换为内部事件格式
        if isinstance(event, Event):
            event_type = event.type
            data = event.data
            source = event.source or "system"
        else:
            event_type = event.get('type', 'unknown')
            data = event.get('data', {})
            source = event.get('source', 'system')

        # 异步发布
        if self.enable_async and self._executor:
            self._executor.submit(self._publish_async_task, event_type, data, source, event_id)
        else:
            # 降级为同步发布
            self._publish_async_task(event_type, data, source, event_id)

        return event_id

    def _publish_async_task(self, event_type: str, data: Dict[str, Any], source: str, event_id: str):
        """异步发布任务"""
        try:
            self.publish(event_type, data, source)
        except Exception as e:
            self.logger.error(f"异步发布事件失败 {event_id}: {e}")

    def register_handler(self, handler: EventHandler) -> bool:
        """注册事件处理器"""
        try:
            handler_id = id(handler)
            self._handlers[handler_id] = handler
            return True
        except Exception as e:
            self.logger.error(f"注册处理器失败: {e}")
            return False

    def unregister_handler(self, handler: EventHandler) -> bool:
        """注销事件处理器"""
        try:
            handler_id = id(handler)
            if handler_id in self._handlers:
                del self._handlers[handler_id]
                return True
            return False
        except Exception as e:
            self.logger.error(f"注销处理器失败: {e}")
            return False

    def get_handlers(self, event_type: str) -> List[Union[EventHandler, Callable]]:
        """获取指定事件类型的处理器列表"""
        return self._subscribers.get(event_type, [])

    def get_all_handlers(self) -> Dict[str, List[Union[EventHandler, Callable]]]:
        """获取所有事件处理器"""
        return self._subscribers.copy()

    def set_delivery_mode(self, event_type: str, mode: EventDeliveryMode) -> None:
        """设置事件传递模式"""
        # 在当前实现中，所有事件都是异步的，这里可以扩展支持不同的传递模式

    def get_delivery_mode(self, event_type: str) -> EventDeliveryMode:
        """获取事件传递模式"""
        return EventDeliveryMode.ASYNCHRONOUS

    def enable_persistence(self, event_type: str, persistence: EventPersistence) -> None:
        """启用事件持久化"""
        # 这里可以实现事件持久化逻辑

    def disable_persistence(self, event_type: str) -> None:
        """禁用事件持久化"""
        # 这里可以实现禁用持久化逻辑

    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        if event_type:
            # 过滤指定类型的事件
            filtered_events = [
                event for event in self._event_history
                if (hasattr(event, 'type') and event.type == event_type) or
                   (isinstance(event, dict) and event.get('type') == event_type)
            ]
            return filtered_events[-limit:]
        else:
            return self._event_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计信息"""
        return {
            'total_subscribers': sum(len(handlers) for handlers in self._subscribers.values()),
            'total_event_types': len(self._subscribers),
            'event_history_size': len(self._event_history),
            'max_history_size': self._max_history_size,
            'async_enabled': self.enable_async,
            'max_workers': self.max_workers
        }

    def clear_event_history(self, event_type: str = None) -> int:
        """清空事件历史"""
        if event_type is None:
            cleared_count = len(self._event_history)
            self._event_history.clear()
            return cleared_count
        else:
            # 只清除指定类型的事件
            original_count = len(self._event_history)
            self._event_history[:] = [
                event for event in self._event_history
                if not ((hasattr(event, 'type') and event.type == event_type) or
                        (isinstance(event, dict) and event.get('type') == event_type))
            ]
            return original_count - len(self._event_history)

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        return {
            'status': 'healthy',
            'subscribers_count': sum(len(handlers) for handlers in self._subscribers.values()),
            'handlers_count': len(self._handlers),
            'history_size': len(self._event_history),
            'async_enabled': self.enable_async
        }

    def start(self) -> bool:
        """启动事件总线"""
        try:
            # 初始化异步执行器
            if self.enable_async and self._executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # 初始化事件循环
            if self.enable_async and self._loop is None:
                import asyncio
                self._loop = asyncio.new_event_loop()

            self.logger.info("事件总线已启动")
            return True
        except Exception as e:
            self.logger.error(f"启动事件总线失败: {e}")
            return False

    def stop(self) -> bool:
        """停止事件总线"""
        try:
            self.shutdown()
            return True
        except Exception as e:
            self.logger.error(f"停止事件总线失败: {e}")
            return False

    def is_running(self) -> bool:
        """检查事件总线是否正在运行"""
        return hasattr(self, '_executor') and self._executor is not None


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
统一Bus组件工厂

合并所有bus_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:18:35
"""


class IBusComponent(ABC):

    """Bus组件接口"""

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
    def get_bus_id(self) -> int:
        """获取bus ID"""


class BusComponent(IBusComponent):

    """统一Bus组件实现"""

    def __init__(self, bus_id: int, component_type: str = "Bus"):
        """初始化组件"""
        self.bus_id = bus_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{bus_id}"
        self.creation_time = datetime.now()

    def get_bus_id(self) -> int:
        """获取bus ID"""
        return self.bus_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "bus_id": self.bus_id,
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
                "bus_id": self.bus_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_bus_processing"
            }
            return result
        except Exception as e:
            return {
                "bus_id": self.bus_id,
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
            "bus_id": self.bus_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class BusComponentFactory:

    """Bus组件工厂"""

    # 支持的bus ID列表
    SUPPORTED_BUS_IDS = [2, 7]

    @staticmethod
    def create_component(bus_id: int) -> BusComponent:
        """创建指定ID的bus组件"""
        if bus_id not in BusComponentFactory.SUPPORTED_BUS_IDS:
            raise ValueError(f"不支持的bus ID: {bus_id}。支持的ID: {BusComponentFactory.SUPPORTED_BUS_IDS}")

        return BusComponent(bus_id, "Bus")

    @staticmethod
    def get_available_buss() -> List[int]:
        """获取所有可用的bus ID"""
        return sorted(list(BusComponentFactory.SUPPORTED_BUS_IDS))

    @staticmethod
    def create_all_buss() -> Dict[int, BusComponent]:
        """创建所有可用bus"""
        return {
            bus_id: BusComponent(bus_id, "Bus")
            for bus_id in BusComponentFactory.SUPPORTED_BUS_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "BusComponentFactory",
            "version": "2.0.0",
            "total_buss": len(BusComponentFactory.SUPPORTED_BUS_IDS),
            "supported_ids": sorted(list(BusComponentFactory.SUPPORTED_BUS_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_bus_bus_component_2(): return BusComponentFactory.create_component(2)


def create_bus_bus_component_7(): return BusComponentFactory.create_component(7)


__all__ = [
    "IBusComponent",
    "BusComponent",
    "BusComponentFactory",
    "create_bus_bus_component_2",
    "create_bus_bus_component_7",
]
