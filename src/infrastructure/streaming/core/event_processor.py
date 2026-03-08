"""
Event Processor Module
事件处理器模块

This module provides event processing capabilities for streaming systems
此模块为流系统提供事件处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types enumeration"""
    DATA_ARRIVAL = "data_arrival"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_STATUS = "system_status"
    PERFORMANCE_ALERT = "performance_alert"
    CUSTOM_EVENT = "custom_event"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class StreamingEvent:
    """
    Streaming Event Class
    流事件类

    Represents an event in the streaming system
    表示流系统中的事件
    """

    def __init__(self,
                 event_type: EventType,
                 source: str,
                 data: Any = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 timestamp: Optional[datetime] = None):
        """
        Initialize a streaming event
        初始化流事件

        Args:
            event_type: Type of the event
                       事件类型
            source: Source that generated the event
                   生成事件的来源
            data: Event data payload
                 事件数据负载
            priority: Event priority level
                     事件优先级
            timestamp: Event timestamp (auto - generated if None)
                      事件时间戳（如果为None则自动生成）
        """
        self.event_type = event_type
        self.source = source
        self.data = data
        self.priority = priority
        self.timestamp = timestamp or datetime.now()
        self.event_id = f"{event_type.value}_{source}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def __lt__(self, other):
        """Less than comparison for PriorityQueue"""
        if not isinstance(other, StreamingEvent):
            return NotImplemented
        # Compare by priority first, then by timestamp
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority = lower number
        return self.timestamp < other.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary
        将事件转换为字典

        Returns:
            dict: Event data as dictionary
                  事件数据字典
        """
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'data': self.data,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        return f"Event({self.event_id}: {self.event_type.value} from {self.source})"


class EventProcessor:
    """
    Event Processor for Streaming Systems
    流系统事件处理器

    Processes and routes events in streaming pipelines
    处理和路由流管道中的事件
    """

    def __init__(self, processor_name: str = "default_event_processor"):
        """
        Initialize the event processor
        初始化事件处理器

        Args:
            processor_name: Name of this processor
                          此处理器的名称
        """
        self.processor_name = processor_name
        self.event_queue = queue.PriorityQueue()
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.is_running = False
        self.processing_thread = None
        self.processed_events = 0
        self.error_count = 0

        # Initialize empty handler lists for all event types
        for event_type in EventType:
            self.event_handlers[event_type] = []

        logger.info(f"Event processor {processor_name} initialized")

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register an event handler
        注册事件处理器

        Args:
            event_type: Type of event to handle
                       要处理的事件类型
            handler: Handler function
                    处理函数
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} in {self.processor_name}")

    def unregister_handler(self, event_type: EventType, handler: Callable) -> bool:
        """
        Unregister an event handler
        取消注册事件处理器

        Args:
            event_type: Type of event
                       事件类型
            handler: Handler function to remove
                    要移除的处理函数

        Returns:
            bool: True if handler was removed, False otherwise
                  如果处理器被移除则返回True，否则返回False
        """
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            logger.info(f"Unregistered handler for {event_type.value} in {self.processor_name}")
            return True
        return False

    def emit_event(self, event: StreamingEvent) -> bool:
        """
        Emit an event to the processing queue
        将事件发送到处理队列

        Args:
            event: Event to emit
                  要发送的事件

        Returns:
            bool: True if event was queued successfully, False otherwise
                  如果事件成功排队则返回True，否则返回False
        """
        try:
            # Use negative priority for PriorityQueue (higher priority = lower number)
            priority_item = (-event.priority.value, event)
            self.event_queue.put(priority_item, timeout=1.0)
            logger.debug(f"Event emitted: {event}")
            return True
        except queue.Full:
            logger.warning(f"Event queue full in {self.processor_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to emit event in {self.processor_name}: {str(e)}")
            return False

    def start_processing(self) -> bool:
        """
        Start event processing
        开始事件处理

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.processor_name} is already running")
            return False

        try:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info(f"Event processing started in {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start event processing in {self.processor_name}: {str(e)}")
            self.is_running = False
            return False

    def stop_processing(self) -> bool:
        """
        Stop event processing
        停止事件处理

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.processor_name} is not running")
            return False

        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                # Send a sentinel event to stop the processing loop
                sentinel_event = StreamingEvent(EventType.CUSTOM_EVENT, "system", "stop")
                self.event_queue.put((-999, sentinel_event))
                self.processing_thread.join(timeout=5.0)
            logger.info(f"Event processing stopped in {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop event processing in {self.processor_name}: {str(e)}")
            return False

    def _processing_loop(self) -> None:
        """
        Main event processing loop
        主要事件处理循环
        """
        logger.info(f"Event processing loop started for {self.processor_name}")

        while self.is_running:
            try:
                # Get event from queue with timeout
                priority_item = self.event_queue.get(timeout=0.5)
                _, event = priority_item

                # Skip sentinel events
                if hasattr(event, 'data') and event.data == "stop":
                    break

                # Process the event
                self._process_event(event)

                self.event_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event processing error in {self.processor_name}: {str(e)}")
                self.error_count += 1

        logger.info(f"Event processing loop stopped for {self.processor_name}")

    def _process_event(self, event: StreamingEvent) -> None:
        """
        Process a single event
        处理单个事件

        Args:
            event: Event to process
                  要处理的事件
        """
        try:
            handlers = self.event_handlers.get(event.event_type, [])

            if not handlers:
                logger.debug(f"No handlers registered for {event.event_type.value}")
                return

            # Call all registered handlers
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event.event_type.value}: {str(e)}")
                    self.error_count += 1

            self.processed_events += 1
            logger.debug(f"Event processed: {event}")

        except Exception as e:
            logger.error(f"Failed to process event {event}: {str(e)}")
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        获取处理器统计信息

        Returns:
            dict: Processor statistics
                  处理器统计信息
        """
        handler_counts = {}
        for event_type, handlers in self.event_handlers.items():
            handler_counts[event_type.value] = len(handlers)

        return {
            'processor_name': self.processor_name,
            'is_running': self.is_running,
            'processed_events': self.processed_events,
            'error_count': self.error_count,
            'queue_size': self.event_queue.qsize(),
            'handler_counts': handler_counts,
            'success_rate': (self.processed_events / max(self.processed_events + self.error_count, 1)) * 100
        }

    def clear_handlers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear event handlers
        清除事件处理器

        Args:
            event_type: Specific event type to clear (None for all)
                       要清除的特定事件类型（None表示全部）
        """
        if event_type:
            self.event_handlers[event_type] = []
            logger.info(f"Cleared handlers for {event_type.value}")
        else:
            for et in self.event_handlers:
                self.event_handlers[et] = []
            logger.info("Cleared all event handlers")


# Utility functions for event handling

def create_data_event(source: str, data: Any, priority: EventPriority = EventPriority.NORMAL) -> StreamingEvent:
    """
    Create a data arrival event
    创建数据到达事件

    Args:
        source: Event source
               事件来源
        data: Event data
             事件数据
        priority: Event priority
                 事件优先级

    Returns:
        StreamingEvent: Created event
                       创建的事件
    """
    return StreamingEvent(EventType.DATA_ARRIVAL, source, data, priority)


def create_error_event(source: str, error_data: Any) -> StreamingEvent:
    """
    Create an error event
    创建错误事件

    Args:
        source: Event source
               事件来源
        error_data: Error data
                   错误数据

    Returns:
        StreamingEvent: Created error event
                       创建的错误事件
    """
    return StreamingEvent(EventType.ERROR_OCCURRED, source, error_data, EventPriority.HIGH)


def create_performance_event(source: str, metrics: Dict[str, Any]) -> StreamingEvent:
    """
    Create a performance alert event
    创建性能警报事件

    Args:
        source: Event source
               事件来源
        metrics: Performance metrics
                性能指标

    Returns:
        StreamingEvent: Created performance event
                       创建的性能事件
    """
    priority = EventPriority.HIGH if any(v > 90 for v in metrics.values(
    ) if isinstance(v, (int, float))) else EventPriority.NORMAL
    return StreamingEvent(EventType.PERFORMANCE_ALERT, source, metrics, priority)


# Global default event processor instance
# 全局默认事件处理器实例

default_event_processor = EventProcessor("default_event_processor")

__all__ = [
    'EventType',
    'EventPriority',
    'StreamingEvent',
    'EventProcessor',
    'default_event_processor',
    'create_data_event',
    'create_error_event',
    'create_performance_event'
]
