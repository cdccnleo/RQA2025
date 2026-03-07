"""
事件总线组件测试覆盖率补充

补充event_publisher、event_subscriber、event_processor、event_monitor的测试覆盖
"""

import time
import queue
from unittest.mock import Mock, MagicMock, patch
from collections import deque
import pytest

from src.core.event_bus.components.event_publisher import EventPublisher
from src.core.event_bus.components.event_subscriber import EventSubscriber
from src.core.event_bus.components.event_processor import EventProcessor, EventProcessingResult
from src.core.event_bus.components.event_monitor import EventMonitor
from src.core.event_bus.models import Event, EventHandler
from src.core.event_bus.types import EventType, EventPriority
from src.core.foundation.base import ComponentHealth


@pytest.fixture
def mock_managers():
    """创建Mock管理器"""
    filter_manager = Mock()
    filter_manager.apply_filters.return_value = True
    filter_manager.apply_transformers.side_effect = lambda e: e
    
    routing_manager = Mock()
    routing_manager.route_event.return_value = []
    routing_manager._routes = {}
    routing_manager._dead_letter_queue = deque(maxlen=100)
    
    persistence_manager = Mock()
    persistence_manager.persist_event.return_value = None
    persistence_manager.add_to_history.return_value = None
    
    statistics_manager = Mock()
    statistics_manager.update_statistics.return_value = None
    statistics_manager.get_statistics.return_value = {}
    
    return {
        'filter_manager': filter_manager,
        'routing_manager': routing_manager,
        'persistence_manager': persistence_manager,
        'statistics_manager': statistics_manager
    }


@pytest.fixture
def event_publisher(mock_managers):
    """创建EventPublisher实例"""
    event_queue = queue.PriorityQueue()
    lock = Mock()
    dead_letter_queue = deque(maxlen=100)
    
    return EventPublisher(
        filter_manager=mock_managers['filter_manager'],
        routing_manager=mock_managers['routing_manager'],
        persistence_manager=mock_managers['persistence_manager'],
        statistics_manager=mock_managers['statistics_manager'],
        event_queue=event_queue,
        lock=lock,
        dead_letter_queue=dead_letter_queue
    )


@pytest.fixture
def event_subscriber():
    """创建EventSubscriber实例"""
    handlers = {}
    async_handlers = {}
    lock = Mock()
    
    return EventSubscriber(
        handlers=handlers,
        async_handlers=async_handlers,
        lock=lock
    )


@pytest.fixture
def event_processor(event_subscriber, mock_managers):
    """创建EventProcessor实例"""
    lock = Mock()
    
    return EventProcessor(
        subscriber=event_subscriber,
        statistics_manager=mock_managers['statistics_manager'],
        lock=lock,
        retry_manager=None,
        performance_monitor=None
    )


@pytest.fixture
def event_monitor(mock_managers):
    """创建EventMonitor实例"""
    event_queue = queue.Queue()
    worker_threads = [Mock(is_alive=lambda: True)]
    
    return EventMonitor(
        statistics_manager=mock_managers['statistics_manager'],
        event_queue=event_queue,
        max_queue_size=1000,
        worker_threads=worker_threads,
        start_time=time.time()
    )


class TestEventPublisher:
    """测试EventPublisher组件"""

    def test_publish_with_parameters(self, event_publisher):
        """测试使用参数发布事件"""
        event_id = event_publisher.publish(
            event_type=EventType.BUSINESS,
            data={"key": "value"},
            source="test_source",
            priority=EventPriority.HIGH,
            event_id="test_event_1",
            correlation_id="corr_1"
        )
        
        assert event_id == "test_event_1"
        event_publisher.persistence_manager.persist_event.assert_called_once()

    def test_publish_event_object(self, event_publisher):
        """测试发布事件对象"""
        event = Event(
            event_type=EventType.SYSTEM,
            data={"test": "data"},
            source="test",
            priority=EventPriority.NORMAL,
            event_id="test_event_2"
        )
        
        event_id = event_publisher.publish_event(event)
        
        assert event_id == "test_event_2"
        event_publisher.persistence_manager.persist_event.assert_called_once()

    def test_publish_event_filtered(self, event_publisher, mock_managers):
        """测试事件被过滤"""
        mock_managers['filter_manager'].apply_filters.return_value = False
        
        event_id = event_publisher.publish(
            event_type=EventType.BUSINESS,
            data={},
            event_id="filtered_event"
        )
        
        assert event_id == "filtered_event"
        # 被过滤的事件不应该持久化
        event_publisher.persistence_manager.persist_event.assert_not_called()

    def test_publish_event_with_routing(self, event_publisher, mock_managers):
        """测试事件路由"""
        mock_managers['routing_manager'].route_event.return_value = ["routed_event_type"]
        
        event = Event(
            event_type=EventType.BUSINESS,
            data={"key": "value"},
            event_id="routed_event"
        )
        
        event_id = event_publisher.publish_event(event)
        
        assert event_id == "routed_event"
        # 应该创建路由事件
        assert mock_managers['routing_manager'].route_event.called

    def test_publish_event_error_handling(self, event_publisher, mock_managers):
        """测试发布事件错误处理"""
        mock_managers['persistence_manager'].persist_event.side_effect = Exception("Persistence failed")
        
        event = Event(
            event_type=EventType.BUSINESS,
            data={},
            event_id="error_event"
        )
        
        with pytest.raises(Exception):
            event_publisher.publish_event(event)
        
        # 错误应该被记录到死信队列
        assert len(event_publisher._dead_letter_queue) > 0


class TestEventSubscriber:
    """测试EventSubscriber组件"""

    def test_subscribe_sync_handler(self, event_subscriber):
        """测试订阅同步处理器"""
        handler = Mock()
        
        result = event_subscriber.subscribe(
            event_type=EventType.BUSINESS,
            handler=handler,
            priority=EventPriority.NORMAL,
            async_handler=False
        )
        
        assert result is True
        handlers, async_handlers = event_subscriber.get_handlers(str(EventType.BUSINESS))
        assert len(handlers) == 1
        assert len(async_handlers) == 0

    def test_subscribe_async_handler(self, event_subscriber):
        """测试订阅异步处理器"""
        handler = Mock()
        
        result = event_subscriber.subscribe(
            event_type=EventType.SYSTEM,
            handler=handler,
            async_handler=True
        )
        
        assert result is True
        handlers, async_handlers = event_subscriber.get_handlers(str(EventType.SYSTEM))
        assert len(handlers) == 0
        assert len(async_handlers) == 1

    def test_subscribe_async_method(self, event_subscriber):
        """测试subscribe_async方法"""
        handler = Mock()
        
        result = event_subscriber.subscribe_async(
            event_type=EventType.DATA,
            handler=handler
        )
        
        assert result is True
        handlers, async_handlers = event_subscriber.get_handlers(str(EventType.DATA))
        assert len(async_handlers) == 1

    def test_unsubscribe(self, event_subscriber):
        """测试取消订阅"""
        handler1 = Mock()
        handler2 = Mock()
        
        # 订阅两个处理器
        event_subscriber.subscribe(EventType.BUSINESS, handler1)
        event_subscriber.subscribe(EventType.BUSINESS, handler2)
        
        # 取消订阅handler1
        result = event_subscriber.unsubscribe(EventType.BUSINESS, handler1)
        
        assert result is True
        handlers, _ = event_subscriber.get_handlers(str(EventType.BUSINESS))
        assert len(handlers) == 1
        assert handlers[0].handler == handler2

    def test_get_subscriber_count(self, event_subscriber):
        """测试获取订阅者数量"""
        handler1 = Mock()
        handler2 = Mock()
        async_handler = Mock()
        
        event_subscriber.subscribe(EventType.BUSINESS, handler1)
        event_subscriber.subscribe(EventType.BUSINESS, handler2)
        event_subscriber.subscribe(EventType.BUSINESS, async_handler, async_handler=True)
        
        count = event_subscriber.get_subscriber_count(EventType.BUSINESS)
        assert count == 3


class TestEventProcessor:
    """测试EventProcessor组件"""

    def test_handle_event_success(self, event_processor, event_subscriber):
        """测试成功处理事件"""
        handler = Mock()
        event_subscriber.subscribe(EventType.BUSINESS, handler)
        
        event = Event(
            event_type=EventType.BUSINESS,
            data={"test": "data"},
            event_id="test_event"
        )
        
        result = event_processor.handle_event(event)
        
        assert isinstance(result, EventProcessingResult)
        assert result.success is True
        assert result.event_id == "test_event"
        assert result.sync_handlers_executed > 0
        handler.assert_called_once()

    def test_handle_event_no_handlers(self, event_processor):
        """测试没有处理器的事件"""
        event = Event(
            event_type=EventType.BUSINESS,
            data={},
            event_id="no_handler_event"
        )
        
        result = event_processor.handle_event(event)
        
        assert result.success is True  # 没有处理器也算成功
        assert result.sync_handlers_executed == 0

    def test_handle_event_handler_error(self, event_processor, event_subscriber):
        """测试处理器错误"""
        handler = Mock(side_effect=Exception("Handler error"))
        event_subscriber.subscribe(EventType.BUSINESS, handler)
        
        event = Event(
            event_type=EventType.BUSINESS,
            data={},
            event_id="error_event"
        )
        
        result = event_processor.handle_event(event)
        
        assert result.success is False
        assert len(result.errors) > 0

    def test_handle_event_async_handlers(self, event_processor, event_subscriber):
        """测试异步处理器"""
        async_handler = Mock()
        event_subscriber.subscribe(EventType.BUSINESS, async_handler, async_handler=True)
        
        event = Event(
            event_type=EventType.BUSINESS,
            data={},
            event_id="async_event"
        )
        
        result = event_processor.handle_event(event)
        
        assert result.async_handlers_executed > 0


class TestEventMonitor:
    """测试EventMonitor组件"""

    def test_check_health_healthy(self, event_monitor):
        """测试健康检查 - 健康状态"""
        health = event_monitor.check_health('RUNNING')
        
        assert health == ComponentHealth.HEALTHY

    def test_check_health_unhealthy_status(self, event_monitor):
        """测试健康检查 - 不健康状态"""
        health = event_monitor.check_health('ERROR')
        
        assert health == ComponentHealth.UNHEALTHY

    def test_check_health_queue_full(self, event_monitor):
        """测试健康检查 - 队列满"""
        # 填满队列
        for i in range(950):  # 90% of 1000
            event_monitor._event_queue.put(i)
        
        health = event_monitor.check_health('RUNNING')
        
        assert health == ComponentHealth.UNHEALTHY

    def test_check_health_thread_dead(self, event_monitor):
        """测试健康检查 - 线程死亡"""
        dead_thread = Mock(is_alive=lambda: False)
        event_monitor._worker_threads = [dead_thread]
        
        health = event_monitor.check_health('RUNNING')
        
        assert health == ComponentHealth.UNHEALTHY

    def test_get_statistics(self, event_monitor, mock_managers):
        """测试获取统计信息"""
        handlers = {str(EventType.BUSINESS): [Mock()]}
        async_handlers = {str(EventType.SYSTEM): [Mock()]}
        
        stats = event_monitor.get_statistics(
            event_counter=100,
            processed_counter=95,
            handlers=handlers,
            async_handlers=async_handlers,
            filter_manager=mock_managers['filter_manager'],
            routing_manager=mock_managers['routing_manager']
        )
        
        assert isinstance(stats, dict)
        assert stats['total_events_published'] == 100
        assert stats['total_events_processed'] == 95
        assert stats['active_handlers'] == 2
        assert 'queue_size' in stats
        assert 'worker_threads' in stats

    def test_get_statistics_with_performance_monitor(self, event_monitor, mock_managers):
        """测试带性能监控器的统计信息"""
        performance_monitor = Mock()
        performance_monitor.get_metrics.return_value = {
            'avg_processing_time': 0.1,
            'max_processing_time': 0.5
        }
        
        stats = event_monitor.get_statistics(
            event_counter=50,
            processed_counter=48,
            handlers={},
            async_handlers={},
            filter_manager=mock_managers['filter_manager'],
            routing_manager=mock_managers['routing_manager'],
            performance_monitor=performance_monitor
        )
        
        assert 'performance_metrics' in stats or 'avg_processing_time' in stats

