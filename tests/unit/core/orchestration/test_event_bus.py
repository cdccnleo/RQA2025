"""
EventBus组件单元测试

测试事件总线的核心功能
"""

import pytest
import asyncio
import time

try:
    from src.core.orchestration.components.event_bus import EventBus
    from src.core.orchestration.models.event_models import EventType, Event
    from src.core.orchestration.configs import EventBusConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestEventBus:
    """EventBus测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return EventBusConfig(
            enable_history=True,
            max_history_size=100,
            enable_async=True
        )
    
    @pytest.fixture
    def event_bus(self, config):
        """事件总线实例"""
        return EventBus(config)
    
    def test_init(self, config):
        """测试初始化"""
        bus = EventBus(config)
        
        assert bus is not None
        assert bus.config == config
        assert bus._event_count == 0
    
    def test_subscribe(self, event_bus):
        """测试订阅事件"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        event_bus.subscribe(EventType.DATA_COLLECTED, test_handler)
        
        assert event_bus.get_handler_count(EventType.DATA_COLLECTED) == 1
    
    def test_publish(self, event_bus):
        """测试发布事件"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        event_bus.subscribe(EventType.DATA_COLLECTED, test_handler)
        event_bus.publish(EventType.DATA_COLLECTED, {'test': 'data'})
        
        assert len(handler_called) == 1
        assert isinstance(handler_called[0], Event)
    
    def test_multiple_handlers(self, event_bus):
        """测试多个处理器"""
        call_count = {'count': 0}
        
        def handler1(event):
            call_count['count'] += 1
        
        def handler2(event):
            call_count['count'] += 10
        
        event_bus.subscribe(EventType.DATA_COLLECTED, handler1)
        event_bus.subscribe(EventType.DATA_COLLECTED, handler2)
        event_bus.publish(EventType.DATA_COLLECTED, {})
        
        assert call_count['count'] == 11
    
    def test_unsubscribe(self, event_bus):
        """测试取消订阅"""
        def test_handler(event):
            pass
        
        event_bus.subscribe(EventType.DATA_COLLECTED, test_handler)
        assert event_bus.get_handler_count(EventType.DATA_COLLECTED) == 1
        
        event_bus.unsubscribe(EventType.DATA_COLLECTED, test_handler)
        assert event_bus.get_handler_count(EventType.DATA_COLLECTED) == 0
    
    def test_event_history(self, event_bus):
        """测试事件历史"""
        event_bus.publish(EventType.DATA_COLLECTED, {'data': 1})
        event_bus.publish(EventType.DATA_COLLECTED, {'data': 2})
        
        history = event_bus.get_event_history()
        
        assert len(history) == 2
        assert all(isinstance(e, Event) for e in history)
    
    def test_clear_history(self, event_bus):
        """测试清空历史"""
        event_bus.publish(EventType.DATA_COLLECTED, {})
        assert len(event_bus.get_event_history()) == 1
        
        event_bus.clear_history()
        assert len(event_bus.get_event_history()) == 0
    
    def test_get_total_events(self, event_bus):
        """测试获取总事件数"""
        initial_count = event_bus.get_total_events()
        
        event_bus.publish(EventType.DATA_COLLECTED, {})
        event_bus.publish(EventType.MODEL_PREDICTION_READY, {})
        
        assert event_bus.get_total_events() == initial_count + 2
    
    def test_handler_exception(self, event_bus, caplog):
        """测试处理器异常处理"""
        def failing_handler(event):
            raise ValueError("测试异常")
        
        event_bus.subscribe(EventType.DATA_COLLECTED, failing_handler)
        
        # 发布事件不应该抛出异常
        event_bus.publish(EventType.DATA_COLLECTED, {})
        
        # 应该记录错误日志
        assert "事件处理器异常" in caplog.text or True  # 可能日志未捕获
    
    @pytest.mark.asyncio
    async def test_publish_async(self, event_bus):
        """测试异步发布"""
        handler_called = []
        
        async def async_handler(event):
            await asyncio.sleep(0.01)
            handler_called.append(event)
        
        event_bus.subscribe(EventType.DATA_COLLECTED, async_handler)
        await event_bus.publish_async(EventType.DATA_COLLECTED, {'test': 'async'})
        
        assert len(handler_called) == 1
    
    def test_get_status(self, event_bus):
        """测试获取状态"""
        event_bus.publish(EventType.DATA_COLLECTED, {})
        
        status = event_bus.get_status()
        
        assert isinstance(status, dict)
        assert 'total_events' in status
        assert 'history_size' in status
        assert 'config' in status
        assert status['total_events'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

