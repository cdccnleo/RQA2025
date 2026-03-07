#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件总线核心功能测试 - 覆盖率提升

测试目标：提升src/core/event_bus/core.py的覆盖率到80%+
注重测试质量，确保测试通过率
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 尝试导入，如果失败则跳过
# conftest.py已经设置了project_root到sys.path
try:
    from src.core.event_bus.core import (
        EventBus,
        EventBusConfig,
        EventProcessingResult,
        EventFilterManager
    )
    from src.core.event_bus.types import EventType, EventPriority
    from src.core.event_bus.models import Event
    IMPORTS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"事件总线模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestEventBusConfig:
    """测试事件总线配置"""

    def test_event_bus_config_defaults(self):
        """测试默认配置"""
        config = EventBusConfig()
        assert config.max_workers > 0
        assert config.enable_async is True
        assert isinstance(config.batch_size, int)

    def test_event_bus_config_custom(self):
        """测试自定义配置"""
        config = EventBusConfig(
            max_workers=10,
            enable_async=False,
            batch_size=20
        )
        assert config.max_workers == 10
        assert config.enable_async is False
        assert config.batch_size == 20


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestEventProcessingResult:
    """测试事件处理结果"""

    def test_event_processing_result_creation(self):
        """测试事件处理结果创建"""
        result = EventProcessingResult(
            event_id="test_id",
            event_type="test_type",
            success=True,
            processing_time=0.1
        )
        assert result.event_id == "test_id"
        assert result.event_type == "test_type"
        assert result.success is True
        assert result.processing_time == 0.1
        assert result.errors == []

    def test_event_processing_result_with_errors(self):
        """测试带错误的事件处理结果"""
        result = EventProcessingResult(
            event_id="test_id",
            event_type="test_type",
            success=False,
            processing_time=0.1,
            errors=["error1", "error2"]
        )
        assert result.success is False
        assert len(result.errors) == 2


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestEventFilterManager:
    """测试事件过滤器管理器"""

    def test_event_filter_manager_init(self):
        """测试过滤器管理器初始化"""
        manager = EventFilterManager()
        assert manager._filters == []
        assert manager._transformers == []

    def test_add_event_filter(self):
        """测试添加事件过滤器"""
        manager = EventFilterManager()
        filter_func = lambda e: True
        manager.add_event_filter(filter_func)
        assert len(manager._filters) == 1
        assert filter_func in manager._filters

    def test_remove_event_filter(self):
        """测试移除事件过滤器"""
        manager = EventFilterManager()
        filter_func = lambda e: True
        manager.add_event_filter(filter_func)
        manager.remove_event_filter(filter_func)
        assert len(manager._filters) == 0

    def test_add_event_transformer(self):
        """测试添加事件转换器"""
        manager = EventFilterManager()
        transformer_func = lambda e: e
        manager.add_event_transformer(transformer_func)
        assert len(manager._transformers) == 1

    def test_remove_event_transformer(self):
        """测试移除事件转换器"""
        manager = EventFilterManager()
        transformer_func = lambda e: e
        manager.add_event_transformer(transformer_func)
        manager.remove_event_transformer(transformer_func)
        assert len(manager._transformers) == 0

    def test_apply_filters_accept(self):
        """测试过滤器接受事件"""
        manager = EventFilterManager()
        filter_func = lambda e: True
        manager.add_event_filter(filter_func)
        
        event = Mock()
        event.event_type = "test"
        result = manager.apply_filters(event)
        assert result is True

    def test_apply_filters_reject(self):
        """测试过滤器拒绝事件"""
        manager = EventFilterManager()
        filter_func = lambda e: False
        manager.add_event_filter(filter_func)
        
        event = Mock()
        event.event_type = "test"
        result = manager.apply_filters(event)
        assert result is False

    def test_apply_transforms(self):
        """测试应用转换器"""
        manager = EventFilterManager()
        transformer_func = lambda e: e
        manager.add_event_transformer(transformer_func)
        
        event = Mock()
        event.event_type = "test"
        result = manager.apply_transforms(event)
        assert result == event


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestEventBusBasic:
    """测试事件总线基础功能"""

    @pytest.fixture
    def event_bus(self):
        """创建事件总线实例"""
        config = EventBusConfig(enable_persistence=False, enable_retry=False)
        return EventBus(config=config)

    def test_event_bus_initialization(self, event_bus):
        """测试事件总线初始化"""
        assert event_bus is not None
        assert hasattr(event_bus, 'subscribe')
        assert hasattr(event_bus, 'publish')

    def test_event_bus_subscribe(self, event_bus):
        """测试订阅事件"""
        handler_called = []
        
        def handler(event):
            handler_called.append(event)
        
        event_bus.subscribe(EventType.INFO, handler)
        assert len(event_bus._handlers[EventType.INFO]) == 1

    def test_event_bus_unsubscribe(self, event_bus):
        """测试取消订阅"""
        handler = Mock()
        event_bus.subscribe(EventType.INFO, handler)
        event_bus.unsubscribe(EventType.INFO, handler)
        assert len(event_bus._handlers.get(EventType.INFO, [])) == 0

    def test_event_bus_publish_sync(self, event_bus):
        """测试同步发布事件"""
        handler_called = []
        
        def handler(event):
            handler_called.append(event)
        
        event_bus.subscribe(EventType.INFO, handler)
        event_data = {"message": "test"}
        event_bus.publish(EventType.INFO, event_data)
        
        # 等待处理完成
        time.sleep(0.1)
        assert len(handler_called) > 0

    def test_event_bus_get_event_history(self, event_bus):
        """测试获取事件历史"""
        event_data = {"message": "test"}
        event_bus.publish(EventType.INFO, event_data)
        history = event_bus.get_event_history()
        assert len(history) > 0

    def test_event_bus_clear_history(self, event_bus):
        """测试清空事件历史"""
        event_data = {"message": "test"}
        event_bus.publish(EventType.INFO, event_data)
        event_bus.clear_history()
        history = event_bus.get_event_history()
        assert len(history) == 0

    def test_event_bus_get_stats(self, event_bus):
        """测试获取统计信息"""
        stats = event_bus.get_stats()
        assert isinstance(stats, dict)
        assert "total_events" in stats or "published_events" in stats


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestEventBusAdvanced:
    """测试事件总线高级功能"""

    @pytest.fixture
    def event_bus(self):
        """创建事件总线实例"""
        config = EventBusConfig(enable_persistence=False, enable_retry=False)
        return EventBus(config=config)

    def test_event_bus_multiple_handlers(self, event_bus):
        """测试多个处理器"""
        handler1_called = []
        handler2_called = []
        
        def handler1(event):
            handler1_called.append(event)
        
        def handler2(event):
            handler2_called.append(event)
        
        event_bus.subscribe(EventType.INFO, handler1)
        event_bus.subscribe(EventType.INFO, handler2)
        
        event_data = {"message": "test"}
        event_bus.publish(EventType.INFO, event_data)
        
        time.sleep(0.1)
        assert len(handler1_called) > 0
        assert len(handler2_called) > 0

    def test_event_bus_handler_exception(self, event_bus):
        """测试处理器异常处理"""
        def failing_handler(event):
            raise ValueError("Test error")
        
        event_bus.subscribe(EventType.INFO, failing_handler)
        event_data = {"message": "test"}
        # 应该不会抛出异常，而是被捕获
        event_bus.publish(EventType.INFO, event_data)
        time.sleep(0.1)
        # 测试通过如果没有异常抛出

    def test_event_bus_filtering(self, event_bus):
        """测试事件过滤"""
        handler_called = []
        
        def handler(event):
            handler_called.append(event)
        
        def filter_func(event):
            return event.get("priority", 0) > 1
        
        event_bus.add_event_filter(filter_func)
        event_bus.subscribe(EventType.INFO, handler)
        
        # 低优先级事件应该被过滤
        event_bus.publish(EventType.INFO, {"priority": 0})
        time.sleep(0.1)
        # 高优先级事件应该通过
        event_bus.publish(EventType.INFO, {"priority": 2})
        time.sleep(0.1)

