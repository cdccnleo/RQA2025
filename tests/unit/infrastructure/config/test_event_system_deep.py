#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Config事件系统深度测试"""

import pytest
import os


def test_config_events_constants():
    """测试ConfigEvents常量定义"""
    from src.infrastructure.config.services.event import ConfigEvents
    
    assert ConfigEvents.CONFIG_CHANGED == "config_changed"
    assert ConfigEvents.CONFIG_RELOADED == "config_reloaded"
    assert ConfigEvents.CONFIG_ERROR == "config_error"


def test_event_class():
    """测试Event类"""
    from src.infrastructure.config.services.event import Event
    
    event = Event("test_event", {"key": "value"})
    assert event.name == "test_event"
    assert event.data == {"key": "value"}


def test_event_with_none_data():
    """测试Event无数据"""
    from src.infrastructure.config.services.event import Event
    
    event = Event("test_event")
    assert event.name == "test_event"
    assert event.data is None


def test_event_system_singleton():
    """测试EventSystem单例模式"""
    from src.infrastructure.config.services.event import EventSystem
    
    system1 = EventSystem.get_default()
    system2 = EventSystem.get_default()
    
    assert system1 is not None
    assert system2 is not None
    assert system1 is system2  # 必须是同一个实例


def test_event_system_subscribe():
    """测试事件订阅"""
    from src.infrastructure.config.services.event import EventSystem
    
    system = EventSystem()
    called = []
    
    def callback(data):
        called.append(data)
    
    sub_id = system.subscribe("test_event", callback)
    assert isinstance(sub_id, str)
    assert len(sub_id) > 0


def test_event_system_publish_and_receive():
    """测试事件发布和接收"""
    from src.infrastructure.config.services.event import EventSystem
    
    system = EventSystem()
    received_data = []
    
    def callback(data):
        received_data.append(data)
    
    # 订阅
    sub_id = system.subscribe("test_event", callback)
    
    # 发布
    system.publish("test_event", {"message": "hello"})
    
    # 验证接收
    assert len(received_data) == 1
    assert received_data[0] == {"message": "hello"}


def test_event_system_unsubscribe_by_id():
    """测试通过ID取消订阅"""
    from src.infrastructure.config.services.event import EventSystem
    
    system = EventSystem()
    called = []
    
    def callback(data):
        called.append(data)
    
    # 订阅
    sub_id = system.subscribe("test_event", callback)
    
    # 发布一次
    system.publish("test_event", "data1")
    assert len(called) == 1
    
    # 取消订阅
    system.unsubscribe("test_event", sub_id)
    
    # 再次发布
    system.publish("test_event", "data2")
    
    # 不应再接收
    assert len(called) == 1  # 仍然是1


def test_event_system_unsubscribe_by_callback():
    """测试通过回调函数取消订阅"""
    from src.infrastructure.config.services.event import EventSystem
    
    system = EventSystem()
    called = []
    
    def callback(data):
        called.append(data)
    
    # 订阅
    system.subscribe("test_event", callback)
    
    # 发布一次
    system.publish("test_event", "data1")
    assert len(called) == 1
    
    # 通过callback取消订阅
    system.unsubscribe("test_event", callback)
    
    # 再次发布
    system.publish("test_event", "data2")
    
    # 不应再接收
    assert len(called) == 1


def test_event_system_multiple_subscribers():
    """测试多个订阅者"""
    from src.infrastructure.config.services.event import EventSystem
    
    system = EventSystem()
    results = {'callback1': [], 'callback2': []}
    
    def callback1(data):
        results['callback1'].append(data)
    
    def callback2(data):
        results['callback2'].append(data)
    
    # 订阅
    system.subscribe("test_event", callback1)
    system.subscribe("test_event", callback2)
    
    # 发布
    system.publish("test_event", "test_data")
    
    # 两个回调都应收到
    assert len(results['callback1']) == 1
    assert len(results['callback2']) == 1


def test_event_system_get_events():
    """测试获取事件记录（测试模式）"""
    from src.infrastructure.config.services.event import EventSystem
    
    # 设置测试环境变量
    os.environ['TESTING'] = 'true'
    
    try:
        system = EventSystem()
        
        # 发布事件
        system.publish("test_event", "data1")
        system.publish("test_event", "data2")
        
        # 获取事件
        events = system.get_events("test_event")
        assert len(events) == 2
        assert "data1" in events
        assert "data2" in events
    finally:
        os.environ.pop('TESTING', None)


def test_event_system_clear_events():
    """测试清除事件记录"""
    from src.infrastructure.config.services.event import EventSystem
    
    os.environ['TESTING'] = 'true'
    
    try:
        system = EventSystem()
        
        # 发布事件
        system.publish("event1", "data1")
        system.publish("event2", "data2")
        
        # 清除特定事件
        system.clear_events("event1")
        events1 = system.get_events("event1")
        events2 = system.get_events("event2")
        
        assert len(events1) == 0
        assert len(events2) == 1
        
        # 清除所有事件
        system.clear_events()
        events2_after = system.get_events("event2")
        assert len(events2_after) == 0
    finally:
        os.environ.pop('TESTING', None)


def test_event_bus():
    """测试EventBus"""
    from src.infrastructure.config.services.event import EventBus
    
    bus = EventBus()
    assert bus is not None
    assert hasattr(bus, '_event_system')


def test_event_bus_subscribe_publish():
    """测试EventBus订阅和发布"""
    from src.infrastructure.config.services.event import EventBus
    
    bus = EventBus()
    received = []
    
    def callback(data):
        received.append(data)
    
    # 订阅
    sub_id = bus.subscribe("test", callback)
    
    # 发布
    bus.publish("test", "message")
    
    # 验证
    assert len(received) == 1
    assert received[0] == "message"


