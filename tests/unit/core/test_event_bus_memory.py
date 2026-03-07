#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件总线内存管理测试
验证内存泄漏修复和定期清理机制
"""

import sys
import os
import time
import threading
import gc
from pathlib import Path
import pytest
from unittest.mock import Mock

# 添加项目根目录到路径

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.event_bus.bus_components import EventBus
from src.core.event_bus.event_bus import Event, EventType, EventPriority


class MockEventHandler:
    """Mock事件处理器"""

    def __init__(self):
        self.handled_events = []

    def handle_event(self, event_data):
        """处理事件"""
        self.handled_events.append(event_data)
        return True


@pytest.fixture
def event_bus():
    """事件总线fixture"""
    bus = EventBus(max_workers=2, enable_async=False)  # 禁用异步处理，确保事件立即被处理
    yield bus
    # 清理
    bus.shutdown()


def test_event_history_memory_limit(event_bus):
    """测试事件历史内存限制"""
    # 设置订阅者，确保事件被处理
    mock_handler = MockEventHandler()
    event_bus.subscribe(EventType.DATA_RECEIVED, mock_handler)

    # 创建大量事件
    for i in range(150):  # 创建150个事件测试基本功能
        event = Event(
            event_type=EventType.DATA_RECEIVED,
            data={'index': i, 'data': 'A' * 1000},  # 1KB数据
            source=f'test_source_{i}'
        )
        event_bus.publish(event.event_type, event.data, event.source)

    # 验证历史记录数量不超过限制
    history = event_bus.get_event_history()
    assert len(history) <= 10000

    # 验证所有150个事件都被保留（因为没有超过限制）
    assert len(history) == 150
    assert history[-1]['data']['index'] == 149  # 最新的事件


def test_event_history_cleanup_mechanism(event_bus):
    """测试事件历史清理机制"""
    # 创建一些旧事件
    old_time = time.time() - 86400  # 24小时前

    # 模拟一些旧事件
    for i in range(50):
        event = Event(
            event_type=EventType.DATA_STORED,
            data={'index': i},
            source='old_source'
        )
        # 手动设置旧时间戳
        event.timestamp = old_time - i * 3600  # 每小时一个
        event_bus._event_history.append(event)

    original_count = len(event_bus._event_history)

    # 执行清理
    event_bus.clear_history(before_time=time.time() - 43200)  # 12小时前

    # 验证旧事件被清理
    history = event_bus.get_event_history()
    assert len(history) < original_count

    # 验证保留的事件都是较新的
    for event_data in history:
        assert event_data['timestamp'] > time.time() - 43200


def test_event_bus_performance_under_load(event_bus):
    """测试事件总线在高负载下的性能"""
    import psutil
    import os

    # 获取初始内存使用
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 创建大量并发事件
    def create_events():
        for i in range(1000):
            event = Event(
                event_type=EventType.DATA_RECEIVED,
                data={'index': i, 'payload': 'X' * 100},  # 100字节负载
                source='performance_test'
            )
            event_bus.publish(event.event_type, event.data, event.source)

    # 创建多个线程并发发布事件
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=create_events)
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 获取最终内存使用
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 验证内存增长在合理范围内 (不应超过100MB)
    memory_growth = final_memory - initial_memory
    assert memory_growth < 100, f"内存增长过大: {memory_growth:.2f}MB"

    # 验证事件历史没有无限制增长
    history = event_bus.get_event_history()
    assert len(history) <= 10000


def test_event_bus_cleanup_timer(event_bus):
    """测试事件总线清理定时器"""
    # 验证清理定时器已启动
    assert hasattr(event_bus, '_cleanup_timer')
    assert event_bus._cleanup_timer is not None

    # 等待一段时间让清理定时器有机会执行
    time.sleep(1)

    # 验证定时器仍在运行
    assert event_bus._cleanup_timer.is_alive()


def test_event_bus_large_payload_handling(event_bus):
    """测试事件总线处理大负载事件"""
    # 创建大负载事件
    large_payload = 'A' * 1000000  # 1MB数据

    event = Event(
        event_type=EventType.DATA_RECEIVED,
        data={'payload': large_payload},
        source='large_payload_test'
    )

    # 发布事件
    result = event_bus.publish_sync(event)
    assert result is True

    # 验证事件被正确处理
    history = event_bus.get_event_history()
    assert len(history) > 0

    # 验证大负载事件的数据完整性
    latest_event = history[-1]
    assert latest_event['data']['payload'] == large_payload


def test_event_bus_concurrent_access(event_bus):
    """测试事件总线并发访问安全性"""
    results = []
    errors = []

    def concurrent_publisher(thread_id):
        """并发发布者"""
        try:
            for i in range(100):
                event = Event(
                    event_type=EventType.DATA_RECEIVED,
                    data={'thread_id': thread_id, 'index': i},
                    source=f'thread_{thread_id}'
                )
                event_bus.publish(event.event_type, event.data, event.source)
                results.append(f"thread_{thread_id}_event_{i}")
        except Exception as e:
            errors.append(f"thread_{thread_id}_error: {str(e)}")

    # 创建多个并发线程
    threads = []
    for i in range(10):
        thread = threading.Thread(target=concurrent_publisher, args=(i,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 验证没有并发错误
    assert len(errors) == 0, f"并发错误: {errors}"

    # 验证所有事件都被正确处理
    history = event_bus.get_event_history()
    assert len(history) == 1000  # 10线程 * 100事件

    # 验证事件数据完整性
    thread_counts = {}
    for event_data in history:
        thread_id = event_data['data']['thread_id']
        thread_counts[thread_id] = thread_counts.get(thread_id, 0) + 1

    # 验证每个线程的100个事件都被正确记录
    for thread_id in range(10):
        assert thread_counts.get(thread_id, 0) == 100


def test_event_bus_memory_cleanup_integration(event_bus):
    """测试事件总线内存清理的集成效果"""
    # 创建大量事件填充历史记录
    for i in range(5000):
        event = Event(
            event_type=EventType.DATA_STORED,
            data={'index': i, 'data': 'test_data'},
            source='cleanup_test'
        )
        event_bus.publish(event.event_type, event.data, event.source)

    # 验证历史记录已满
    history = event_bus.get_event_history()
    assert len(history) == 5000

    # 手动触发清理（清理1小时前的数据）
    event_bus.clear_history(before_time=time.time() - 3600)

    # 验证清理后的历史记录
    cleaned_history = event_bus.get_event_history()

    # 由于所有事件都是刚刚创建的，应该都被保留
    assert len(cleaned_history) == 5000

    # 现在清理所有数据
    event_bus.clear_history()

    # 验证历史记录被清空
    empty_history = event_bus.get_event_history()
    assert len(empty_history) == 0


def test_event_bus_resource_management(event_bus):
    """测试事件总线资源管理"""
    import weakref

    # 创建弱引用来监控对象生命周期
    handler_refs = []

    # 创建多个处理器
    for i in range(10):
        handler = MockEventHandler()
        handler_ref = weakref.ref(handler)
        handler_refs.append(handler_ref)

        # 订阅事件
        event_bus.subscribe(f'test_event_{i}', handler.handle_event)

    # 强制垃圾回收
    gc.collect()

    # 验证处理器对象仍然存在（被事件总线引用）
    active_handlers = sum(1 for ref in handler_refs if ref() is not None)
    assert active_handlers == 10

    # 发布一些事件
    for i in range(10):
        event = Event(
            event_type=f'test_event_{i}',
            data={'test': True},
            source='resource_test'
        )
        event_bus.publish(event.event_type, event.data, event.source)

    # 验证事件被正确处理
    total_handled = sum(len(ref().handled_events) for ref in handler_refs if ref() is not None)
    assert total_handled == 10  # 每个处理器处理1个事件
