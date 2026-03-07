"""
测试事件存储器

覆盖 event_storage.py 中的所有类和功能
"""

import pytest
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.infrastructure.resource.core.event_storage import EventStorage


class TestEventStorage:
    """EventStorage 类测试"""

    def test_initialization(self):
        """测试初始化"""
        storage = EventStorage()

        assert hasattr(storage, 'max_events')
        assert hasattr(storage, 'retention_period')
        assert hasattr(storage, 'logger')
        assert hasattr(storage, '_events')
        assert hasattr(storage, '_lock')

        assert storage.max_events == 1000
        assert storage.retention_period == timedelta(hours=24)
        assert isinstance(storage._events, type(storage._events))  # deque
        assert len(storage._events) == 0

    def test_initialization_with_custom_params(self):
        """测试自定义参数初始化"""
        custom_max = 500
        custom_hours = 48

        storage = EventStorage(max_events=custom_max, retention_hours=custom_hours)

        assert storage.max_events == custom_max
        assert storage.retention_period == timedelta(hours=custom_hours)

    def test_store_event(self):
        """测试存储事件"""
        storage = EventStorage()
        event_type = "test_event"
        event_data = {"key": "value"}

        storage.store_event(event_type, event_data)

        assert len(storage._events) == 1
        event = storage._events[0]
        assert event['type'] == event_type
        assert event['data'] == event_data
        assert 'timestamp' in event
        assert 'id' in event
        assert event['id'].startswith(f"{event_type}_")

    def test_store_event_with_custom_timestamp(self):
        """测试存储带有自定义时间戳的事件"""
        storage = EventStorage()
        event_type = "custom_time_event"
        event_data = {"custom": "data"}
        custom_time = datetime(2023, 1, 1, 12, 0, 0)

        storage.store_event(event_type, event_data, custom_time)

        assert len(storage._events) == 1
        event = storage._events[0]
        assert event['timestamp'] == custom_time
        assert event['id'] == f"{event_type}_{custom_time.timestamp()}"

    def test_get_events_empty(self):
        """测试获取空事件列表"""
        storage = EventStorage()

        events = storage.get_events()

        assert events == []

    def test_get_events_all(self):
        """测试获取所有事件"""
        storage = EventStorage()

        # 存储多个事件
        events_data = [
            ("event1", {"data": 1}),
            ("event2", {"data": 2}),
            ("event1", {"data": 3}),
        ]

        for event_type, event_data in events_data:
            storage.store_event(event_type, event_data)

        events = storage.get_events()

        assert len(events) == 3
        assert events[0]['type'] == "event1"
        assert events[1]['type'] == "event2"
        assert events[2]['type'] == "event1"

    def test_get_events_filtered_by_type(self):
        """测试按类型过滤获取事件"""
        storage = EventStorage()

        # 存储不同类型的事件
        storage.store_event("type_a", {"value": 1})
        storage.store_event("type_b", {"value": 2})
        storage.store_event("type_a", {"value": 3})
        storage.store_event("type_c", {"value": 4})

        # 获取type_a的事件
        events_a = storage.get_events(event_type="type_a")
        assert len(events_a) == 2
        assert all(e['type'] == "type_a" for e in events_a)

        # 获取type_b的事件
        events_b = storage.get_events(event_type="type_b")
        assert len(events_b) == 1
        assert events_b[0]['type'] == "type_b"

        # 获取不存在类型的事件
        events_none = storage.get_events(event_type="nonexistent")
        assert events_none == []

    def test_get_events_filtered_by_time(self):
        """测试按时间过滤获取事件"""
        storage = EventStorage()

        base_time = datetime.now()

        # 存储不同时间的事件
        storage.store_event("event", {"id": 1}, base_time - timedelta(hours=3))
        storage.store_event("event", {"id": 2}, base_time - timedelta(hours=1))
        storage.store_event("event", {"id": 3}, base_time + timedelta(hours=1))

        # 获取最近2小时的事件
        recent_events = storage.get_events(since=base_time - timedelta(hours=2))

        assert len(recent_events) == 2
        assert recent_events[0]['data']['id'] == 2
        assert recent_events[1]['data']['id'] == 3

    def test_get_events_with_limit(self):
        """测试限制数量获取事件"""
        storage = EventStorage()

        # 存储多个事件
        for i in range(10):
            storage.store_event("event", {"id": i})

        # 获取最新的5个事件
        limited_events = storage.get_events(limit=5)

        assert len(limited_events) == 5
        # 应该是最新的5个（ID从5到9）
        assert limited_events[0]['data']['id'] == 5
        assert limited_events[4]['data']['id'] == 9

    def test_get_events_combined_filters(self):
        """测试组合过滤条件"""
        storage = EventStorage()

        base_time = datetime.now()

        # 存储不同类型和时间的事件
        storage.store_event("type_a", {"id": 1}, base_time - timedelta(hours=3))
        storage.store_event("type_b", {"id": 2}, base_time - timedelta(hours=2))
        storage.store_event("type_a", {"id": 3}, base_time - timedelta(hours=1))
        storage.store_event("type_a", {"id": 4}, base_time)

        # 获取type_a类型且在最近2小时内的事件，最多2个
        filtered_events = storage.get_events(
            event_type="type_a",
            since=base_time - timedelta(hours=2),
            limit=2
        )

        assert len(filtered_events) == 2
        assert all(e['type'] == "type_a" for e in filtered_events)
        assert filtered_events[0]['data']['id'] == 3
        assert filtered_events[1]['data']['id'] == 4

    def test_get_event_count_empty(self):
        """测试获取空存储的事件数量"""
        storage = EventStorage()

        assert storage.get_event_count() == 0
        assert storage.get_event_count("any_type") == 0

    def test_get_event_count_all(self):
        """测试获取所有事件数量"""
        storage = EventStorage()

        # 存储事件
        for i in range(5):
            storage.store_event("event", {"id": i})

        assert storage.get_event_count() == 5

    def test_get_event_count_by_type(self):
        """测试按类型获取事件数量"""
        storage = EventStorage()

        # 存储不同类型的事件
        storage.store_event("type_a", {"id": 1})
        storage.store_event("type_b", {"id": 2})
        storage.store_event("type_a", {"id": 3})
        storage.store_event("type_c", {"id": 4})

        assert storage.get_event_count("type_a") == 2
        assert storage.get_event_count("type_b") == 1
        assert storage.get_event_count("type_c") == 1
        assert storage.get_event_count("nonexistent") == 0

    def test_clear_events_all(self):
        """测试清除所有事件"""
        storage = EventStorage()

        # 存储一些事件
        for i in range(3):
            storage.store_event("event", {"id": i})

        assert storage.get_event_count() == 3

        # 清除所有事件
        storage.clear_events()

        assert storage.get_event_count() == 0

    def test_clear_events_by_type(self):
        """测试按类型清除事件"""
        storage = EventStorage()

        # 存储不同类型的事件
        storage.store_event("keep", {"id": 1})
        storage.store_event("remove", {"id": 2})
        storage.store_event("remove", {"id": 3})
        storage.store_event("keep", {"id": 4})

        assert storage.get_event_count() == 4
        assert storage.get_event_count("remove") == 2
        assert storage.get_event_count("keep") == 2

        # 清除remove类型的事件
        storage.clear_events("remove")

        assert storage.get_event_count() == 2
        assert storage.get_event_count("remove") == 0
        assert storage.get_event_count("keep") == 2

    def test_clear_events_nonexistent_type(self):
        """测试清除不存在类型的事件"""
        storage = EventStorage()

        storage.store_event("existing", {"id": 1})

        # 清除不存在的类型
        storage.clear_events("nonexistent")

        # 现有事件应该保持不变
        assert storage.get_event_count() == 1

    def test_cleanup_expired_events(self):
        """测试清理过期事件"""
        storage = EventStorage(retention_hours=1)  # 1小时保留期

        now = datetime.now()

        # 存储过期和未过期的事件
        storage.store_event("expired", {"id": 1}, now - timedelta(hours=2))  # 2小时前，过期
        storage.store_event("expired", {"id": 2}, now - timedelta(hours=3))  # 3小时前，过期
        storage.store_event("fresh", {"id": 3}, now - timedelta(minutes=30))  # 30分钟前，新鲜

        assert storage.get_event_count() == 3

        # 手动触发清理（正常情况下在get_events时自动调用）
        storage._cleanup_expired_events()

        # 应该只保留新鲜的事件
        assert storage.get_event_count() == 1
        remaining_events = storage.get_events()
        assert len(remaining_events) == 1
        assert remaining_events[0]['data']['id'] == 3

    def test_cleanup_expired_events_no_expired(self):
        """测试清理无过期事件的情况"""
        storage = EventStorage(retention_hours=24)

        now = datetime.now()

        # 存储新鲜的事件
        storage.store_event("fresh1", {"id": 1}, now - timedelta(hours=1))
        storage.store_event("fresh2", {"id": 2}, now - timedelta(minutes=30))

        initial_count = storage.get_event_count()

        # 清理
        storage._cleanup_expired_events()

        # 事件数量应该不变
        assert storage.get_event_count() == initial_count

    def test_cleanup_expired_events_empty_storage(self):
        """测试清理空存储器"""
        storage = EventStorage()

        # 空存储器清理应该没有问题
        storage._cleanup_expired_events()

        assert storage.get_event_count() == 0

    def test_max_events_limit(self):
        """测试最大事件数量限制"""
        max_events = 5
        storage = EventStorage(max_events=max_events)

        # 存储超过限制的事件
        for i in range(max_events + 3):
            storage.store_event("event", {"id": i})

        # 应该只保留最新的max_events个事件
        assert storage.get_event_count() == max_events

        # 验证保留的是最新的事件
        events = storage.get_events()
        assert len(events) == max_events
        # 最早的事件应该是ID为 3,4,5,6,7的（总共8个事件，保留最后5个）
        expected_ids = [3, 4, 5, 6, 7]
        actual_ids = [e['data']['id'] for e in events]
        assert actual_ids == expected_ids

    def test_get_storage_stats_empty(self):
        """测试获取空存储的统计信息"""
        storage = EventStorage(max_events=100, retention_hours=48)

        stats = storage.get_storage_stats()

        assert stats['total_events'] == 0
        assert stats['max_capacity'] == 100
        assert stats['retention_hours'] == 48
        assert stats['oldest_event_age_hours'] == 0  # 空存储的年龄为0
        assert stats['utilization_percent'] == 0.0

    def test_get_storage_stats_with_events(self):
        """测试获取有事件的存储统计信息"""
        storage = EventStorage(max_events=100, retention_hours=24)

        now = datetime.now()

        # 存储一些事件
        storage.store_event("event1", {"id": 1}, now - timedelta(hours=2))
        storage.store_event("event2", {"id": 2}, now - timedelta(hours=1))

        stats = storage.get_storage_stats()

        assert stats['total_events'] == 2
        assert stats['max_capacity'] == 100
        assert stats['retention_hours'] == 24
        assert stats['oldest_event_age_hours'] == pytest.approx(2.0, abs=0.1)  # 大约2小时
        assert stats['utilization_percent'] == 2.0

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        storage = EventStorage(max_events=1000)

        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                # 每个线程存储多个事件
                for i in range(10):
                    event_id = f"thread_{thread_id}_event_{i}"
                    storage.store_event("test_event", {"id": event_id, "thread": thread_id})

                # 查询操作
                count = storage.get_event_count()
                results.append(("store", thread_id, count))

                # 获取事件
                events = storage.get_events("test_event")
                results.append(("retrieve", thread_id, len(events)))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证存储的事件总数正确
        total_expected = num_threads * 10
        assert storage.get_event_count() == total_expected

        # 验证所有事件都被正确存储
        all_events = storage.get_events("test_event")
        assert len(all_events) == total_expected

    def test_event_storage_with_complex_data(self):
        """测试存储复杂数据的事件"""
        storage = EventStorage()

        complex_data = {
            "nested": {
                "array": [1, 2, {"deep": "value"}],
                "dict": {"key": "value"},
                "number": 42.5,
                "boolean": True,
                "null": None
            },
            "metadata": {
                "source": "test_system",
                "priority": "high",
                "tags": ["important", "urgent"]
            }
        }

        storage.store_event("complex_event", complex_data)

        # 验证存储和检索
        events = storage.get_events("complex_event")
        assert len(events) == 1

        retrieved_data = events[0]['data']
        assert retrieved_data == complex_data

        # 验证嵌套结构
        assert retrieved_data['nested']['array'][2]['deep'] == "value"
        assert retrieved_data['metadata']['tags'][1] == "urgent"

    def test_event_storage_performance(self):
        """测试事件存储性能"""
        import time
        storage = EventStorage(max_events=10000)

        # 测试大量事件存储性能
        start_time = time.time()

        num_events = 1000
        for i in range(num_events):
            storage.store_event("perf_test", {"id": i, "data": "x" * 100})  # 较大数据

        storage_time = time.time() - start_time

        # 验证所有事件都被存储
        assert storage.get_event_count() == num_events

        # 测试查询性能
        start_time = time.time()
        events = storage.get_events("perf_test")
        query_time = time.time() - start_time

        assert len(events) == num_events

        # 性能断言（根据实际情况调整）
        assert storage_time < 2.0  # 存储1000个事件应该在2秒内完成
        assert query_time < 0.5    # 查询1000个事件应该在0.5秒内完成

    def test_event_storage_memory_management(self):
        """测试事件存储内存管理"""
        import gc
        storage = EventStorage(max_events=100)

        # 存储大量事件
        large_events = []
        for i in range(200):  # 超过max_events限制
            event_data = {"id": i, "large_data": "x" * 1000}  # 1KB数据
            large_events.append(event_data)
            storage.store_event("memory_test", event_data)

        # 验证只保留最新的max_events个事件
        assert storage.get_event_count() == 100

        # 强制垃圾回收
        del large_events
        gc.collect()

        # 验证存储器仍然工作正常
        stats = storage.get_storage_stats()
        assert stats['total_events'] == 100
        assert stats['utilization_percent'] == 100.0

    def test_event_storage_edge_cases(self):
        """测试事件存储边界情况"""
        storage = EventStorage()

        # 测试空事件类型
        storage.store_event("", {"empty_type": True})
        events = storage.get_events("")
        assert len(events) == 1

        # 测试None数据
        storage.store_event("none_data", None)
        events = storage.get_events("none_data")
        assert len(events) == 1
        assert events[0]['data'] is None

        # 测试空数据
        storage.store_event("empty_data", {})
        events = storage.get_events("empty_data")
        assert len(events) == 1
        assert events[0]['data'] == {}

        # 测试特殊字符
        special_type = "特殊字符@#$%^&*()"
        storage.store_event(special_type, {"special": True})
        events = storage.get_events(special_type)
        assert len(events) == 1

    def test_event_storage_timestamp_precision(self):
        """测试时间戳精度"""
        import time
        storage = EventStorage()

        # 存储多个事件，间隔很短
        timestamps = []
        for i in range(5):
            ts = datetime.now()
            timestamps.append(ts)
            storage.store_event("timing_test", {"id": i}, ts)
            time.sleep(0.001)  # 1毫秒间隔

        events = storage.get_events("timing_test")

        # 验证时间戳顺序正确
        for i in range(4):
            assert events[i]['timestamp'] <= events[i+1]['timestamp']

        # 验证时间戳ID唯一性
        ids = [e['id'] for e in events]
        assert len(set(ids)) == len(ids)  # 所有ID都唯一

    def test_event_storage_concurrent_cleanup(self):
        """测试并发清理操作"""
        storage = EventStorage(retention_hours=0)  # 立即过期

        # 存储一些事件
        for i in range(10):
            storage.store_event("cleanup_test", {"id": i})

        # 并发执行清理和查询
        import threading
        results = []

        def cleanup_worker():
            storage._cleanup_expired_events()
            results.append("cleanup_done")

        def query_worker():
            events = storage.get_events("cleanup_test")
            results.append(("query_result", len(events)))

        threads = [
            threading.Thread(target=cleanup_worker),
            threading.Thread(target=query_worker),
            threading.Thread(target=query_worker),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证没有崩溃，所有操作都完成
        assert len(results) == 3
        assert "cleanup_done" in results

        # 查询结果应该是一致的
        query_results = [r for r in results if isinstance(r, tuple)]
        # 在并发环境下，结果可能不同，但不应该崩溃
