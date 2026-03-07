#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 高级日志类型

测试logging/advanced/types.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock

from src.infrastructure.logging.advanced.types import (
    LogPriority, LogCompression, LogEntry, LogEntryPool
)


class TestLogPriority:
    """测试日志优先级枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LogPriority.LOW.value == 1
        assert LogPriority.NORMAL.value == 2
        assert LogPriority.HIGH.value == 3
        assert LogPriority.CRITICAL.value == 4

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(LogPriority) == 4
        assert LogPriority.LOW in LogPriority
        assert LogPriority.NORMAL in LogPriority
        assert LogPriority.HIGH in LogPriority
        assert LogPriority.CRITICAL in LogPriority

    def test_enum_ordering(self):
        """测试枚举排序"""
        assert LogPriority.LOW.value < LogPriority.NORMAL.value < LogPriority.HIGH.value < LogPriority.CRITICAL.value
        assert LogPriority.CRITICAL.value > LogPriority.HIGH.value > LogPriority.NORMAL.value > LogPriority.LOW.value


class TestLogCompression:
    """测试日志压缩枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LogCompression.NONE.value == "none"
        assert LogCompression.GZIP.value == "gzip"
        assert LogCompression.LZ4.value == "lz4"

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(LogCompression) == 3
        assert LogCompression.NONE in LogCompression
        assert LogCompression.GZIP in LogCompression
        assert LogCompression.LZ4 in LogCompression


class TestLogEntry:
    """测试日志条目数据类"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        timestamp = time.time()
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test_logger",
            message="Test message"
        )

        assert entry.timestamp == timestamp
        assert entry.level == "INFO"
        assert entry.logger_name == "test_logger"
        assert entry.message == "Test message"
        assert entry.priority == LogPriority.NORMAL
        assert entry.metadata == {}
        assert entry.correlation_id is None
        assert entry.thread_id is None
        assert entry.process_id is None

    def test_initialization_with_all_params(self):
        """测试完整参数初始化"""
        timestamp = time.time()
        metadata = {"key": "value", "number": 42}
        correlation_id = "corr-123"
        thread_id = 1234
        process_id = 5678

        entry = LogEntry(
            timestamp=timestamp,
            level="ERROR",
            logger_name="error_logger",
            message="Error occurred",
            priority=LogPriority.HIGH,
            metadata=metadata,
            correlation_id=correlation_id,
            thread_id=thread_id,
            process_id=process_id
        )

        assert entry.timestamp == timestamp
        assert entry.level == "ERROR"
        assert entry.logger_name == "error_logger"
        assert entry.message == "Error occurred"
        assert entry.priority == LogPriority.HIGH
        assert entry.metadata == metadata
        assert entry.correlation_id == correlation_id
        assert entry.thread_id == thread_id
        assert entry.process_id == process_id

    def test_initialization_with_different_priorities(self):
        """测试不同优先级初始化"""
        priorities = [LogPriority.LOW, LogPriority.NORMAL, LogPriority.HIGH, LogPriority.CRITICAL]

        for priority in priorities:
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name="test",
                message="Test",
                priority=priority
            )
            assert entry.priority == priority

    def test_dataclass_immutability(self):
        """测试数据类不可变性"""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Original message"
        )

        # 尝试修改（这应该会失败或创建新对象）
        # LogEntry是frozen=False的，所以可以修改
        entry.message = "Modified message"
        assert entry.message == "Modified message"

    def test_equality(self):
        """测试相等性"""
        timestamp = time.time()

        entry1 = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test",
            message="Test message",
            priority=LogPriority.NORMAL
        )

        entry2 = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test",
            message="Test message",
            priority=LogPriority.NORMAL
        )

        entry3 = LogEntry(
            timestamp=timestamp,
            level="ERROR",  # 不同级别
            logger_name="test",
            message="Test message",
            priority=LogPriority.NORMAL
        )

        assert entry1 == entry2
        assert entry1 != entry3

    def test_hash_consistency(self):
        """测试哈希一致性"""
        timestamp = time.time()

        entry1 = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test",
            message="Test message"
        )

        entry2 = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test",
            message="Test message"
        )

        # 相同内容的条目应该相等
        assert entry1 == entry2

        # 由于包含可变对象，不支持哈希，但应该有合理的字符串表示
        str1 = str(entry1)
        str2 = str(entry2)
        assert str1 == str2

    def test_metadata_operations(self):
        """测试元数据操作"""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Test message",
            metadata={"initial": "value"}
        )

        # 添加元数据
        entry.metadata["added"] = "new_value"
        assert entry.metadata["added"] == "new_value"

        # 修改现有元数据
        entry.metadata["initial"] = "modified"
        assert entry.metadata["initial"] == "modified"

        # 删除元数据
        del entry.metadata["initial"]
        assert "initial" not in entry.metadata

    def test_correlation_id_handling(self):
        """测试关联ID处理"""
        # 无关联ID
        entry1 = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="No correlation"
        )
        assert entry1.correlation_id is None

        # 有关联ID
        entry2 = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="With correlation",
            correlation_id="corr-12345"
        )
        assert entry2.correlation_id == "corr-12345"

    def test_thread_process_id_handling(self):
        """测试线程和进程ID处理"""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Thread and process test",
            thread_id=1234,
            process_id=5678
        )

        assert entry.thread_id == 1234
        assert entry.process_id == 5678

        # 测试实际的线程和进程ID
        import threading
        import os

        real_entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Real IDs",
            thread_id=threading.get_ident(),
            process_id=os.getpid()
        )

        assert real_entry.thread_id == threading.get_ident()
        assert real_entry.process_id == os.getpid()

    def test_serialization_friendly(self):
        """测试序列化友好性"""
        import json

        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Serialization test",
            metadata={"key": "value", "number": 42},
            correlation_id="corr-123"
        )

        # 转换为字典（用于序列化）
        entry_dict = {
            "timestamp": entry.timestamp,
            "level": entry.level,
            "logger_name": entry.logger_name,
            "message": entry.message,
            "priority": entry.priority.value,
            "metadata": entry.metadata,
            "correlation_id": entry.correlation_id,
            "thread_id": entry.thread_id,
            "process_id": entry.process_id
        }

        # 应该可以JSON序列化
        json_str = json.dumps(entry_dict, default=str)
        assert isinstance(json_str, str)

        # 反序列化
        parsed = json.loads(json_str)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Serialization test"


class TestLogEntryPool:
    """测试日志条目对象池"""

    def test_initialization(self):
        """测试初始化"""
        pool = LogEntryPool()

        assert pool.pool == []
        assert pool.max_size == 1000

    def test_initialization_with_custom_size(self):
        """测试自定义大小初始化"""
        pool = LogEntryPool(max_size=500)

        assert pool.pool == []
        assert pool.max_size == 500

    def test_get_from_empty_pool(self):
        """测试从空池获取"""
        pool = LogEntryPool()

        entry = pool.get()

        # 应该创建一个新的LogEntry对象
        assert isinstance(entry, LogEntry)
        assert entry.timestamp is not None
        assert entry.level == ""
        assert entry.logger_name == ""
        assert entry.message == ""

    def test_put_and_get(self):
        """测试放入和获取"""
        pool = LogEntryPool()

        # 创建一个条目并放入池中
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            logger_name="test",
            message="Test message",
            metadata={"test": True}
        )

        pool.put(entry)

        # 从池中获取
        retrieved = pool.get()

        # 应该得到相同的对象
        assert retrieved is entry
        assert retrieved.level == "INFO"
        assert retrieved.logger_name == "test"
        assert retrieved.message == "Test message"
        assert retrieved.metadata["test"] is True

    def test_pool_size_limit(self):
        """测试池大小限制"""
        pool = LogEntryPool(max_size=3)

        # 放入超过限制的条目
        for i in range(5):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"logger_{i}",
                message=f"Message {i}"
            )
            pool.put(entry)

        # 池大小应该被限制
        assert len(pool.pool) <= pool.max_size

        # 前面的条目应该被丢弃
        assert len(pool.pool) == 3

    def test_pool_size_no_limit(self):
        """测试无大小限制的池"""
        pool = LogEntryPool(max_size=0)  # 0表示无限制

        # 放入很多条目
        for i in range(10):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"logger_{i}",
                message=f"Message {i}"
            )
            pool.put(entry)

        # 应该保留所有条目
        assert len(pool.pool) == 10

    def test_get_multiple_times(self):
        """测试多次获取"""
        pool = LogEntryPool()

        # 放入多个条目
        entries = []
        for i in range(3):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"logger_{i}",
                message=f"Message {i}"
            )
            entries.append(entry)
            pool.put(entry)

        # 多次获取
        retrieved = []
        for _ in range(3):
            retrieved.append(pool.get())

        # 应该按照FILO（后进先出）顺序返回
        assert retrieved[0] is entries[2]  # 最后放入的
        assert retrieved[1] is entries[1]  # 中间的
        assert retrieved[2] is entries[0]  # 最先放入的

    def test_put_none_value(self):
        """测试放入None值"""
        pool = LogEntryPool()

        # 尝试放入None
        pool.put(None)

        # 池应该仍然为空（None不应该被添加）
        assert len(pool.pool) == 0

    def test_get_after_pool_exhausted(self):
        """测试池耗尽后的获取"""
        pool = LogEntryPool()

        # 放入2个条目
        pool.put(LogEntry(timestamp=time.time(), level="INFO", logger_name="test1", message="msg1"))
        pool.put(LogEntry(timestamp=time.time(), level="INFO", logger_name="test2", message="msg2"))

        # 获取2个条目
        entry1 = pool.get()
        entry2 = pool.get()

        assert entry1.logger_name in ["test1", "test2"]
        assert entry2.logger_name in ["test1", "test2"]
        assert entry1 is not entry2

        # 池已空，继续获取应该创建新对象
        entry3 = pool.get()
        assert isinstance(entry3, LogEntry)
        assert entry3.logger_name == ""  # 新对象的默认值

    def test_clear_pool(self):
        """测试清空池"""
        pool = LogEntryPool()

        # 放入一些条目
        for i in range(5):
            pool.put(LogEntry(timestamp=time.time(), level="INFO", logger_name=f"test{i}", message=f"msg{i}"))

        assert len(pool.pool) == 5

        # 清空池
        pool.pool.clear()

        assert len(pool.pool) == 0

        # 之后获取应该创建新对象
        entry = pool.get()
        assert isinstance(entry, LogEntry)

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading

        pool = LogEntryPool(max_size=100)
        results = []
        errors = []

        def pool_operation(thread_id):
            try:
                # 每个线程执行多个放入和获取操作
                for i in range(10):
                    # 放入条目
                    entry = LogEntry(
                        timestamp=time.time(),
                        level="INFO",
                        logger_name=f"thread_{thread_id}",
                        message=f"operation_{i}"
                    )
                    pool.put(entry)

                    # 获取条目
                    retrieved = pool.get()
                    if retrieved:
                        results.append(f"thread_{thread_id}_got_{retrieved.logger_name}")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=pool_operation, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 operations each

    def test_memory_efficiency(self):
        """测试内存效率"""
        pool = LogEntryPool(max_size=10)

        # 创建多个条目并放入池中
        entries = []
        for i in range(15):  # 超过池大小限制
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"memory_test_{i}",
                message=f"Memory efficiency test {i}"
            )
            entries.append(entry)
            pool.put(entry)

        # 池大小应该被限制
        assert len(pool.pool) <= pool.max_size

        # 应该能够正常获取条目
        for _ in range(5):
            retrieved = pool.get()
            assert isinstance(retrieved, LogEntry)

    def test_pool_with_complex_metadata(self):
        """测试带有复杂元数据的池"""
        pool = LogEntryPool()

        # 创建带有复杂元数据的条目
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "boolean": True,
            "number": 42.5
        }

        entry = LogEntry(
            timestamp=time.time(),
            level="DEBUG",
            logger_name="complex_test",
            message="Complex metadata test",
            metadata=complex_metadata,
            correlation_id="complex-123"
        )

        pool.put(entry)
        retrieved = pool.get()

        # 元数据应该保持完整
        assert retrieved.metadata == complex_metadata
        assert retrieved.correlation_id == "complex-123"
        assert retrieved.level == "DEBUG"

    def test_pool_garbage_collection_compatibility(self):
        """测试与垃圾回收的兼容性"""
        import gc

        pool = LogEntryPool()

        # 创建条目但不保持引用
        for i in range(10):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name="gc_test",
                message=f"GC test {i}"
            )
            pool.put(entry)
            # 删除局部引用，但对象仍在池中
            del entry

        # 强制垃圾回收
        gc.collect()

        # 池中的对象应该仍然存在
        assert len(pool.pool) == 10

        # 应该能够获取对象
        retrieved = pool.get()
        assert isinstance(retrieved, LogEntry)
        assert retrieved.logger_name == "gc_test"

    def test_performance_comparison(self):
        """测试性能对比"""
        import time

        pool = LogEntryPool(max_size=1000)

        # 测试池化对象的创建性能
        pool_start = time.time()
        for i in range(1000):
            entry = pool.get()
            # 模拟使用
            entry.timestamp = time.time()
            entry.message = f"Pooled entry {i}"
            pool.put(entry)
        pool_end = time.time()
        pool_duration = pool_end - pool_start

        # 测试直接创建对象的性能
        direct_start = time.time()
        for i in range(1000):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name="direct_test",
                message=f"Direct entry {i}"
            )
            # 模拟使用
            entry.message = f"Modified {i}"
        direct_end = time.time()
        direct_duration = direct_end - direct_start

        # 池化应该至少不比直接创建慢太多（确保direct_duration不为0）
        if direct_duration > 0:
            assert pool_duration < direct_duration * 3
        else:
            # 如果direct_duration为0，确保pool_duration也很小
            assert pool_duration < 0.01

    def test_pool_edge_cases(self):
        """测试池的边界情况"""
        # 测试极小池
        tiny_pool = LogEntryPool(max_size=1)

        # 放入多个条目到小池
        for i in range(3):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"tiny_{i}",
                message=f"Tiny pool {i}"
            )
            tiny_pool.put(entry)

        # 池应该只保留第一个（因为max_size=1，后面的不会被添加）
        assert len(tiny_pool.pool) == 1
        retrieved = tiny_pool.get()
        assert retrieved.logger_name == "tiny_0"  # 第一个放入的

        # 测试极大池
        large_pool = LogEntryPool(max_size=10000)

        # 放入大量条目
        for i in range(100):
            entry = LogEntry(
                timestamp=time.time(),
                level="INFO",
                logger_name=f"large_{i}",
                message=f"Large pool {i}"
            )
            large_pool.put(entry)

        assert len(large_pool.pool) == 100
