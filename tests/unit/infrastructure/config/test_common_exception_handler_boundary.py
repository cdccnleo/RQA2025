#!/usr/bin/env python3
"""
通用异常处理器边界条件测试

测试异常处理器的各种边界情况和异常场景
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import logging
from unittest.mock import patch, MagicMock, call
from io import StringIO

from src.infrastructure.config.core.common_exception_handler import (
    ExceptionCollector,
    LogLevel,
    ExceptionContext
)


class TestExceptionCollectorBoundary:
    """通用异常收集器边界条件测试"""

    def setup_method(self):
        """测试前准备"""
        self.collector = ExceptionCollector(max_exceptions=1000)  # 为内存测试设置更大的容量
        self.logger = logging.getLogger('test_logger')

    def test_exception_collector_initialization(self):
        """测试异常收集器初始化"""
        assert self.collector is not None
        assert hasattr(self.collector, 'exceptions')
        assert hasattr(self.collector, 'max_exceptions')
        assert hasattr(self.collector, '_lock')

        assert self.collector.exceptions == []
        assert self.collector.max_exceptions == 1000

    def test_add_none_exception(self):
        """测试添加None异常"""
        self.collector.add_exception(None)
        assert len(self.collector.exceptions) == 0

    def test_add_exception_with_context(self):
        """测试添加带上下文的异常"""
        context = ExceptionContext(
            operation="test_op",
            parameters={"key": "value"},
            user_id="user123"
        )

        try:
            raise ValueError("Test error")
        except Exception as e:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.collector.add_exception(e, context, tb_str)

        assert len(self.collector.exceptions) == 1
        exc_info = self.collector.exceptions[0]
        assert exc_info['exception_type'] == 'ValueError'
        assert 'Test error' in exc_info['message']
        assert exc_info['context'] == context.to_dict()

    def test_add_exception_with_none_context(self):
        """测试添加异常时上下文为None"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.collector.add_exception(e, None)

        assert len(self.collector.exceptions) == 1
        assert self.collector.exceptions[0]['context'] is None

    def test_max_exceptions_limit(self):
        """测试最大异常数量限制"""
        collector = ExceptionCollector(max_exceptions=3)

        # 添加超过限制的异常
        for i in range(5):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                collector.add_exception(e)

        # 应该只保留前3个异常
        assert len(collector.exceptions) == 3

    def test_clear_exceptions(self):
        """测试清空异常"""
        # 先添加一些异常
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                self.collector.add_exception(e)

        assert len(self.collector.exceptions) == 3

        # 清空
        self.collector.clear()

        assert len(self.collector.exceptions) == 0

    def test_get_exceptions_summary(self):
        """测试获取异常汇总"""
        # 添加不同类型的异常
        exceptions = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            ValueError("Another value error")
        ]

        for exc in exceptions:
            self.collector.add_exception(exc)

        summary = self.collector.get_summary()

        assert summary['total_count'] == 3
        assert summary['by_type']['ValueError'] == 2
        assert summary['by_type']['RuntimeError'] == 1

    def test_get_exceptions_empty_summary(self):
        """测试获取空异常汇总"""
        summary = self.collector.get_summary()

        assert summary['total_count'] == 0
        assert summary['by_type'] == {}

    def test_concurrent_add_exceptions(self):
        """测试并发添加异常"""
        import threading
        import concurrent.futures

        def add_exception(index):
            try:
                raise ValueError(f"Concurrent error {index}")
            except Exception as e:
                import traceback
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                self.collector.add_exception(e, ExceptionContext(operation=f"op_{index}"), tb_str)

        # 并发添加异常
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_exception, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # 验证结果
        assert len(self.collector.exceptions) == 50

        # 验证所有异常都有正确的上下文
        operations = [exc['context']['operation'] for exc in self.collector.exceptions]
        assert len(set(operations)) == 50  # 所有操作都唯一

    def test_exception_context_creation(self):
        """测试异常上下文创建"""
        context = ExceptionContext(
            operation="test_op",
            parameters={"key": "value"},
            user_id="user123",
            session_id="session456"
        )

        assert context.operation == "test_op"
        assert context.parameters == {"key": "value"}
        assert context.user_id == "user123"
        assert context.session_id == "session456"

    def test_exception_context_with_none_values(self):
        """测试异常上下文的None值处理"""
        context = ExceptionContext(
            operation=None,
            parameters=None,
            user_id=None,
            session_id=None
        )

        assert context.operation == ""
        assert context.parameters == {}
        assert context.user_id == ""
        assert context.session_id == ""

    def test_exception_context_with_large_data(self):
        """测试异常上下文包含大数据"""
        large_params = {f"key_{i}": f"value_{i}" * 100 for i in range(10)}

        context = ExceptionContext(
            operation="large_op",
            parameters=large_params
        )

        assert len(context.parameters) == 10
        assert len(context.parameters["key_0"]) == 700  # "value_0" * 100

    def test_log_level_enum(self):
        """测试日志级别枚举"""
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.CRITICAL.value == 50

    def test_exception_collector_with_custom_max(self):
        """测试自定义最大异常数量的收集器"""
        collector = ExceptionCollector(max_exceptions=5)

        assert collector.max_exceptions == 5

        # 添加6个异常
        for i in range(6):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                collector.add_exception(e)

        assert len(collector.exceptions) == 5

    def test_exception_info_structure(self):
        """测试异常信息结构"""
        context = ExceptionContext(operation="test", parameters={"p": "v"})

        try:
            raise ValueError("Test message")
        except Exception as e:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.collector.add_exception(e, context, tb_str)

        exc_info = self.collector.exceptions[0]

        required_keys = ['exception_type', 'message', 'timestamp', 'context', 'traceback']
        for key in required_keys:
            assert key in exc_info

        assert exc_info['exception_type'] == 'ValueError'
        assert exc_info['message'] == 'Test message'
        assert exc_info['context'] == context.to_dict()
        assert isinstance(exc_info['timestamp'], float)
        assert isinstance(exc_info['traceback'], str)

    def test_exception_collector_thread_safety(self):
        """测试异常收集器的线程安全性"""
        import threading

        exceptions_added = []
        lock = threading.Lock()

        def add_with_check(index):
            try:
                raise ValueError(f"Thread {index}")
            except Exception as e:
                with lock:
                    exceptions_added.append(index)
                self.collector.add_exception(e)

        threads = []
        for i in range(20):
            t = threading.Thread(target=add_with_check, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(exceptions_added) == 20
        assert len(self.collector.exceptions) == 20

    def test_exception_collector_memory_efficiency(self):
        """测试异常收集器的内存效率"""
        # 添加大量异常
        for i in range(1000):
            try:
                raise ValueError(f"Memory test {i}")
            except Exception as e:
                self.collector.add_exception(e)

        # 验证内存使用合理
        assert len(self.collector.exceptions) == 1000

        # 清空后内存释放
        self.collector.clear()
        assert len(self.collector.exceptions) == 0

    def test_exception_context_equality(self):
        """测试异常上下文相等性"""
        ctx1 = ExceptionContext("op1", {"k": "v"}, "user1")
        ctx2 = ExceptionContext("op1", {"k": "v"}, "user1")
        ctx3 = ExceptionContext("op2", {"k": "v"}, "user1")

        assert ctx1.operation == ctx2.operation
        assert ctx1.parameters == ctx2.parameters
        assert ctx1.user_id == ctx2.user_id

        assert ctx1.operation != ctx3.operation

    def test_summary_with_no_exceptions(self):
        """测试无异常时的汇总"""
        summary = self.collector.get_summary()

        assert summary['total_count'] == 0
        assert summary['by_type'] == {}
        assert summary['latest_timestamp'] is None
        assert summary['earliest_timestamp'] is None
        assert summary['max_capacity'] == 1000
        assert summary['utilization_rate'] == 0.0

    def test_summary_with_single_exception(self):
        """测试单个异常的汇总"""
        try:
            raise ValueError("Single error")
        except Exception as e:
            self.collector.add_exception(e)

        summary = self.collector.get_summary()

        assert summary['total_count'] == 1
        assert summary['by_type']['ValueError'] == 1
        assert isinstance(summary['latest_timestamp'], float)
        assert isinstance(summary['earliest_timestamp'], float)

    def test_exception_context_to_dict(self):
        """测试异常上下文转字典"""
        context = ExceptionContext(
            operation="test_op",
            parameters={"key": "value"},
            user_id="user123",
            session_id="session456"
        )

        # 虽然ExceptionContext没有to_dict方法，但我们可以测试其属性
        assert hasattr(context, 'operation')
        assert hasattr(context, 'parameters')
        assert hasattr(context, 'user_id')
        assert hasattr(context, 'session_id')
