"""
测试错误处理器模块

覆盖 ErrorHandler 类的功能
"""

import pytest
from unittest.mock import Mock
from src.infrastructure.error.error_handler import ErrorHandler


class TestErrorHandler:
    """ErrorHandler 类测试"""

    def test_initialization(self):
        """测试初始化"""
        handler = ErrorHandler()

        assert handler.errors == []
        assert isinstance(handler.errors, list)

    def test_handle_error(self):
        """测试处理错误"""
        handler = ErrorHandler()
        error = ValueError("Test error")

        result = handler.handle_error(error)

        assert result == True
        assert len(handler.errors) == 1
        assert handler.errors[0] == error

    def test_handle_multiple_errors(self):
        """测试处理多个错误"""
        handler = ErrorHandler()

        error1 = ValueError("Error 1")
        error2 = RuntimeError("Error 2")
        error3 = TypeError("Error 3")

        handler.handle_error(error1)
        handler.handle_error(error2)
        handler.handle_error(error3)

        assert len(handler.errors) == 3
        assert handler.errors == [error1, error2, error3]

    def test_get_errors(self):
        """测试获取错误列表"""
        handler = ErrorHandler()

        # 初始状态
        errors = handler.get_errors()
        assert errors == []
        assert errors is handler.errors  # 直接返回原始列表

        # 添加错误后
        error = Exception("Test error")
        handler.handle_error(error)

        errors = handler.get_errors()
        assert len(errors) == 1
        assert errors[0] == error

    def test_get_errors_returns_reference(self):
        """测试get_errors返回引用而不是副本"""
        handler = ErrorHandler()
        error = ValueError("Test")

        handler.handle_error(error)

        errors1 = handler.get_errors()
        errors2 = handler.get_errors()

        # 返回的是同一个对象引用
        assert errors1 is errors2
        assert errors1 is handler.errors

        # 内容相同
        assert errors1 == errors2 == handler.errors

    def test_handle_different_error_types(self):
        """测试处理不同类型的错误"""
        handler = ErrorHandler()

        errors = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
            AttributeError("Attribute error"),
            IndexError("Index error"),
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            ConnectionError("Connection failed"),
            TimeoutError("Timeout")
        ]

        for error in errors:
            result = handler.handle_error(error)
            assert result == True

        assert len(handler.errors) == len(errors)
        assert handler.errors == errors

    def test_handle_custom_exception(self):
        """测试处理自定义异常"""

        class CustomError(Exception):
            """自定义异常"""
            def __init__(self, message, code=None):
                super().__init__(message)
                self.code = code

        handler = ErrorHandler()
        custom_error = CustomError("Custom error occurred", code=500)

        result = handler.handle_error(custom_error)

        assert result == True
        assert len(handler.errors) == 1
        assert handler.errors[0] == custom_error
        assert handler.errors[0].code == 500

    def test_error_accumulation(self):
        """测试错误积累"""
        handler = ErrorHandler()

        # 逐步添加错误
        for i in range(10):
            error = Exception(f"Error {i}")
            handler.handle_error(error)

        assert len(handler.errors) == 10

        # 验证所有错误都被正确存储
        for i in range(10):
            assert str(handler.errors[i]) == f"Error {i}"

    def test_handle_none_error(self):
        """测试处理None错误（虽然不应该发生）"""
        handler = ErrorHandler()

        # 这在实际使用中不应该发生，但测试健壮性
        result = handler.handle_error(None)

        assert result == True
        assert len(handler.errors) == 1
        assert handler.errors[0] is None

    def test_error_handler_isolation(self):
        """测试错误处理器隔离"""
        handler1 = ErrorHandler()
        handler2 = ErrorHandler()

        error1 = ValueError("Handler1 error")
        error2 = RuntimeError("Handler2 error")

        handler1.handle_error(error1)
        handler2.handle_error(error2)

        # 每个处理器应该只包含自己的错误
        assert len(handler1.errors) == 1
        assert handler1.errors[0] == error1

        assert len(handler2.errors) == 1
        assert handler2.errors[0] == error2

        # 互相不影响
        assert error1 not in handler2.errors
        assert error2 not in handler1.errors


class TestErrorHandlerIntegration:
    """ErrorHandler 集成测试"""

    def test_error_handling_workflow(self):
        """测试错误处理工作流"""
        handler = ErrorHandler()

        # 模拟一个工作流程中的错误处理
        def risky_operation(operation_id):
            if operation_id == "fail":
                raise ValueError(f"Operation {operation_id} failed")
            return f"Operation {operation_id} succeeded"

        operations = ["success1", "fail", "success2", "fail", "success3"]

        for op in operations:
            try:
                result = risky_operation(op)
                print(f"Result: {result}")
            except Exception as e:
                handled = handler.handle_error(e)
                assert handled == True

        # 验证错误被正确捕获
        errors = handler.get_errors()
        assert len(errors) == 2  # 两个失败的操作

        assert str(errors[0]) == "Operation fail failed"
        assert str(errors[1]) == "Operation fail failed"

        assert isinstance(errors[0], ValueError)
        assert isinstance(errors[1], ValueError)

    def test_error_statistics(self):
        """测试错误统计"""
        handler = ErrorHandler()

        # 添加不同类型的错误
        error_types = [ValueError, TypeError, RuntimeError, KeyError, AttributeError]

        for error_type in error_types:
            for i in range(3):  # 每种类型3个错误
                handler.handle_error(error_type(f"{error_type.__name__} {i}"))

        errors = handler.get_errors()
        assert len(errors) == 15  # 5种类型 * 3个 = 15个错误

        # 统计每种错误类型的数量
        error_counts = {}
        for error in errors:
            error_type = type(error).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        assert error_counts["ValueError"] == 3
        assert error_counts["TypeError"] == 3
        assert error_counts["RuntimeError"] == 3
        assert error_counts["KeyError"] == 3
        assert error_counts["AttributeError"] == 3

    def test_memory_management(self):
        """测试内存管理"""
        handler = ErrorHandler()

        # 添加大量错误来测试内存使用
        for i in range(1000):
            handler.handle_error(Exception(f"Error {i}"))

        assert len(handler.errors) == 1000

        # 获取错误列表
        errors = handler.get_errors()
        assert len(errors) == 1000

        # 验证所有错误都被保留
        for i in range(1000):
            assert str(handler.errors[i]) == f"Error {i}"

    def test_thread_safety_simulation(self):
        """测试线程安全模拟"""
        import threading

        handler = ErrorHandler()
        errors_per_thread = 50
        num_threads = 4

        def worker_thread(thread_id):
            for i in range(errors_per_thread):
                error = Exception(f"Thread {thread_id} - Error {i}")
                handler.handle_error(error)

        # 创建多个线程
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有错误都被正确处理
        total_expected_errors = num_threads * errors_per_thread
        assert len(handler.errors) == total_expected_errors

        # 验证错误内容
        thread_error_counts = {}
        for error in handler.errors:
            error_msg = str(error)
            thread_id = int(error_msg.split(" - ")[0].split(" ")[1])
            thread_error_counts[thread_id] = thread_error_counts.get(thread_id, 0) + 1

        # 每个线程应该有正确数量的错误
        for thread_id in range(num_threads):
            assert thread_error_counts[thread_id] == errors_per_thread