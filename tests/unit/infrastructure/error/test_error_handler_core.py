"""
基础设施层 - ErrorHandler 单元测试

测试通用错误处理器的核心功能，包括错误处理、策略注册、统计信息等。
覆盖率目标: 80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from src.infrastructure.error.handlers.error_handler import ErrorHandler


class TestErrorHandler(unittest.TestCase):
    """ErrorHandler 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.handler = ErrorHandler()
        self.handler_with_config = ErrorHandler(max_history=50)

    def tearDown(self):
        """测试后清理"""
        pass

    def test_initialization_default(self):
        """测试默认初始化"""
        handler = ErrorHandler()

        self.assertEqual(handler.handlers, {})
        self.assertEqual(handler.strategies, {})
        self.assertEqual(handler.error_stats, {})
        self.assertEqual(handler._error_history, [])
        self.assertEqual(handler._max_history, 1000)

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        handler = ErrorHandler(max_history=500)

        self.assertEqual(handler._max_history, 500)

    def test_register_handler(self):
        """测试注册错误处理器"""
        def test_handler(error, context=None):
            return {"handled": True, "message": "handled"}

        self.handler.register_handler(ValueError, test_handler)

        self.assertIn(ValueError, self.handler.handlers)
        self.assertEqual(self.handler.handlers[ValueError], test_handler)

    def test_register_strategy(self):
        """测试注册错误处理策略"""
        def test_strategy(error, context=None):
            return {"strategy": "test", "result": "applied"}

        self.handler.register_strategy("test_strategy", test_strategy)

        self.assertIn("test_strategy", self.handler.strategies)
        self.assertEqual(self.handler.strategies["test_strategy"], test_strategy)

    def test_handle_error_basic(self):
        """测试基础错误处理"""
        error = ValueError("测试错误")
        context = {"operation": "test_op", "user_id": 123}

        result = self.handler.handle_error(error, context)

        # 验证返回结果结构
        self.assertIn('handled', result)
        self.assertIn('error_type', result)
        self.assertIn('message', result)
        self.assertIn('severity', result)
        self.assertIn('category', result)

        # 验证具体值
        self.assertEqual(result['error_type'], 'ValueError')
        self.assertEqual(result['message'], '测试错误')

        # 验证历史记录
        self.assertEqual(len(self.handler._error_history), 1)

    def test_handle_error_without_context(self):
        """测试无上下文的错误处理"""
        error = RuntimeError("运行时错误")

        result = self.handler.handle_error(error)

        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(result['message'], '运行时错误')

    def test_register_handler(self):
        """测试注册错误处理器"""
        def test_handler(error, context=None):
            return {"handled": True, "message": "handled"}

        self.handler.register_handler(ValueError, test_handler)

        self.assertIn(ValueError, self.handler.handlers)
        self.assertEqual(self.handler.handlers[ValueError], test_handler)

    def test_handle_error_with_registered_strategy(self):
        """测试带已注册策略的错误处理"""
        def custom_strategy(error, context=None):
            return {
                "strategy_applied": True,
                "strategy_name": "custom",
                "processed": True
            }

        self.handler.register_strategy("custom", custom_strategy)

        # 策略在当前实现中不直接调用，这里测试注册功能
        self.assertIn("custom", self.handler.strategies)

    def test_error_statistics_update(self):
        """测试错误统计更新"""
        # 添加多个错误
        errors = [
            ValueError("错误1"),
            RuntimeError("错误2"),
            ValueError("错误3"),
            ConnectionError("错误4")
        ]

        for error in errors:
            self.handler.handle_error(error)

        # 验证统计信息
        self.assertEqual(self.handler.error_stats['ValueError'], 2)
        self.assertEqual(self.handler.error_stats['RuntimeError'], 1)
        self.assertEqual(self.handler.error_stats['ConnectionError'], 1)

    def test_error_history_management(self):
        """测试错误历史管理"""
        # 添加多个错误
        for i in range(5):
            error = ValueError(f"错误{i}")
            self.handler.handle_error(error)

        # 验证历史记录数量
        self.assertEqual(len(self.handler._error_history), 5)

        # 验证历史记录内容
        for i, record in enumerate(self.handler._error_history):
            self.assertEqual(record['error_type'], 'ValueError')
            self.assertEqual(record['message'], f'错误{i}')

    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        handler = ErrorHandler(max_history=3)

        # 添加4个错误
        for i in range(4):
            error = ValueError(f"错误{i}")
            handler.handle_error(error)

        # 验证只保留最新的3个
        self.assertEqual(len(handler._error_history), 3)
        self.assertEqual(handler._error_history[0]['message'], '错误1')
        self.assertEqual(handler._error_history[1]['message'], '错误2')
        self.assertEqual(handler._error_history[2]['message'], '错误3')

    def test_get_error_history(self):
        """测试获取错误历史"""
        # 添加错误
        self.handler.handle_error(ValueError("测试错误"))

        history = self.handler.get_error_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['error_type'], 'ValueError')

    def test_clear_history(self):
        """测试清空历史记录"""
        self.handler.handle_error(ValueError("测试错误"))
        self.assertEqual(len(self.handler.get_error_history()), 1)

        self.handler.clear_history()
        self.assertEqual(len(self.handler.get_error_history()), 0)


    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # 每个线程处理多个错误
                for i in range(10):
                    error = ValueError(f"线程{worker_id}错误{i}")
                    result = self.handler.handle_error(error)
                    results.append((worker_id, result['error_type']))
                time.sleep(0.01)  # 小的延迟以增加并发性
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 30)  # 3线程 * 10错误

        # 验证历史记录总数
        self.assertEqual(len(self.handler._error_history), 30)


if __name__ == '__main__':
    unittest.main()
