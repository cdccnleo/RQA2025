"""
基础设施层 - BaseErrorComponent 单元测试

测试错误处理层基础组件的核心功能，包括错误处理、历史记录、统计信息等。
覆盖率目标: 90%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch

from src.infrastructure.error.core.base import BaseErrorComponent


class TestBaseErrorComponent(unittest.TestCase):
    """BaseErrorComponent 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.component = BaseErrorComponent()
        self.component_with_config = BaseErrorComponent({'max_history': 50})

    def tearDown(self):
        """测试后清理"""
        pass

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        component = BaseErrorComponent()

        self.assertEqual(component.config, {})
        self.assertEqual(component._max_history, 1000)
        self.assertEqual(component._error_history, [])

    def test_initialization_with_config(self):
        """测试自定义配置初始化"""
        config = {'max_history': 500, 'custom_param': 'test'}
        component = BaseErrorComponent(config)

        self.assertEqual(component.config, config)
        self.assertEqual(component._max_history, 500)
        self.assertEqual(component._error_history, [])

    def test_handle_error_basic(self):
        """测试基础错误处理"""
        error = ValueError("测试错误")
        context = {"operation": "test_op", "user_id": 123}

        result = self.component.handle_error(error, context)

        # 验证返回结果
        self.assertIn('timestamp', result)
        self.assertEqual(result['error_type'], 'ValueError')
        self.assertEqual(result['message'], '测试错误')
        self.assertEqual(result['context'], context)
        self.assertEqual(result['handled'], False)

        # 验证历史记录
        self.assertEqual(len(self.component._error_history), 1)
        self.assertEqual(self.component._error_history[0], result)

    def test_handle_error_without_context(self):
        """测试无上下文的错误处理"""
        error = RuntimeError("运行时错误")

        result = self.component.handle_error(error)

        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(result['message'], '运行时错误')
        self.assertEqual(result['context'], {})
        self.assertEqual(result['handled'], False)

    def test_error_history_management(self):
        """测试错误历史管理"""
        # 添加多个错误
        errors = [
            ValueError("错误1"),
            RuntimeError("错误2"),
            ConnectionError("错误3")
        ]

        for error in errors:
            self.component.handle_error(error)

        # 验证历史记录
        history = self.component.get_error_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['error_type'], 'ValueError')
        self.assertEqual(history[1]['error_type'], 'RuntimeError')
        self.assertEqual(history[2]['error_type'], 'ConnectionError')

        # 验证历史记录返回正确数量
        self.assertEqual(len(history), 3)

    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        component = BaseErrorComponent({'max_history': 2})

        # 添加3个错误
        for i in range(3):
            error = ValueError(f"错误{i}")
            component.handle_error(error)

        # 验证只保留最新的2个
        history = component.get_error_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['message'], '错误1')
        self.assertEqual(history[1]['message'], '错误2')

    def test_clear_history(self):
        """测试清空历史记录"""
        # 添加错误
        self.component.handle_error(ValueError("测试错误"))
        self.assertEqual(len(self.component.get_error_history()), 1)

        # 清空历史
        self.component.clear_history()
        self.assertEqual(len(self.component.get_error_history()), 0)
        self.assertEqual(len(self.component._error_history), 0)

    def test_get_stats_empty(self):
        """测试获取空统计信息"""
        stats = self.component.get_stats()

        expected_stats = {
            'total_errors': 0,
            'error_types': {},
            'max_history': 1000,
            'current_history_size': 0
        }
        self.assertEqual(stats, expected_stats)

    def test_get_stats_with_errors(self):
        """测试获取带错误的统计信息"""
        # 添加不同类型的错误
        errors = [
            ValueError("值错误"),
            RuntimeError("运行时错误"),
            ValueError("另一个值错误"),
            ConnectionError("连接错误")
        ]

        for error in errors:
            self.component.handle_error(error)

        stats = self.component.get_stats()

        self.assertEqual(stats['total_errors'], 4)
        self.assertEqual(stats['current_history_size'], 4)
        self.assertEqual(stats['max_history'], 1000)

        # 验证错误类型统计
        error_types = stats['error_types']
        self.assertEqual(error_types['ValueError'], 2)
        self.assertEqual(error_types['RuntimeError'], 1)
        self.assertEqual(error_types['ConnectionError'], 1)

    def test_get_stats_with_config(self):
        """测试带配置的统计信息"""
        component = BaseErrorComponent({'max_history': 200})
        stats = component.get_stats()

        self.assertEqual(stats['max_history'], 200)
        self.assertEqual(stats['total_errors'], 0)

    def test_error_history_independence(self):
        """测试错误历史记录的独立性"""
        # 创建两个组件实例
        component1 = BaseErrorComponent()
        component2 = BaseErrorComponent()

        # 给第一个组件添加错误
        component1.handle_error(ValueError("错误1"))

        # 验证第二个组件不受影响
        self.assertEqual(len(component1.get_error_history()), 1)
        self.assertEqual(len(component2.get_error_history()), 0)

    def test_large_error_history(self):
        """测试大量错误历史记录"""
        component = BaseErrorComponent({'max_history': 100})

        # 添加101个错误
        for i in range(101):
            error = ValueError(f"错误{i}")
            component.handle_error(error)

        history = component.get_error_history()
        self.assertEqual(len(history), 100)

        # 验证保留的是最新的100个
        self.assertEqual(history[0]['message'], '错误1')
        self.assertEqual(history[-1]['message'], '错误100')


if __name__ == '__main__':
    unittest.main()
