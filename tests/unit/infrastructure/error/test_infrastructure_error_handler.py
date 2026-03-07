"""
基础设施层 - InfrastructureErrorHandler 单元测试

测试基础设施错误处理器的核心功能，包括网络、数据库、异步、边界条件错误处理。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from src.infrastructure.error.handlers.infrastructure_error_handler import (
    InfrastructureErrorHandler,
    DatabaseErrorType,
    NetworkErrorType,
    AsyncErrorType,
    BoundaryConditionType
)
from src.infrastructure.error.handlers.boundary_condition_manager import BoundaryConditionManager
from src.infrastructure.error.handlers.error_classifier import ErrorClassifier


class TestInfrastructureErrorHandler(unittest.TestCase):
    """InfrastructureErrorHandler 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.handler = InfrastructureErrorHandler(max_history=100)

    def tearDown(self):
        """测试后清理"""
        self.handler.clear_history()

    def test_initialization(self):
        """测试初始化"""
        handler = InfrastructureErrorHandler(max_history=50)
        self.assertIsInstance(handler, InfrastructureErrorHandler)
        self.assertEqual(handler._max_history, 50)
        self.assertIsInstance(handler._boundary_manager, BoundaryConditionManager)  # 修正为实际存在的属性
        self.assertIsInstance(handler._error_classifier, ErrorClassifier)  # 修正为实际存在的属性

    def test_handle_connection_error(self):
        """测试连接错误处理"""
        error = ConnectionError("连接失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'ConnectionError')
        self.assertEqual(result['category'], 'network')

    def test_handle_timeout_error(self):
        """测试超时错误处理"""
        error = TimeoutError("操作超时")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['category'], 'network')

    def test_handle_registered_connection_error(self):
        """测试注册的连接错误处理器"""
        error = ConnectionError("连接失败")

        # 注册自定义处理器
        mock_handler = Mock(return_value={'custom_action': 'retry'})
        self.handler.register_handler(ConnectionError, mock_handler)

        result = self.handler.handle_error(error)

        mock_handler.assert_called_once_with(error, None)
        self.assertTrue(result['handled'])
        self.assertEqual(result['custom_action'], 'retry')

    def test_handle_registered_timeout_error(self):
        """测试注册的超时错误处理器"""
        error = TimeoutError("超时")

        mock_handler = Mock(return_value={'action': 'retry', 'delay': 5.0})
        self.handler.register_handler(TimeoutError, mock_handler)

        result = self.handler.handle_error(error)

        mock_handler.assert_called_once_with(error, None)
        self.assertTrue(result['handled'])
        self.assertEqual(result['action'], 'retry')

    def test_handle_io_error(self):
        """测试IO错误处理"""
        error = IOError("文件操作失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'IOError')
        self.assertEqual(result['category'], 'system')

    def test_handle_os_error(self):
        """测试OS错误处理"""
        error = OSError("系统操作失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'OSError')
        self.assertEqual(result['category'], 'system')

    def test_handle_async_error(self):
        """测试异步错误处理"""
        error = KeyboardInterrupt("异步取消")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'KeyboardInterrupt')

    def test_system_exit_handling(self):
        """测试系统退出处理"""
        error = SystemExit("系统退出")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'SystemExit')

    def test_boundary_conditions_setup(self):
        """测试边界条件设置"""
        # 验证默认边界条件 - 通过boundary_manager访问
        self.assertGreater(len(self.handler._boundary_manager._boundary_conditions), 0)

        # 查找特定的边界条件
        value_range_condition = None
        for condition in self.handler._boundary_manager._boundary_conditions:
            if condition.condition_type == BoundaryConditionType.VALUE_OUT_OF_RANGE:
                value_range_condition = condition
                break

        self.assertIsNotNone(value_range_condition)
        self.assertEqual(value_range_condition.severity, "warning")
        self.assertIn("超出有效范围", value_range_condition.description)

    def test_add_boundary_condition(self):
        """测试添加边界条件"""
        from src.infrastructure.error.handlers.infrastructure_error_handler import BoundaryCondition

        condition = BoundaryCondition(
            condition_type=BoundaryConditionType.NULL_REFERENCE,
            severity="error",
            description="空引用错误",
            suggested_action="检查对象初始化",
            context={"field": "user"}
        )

        self.handler.add_boundary_condition(
            BoundaryConditionType.NULL_REFERENCE,
            "error",
            "空引用错误",
            "检查对象初始化",
            {"field": "user"}
        )

        # 验证边界条件已添加 - 通过boundary_manager访问
        found = False
        for bc in self.handler._boundary_manager._boundary_conditions:
            if bc.condition_type == BoundaryConditionType.NULL_REFERENCE:
                found = True
                self.assertEqual(bc.severity, "error")
                break

        self.assertTrue(found)

    def test_check_boundary_conditions(self):
        """测试边界条件检查"""
        # 添加一个测试边界条件
        self.handler.add_boundary_condition(
            BoundaryConditionType.VALUE_OUT_OF_RANGE,
            "warning",
            "测试边界条件",
            "调整值范围",
            {"min": 0, "max": 100}
        )

        # 检查边界条件 - 使用正确的方法
        context = {"value": 150}
        results = self.handler._boundary_manager.check_boundary_conditions(context)

        # 验证结果结构
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # 查找我们的测试条件结果
        test_result = None
        for result in results:
            if "测试边界条件" in result.get('description', ''):
                test_result = result
                break

        self.assertIsNotNone(test_result)
        self.assertIn('condition_type', test_result)
        self.assertIn('severity', test_result)
        self.assertIn('description', test_result)
        self.assertIn('suggested_action', test_result)

    def test_error_with_boundary_check(self):
        """测试包含边界检查的错误处理"""
        error = ValueError("测试错误")
        context = {"value": 200, "operation": "validation"}

        result = self.handler.handle_error(error, context)

        # 验证结果包含边界检查信息
        self.assertIn('boundary_check', result)
        self.assertIsInstance(result['boundary_check'], list)

    def test_severity_classification(self):
        """测试错误严重程度分类"""
        # 测试Critical错误
        critical_error = SystemExit("系统退出")
        result = self.handler.handle_error(critical_error)
        self.assertEqual(result['severity'], 'critical')

        # 测试Error级别错误
        connection_error = ConnectionError("连接错误")
        result = self.handler.handle_error(connection_error)
        self.assertEqual(result['category'], 'network')

        # 测试Warning级别错误
        value_error = ValueError("值错误")
        result = self.handler.handle_error(value_error)
        self.assertEqual(result['severity'], 'info')  # ValueError通常是info级别

    def test_category_classification(self):
        """测试错误类别分类"""
        # 测试网络错误
        connection_error = ConnectionError("连接错误")
        result = self.handler.handle_error(connection_error)
        self.assertEqual(result['category'], 'network')

        # 测试数据库相关错误
        # 注意：这里我们用一个模拟的数据库错误
        db_error = Exception("Database connection failed")
        result = self.handler.handle_error(db_error)
        self.assertEqual(result['category'], 'unknown')  # 默认分类

        # 测试系统错误
        io_error = IOError("IO错误")
        result = self.handler.handle_error(io_error)
        self.assertEqual(result['category'], 'system')

    def test_get_stats_comprehensive(self):
        """测试完整的统计信息"""
        # 添加各种类型的错误
        errors = [
            ConnectionError("网络错误"),
            TimeoutError("超时错误"),
            IOError("IO错误"),
            ValueError("值错误")
        ]

        for error in errors:
            self.handler.handle_error(error)

        stats = self.handler.get_stats()

        # 验证统计信息结构
        self.assertIn('total_errors', stats)
        self.assertIn('severity_distribution', stats)
        self.assertIn('category_distribution', stats)
        self.assertIn('registered_handlers', stats)
        self.assertIn('boundary_conditions', stats)

        # 验证数据正确性
        self.assertEqual(stats['total_errors'], 4)
        self.assertGreater(stats['boundary_conditions'], 0)

    def test_history_management_with_boundary(self):
        """测试包含边界条件的错误历史管理"""
        # 添加错误
        error = ConnectionError("测试连接错误")
        context = {"operation": "network_call", "timeout": 30}
        self.handler.handle_error(error, context)

        # 获取历史
        history = self.handler.get_error_history()
        self.assertEqual(len(history), 1)

        # 验证历史记录包含边界检查信息
        error_record = history[0]
        self.assertIn('boundary_check', error_record)
        self.assertIsInstance(error_record['boundary_check'], list)

    def test_registered_handlers_list(self):
        """测试已注册处理器列表"""
        # 初始状态
        handlers = self.handler.get_registered_handlers()
        self.assertIsInstance(handlers, list)

        # 注册处理器后
        self.handler.register_handler(ValueError, lambda e, c: {'handled': True})
        handlers = self.handler.get_registered_handlers()
        self.assertIn('ValueError', handlers)

    def test_registered_strategies_list(self):
        """测试已注册策略列表"""
        # 注册策略
        self.handler.register_strategy('retry_strategy', lambda: {'retry': True})
        strategies = self.handler.get_registered_strategies()
        self.assertIn('retry_strategy', strategies)

    def test_context_preservation(self):
        """测试上下文信息保留"""
        error = ConnectionError("连接失败")
        context = {
            'user_id': 123,
            'operation': 'api_call',
            'endpoint': '/api/data',
            'timeout': 30
        }

        result = self.handler.handle_error(error, context)

        # 验证上下文被完整保留
        self.assertEqual(result['context']['user_id'], 123)
        self.assertEqual(result['context']['operation'], 'api_call')
        self.assertEqual(result['context']['endpoint'], '/api/data')
        self.assertEqual(result['context']['timeout'], 30)


if __name__ == '__main__':
    unittest.main()
