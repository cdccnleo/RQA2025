"""交易执行引擎测试"""

import unittest
from unittest.mock import MagicMock, patch
from src.trading.execution.execution_engine import ExecutionEngine, ExecutionStatus
from src.trading.execution.execution_algorithm import AlgorithmType

class TestExecutionEngine(unittest.TestCase):
    """交易执行引擎测试类"""

    def setUp(self):
        """测试初始化"""
        # 模拟配置和指标
        self.mock_config = MagicMock()
        self.mock_metrics = MagicMock()

        # 创建执行引擎实例
        self.engine = ExecutionEngine(self.mock_config, self.mock_metrics)

        # 模拟路由器和算法
        self.engine.router = MagicMock()
        self.engine.algorithms = {
            AlgorithmType.TWAP: MagicMock(),
            AlgorithmType.VWAP: MagicMock(),
            AlgorithmType.ICEBERG: MagicMock()
        }

        # 模拟风控通过
        self.engine.risk_controller = MagicMock()
        self.engine.risk_controller.check_order.return_value = {'allowed': True}

        # 测试订单
        self.sample_order = {
            'order_id': 'test_order',
            'symbol': '600000.SH',
            'quantity': 1000,
            'price': 10.5,
            'algo_type': 'TWAP'
        }

    def test_successful_execution(self):
        """测试成功执行"""
        # 设置算法返回结果
        self.engine.algorithms[AlgorithmType.TWAP].execute.return_value = [
            {'order_id': 'child1', 'status': 'filled', 'quantity': 1000}
        ]

        # 执行订单
        result = self.engine.execute_order(self.sample_order)

        # 检查结果
        self.assertEqual(result['status'], ExecutionStatus.COMPLETED)
        self.assertEqual(len(result['details']), 1)

        # 检查风控调用
        self.engine.risk_controller.check_order.assert_called_once_with(self.sample_order)

        # 检查指标记录
        self.mock_metrics.record_execution_time.assert_called_once()
        self.mock_metrics.record_order_status.assert_called_with(ExecutionStatus.COMPLETED)

    def test_partial_execution(self):
        """测试部分成交"""
        # 设置部分成交结果
        self.engine.algorithms[AlgorithmType.TWAP].execute.return_value = [
            {'order_id': 'child1', 'status': 'filled', 'quantity': 600},
            {'order_id': 'child2', 'status': 'rejected', 'quantity': 400}
        ]

        # 执行订单
        result = self.engine.execute_order(self.sample_order)

        # 检查结果
        self.assertEqual(result['status'], ExecutionStatus.PARTIAL)
        self.assertEqual(len(result['details']), 2)

    def test_risk_rejection(self):
        """测试风控拒绝"""
        # 设置风控拒绝
        self.engine.risk_controller.check_order.return_value = {
            'allowed': False,
            'reason': 'price_limit'
        }

        # 执行订单
        result = self.engine.execute_order(self.sample_order)

        # 检查结果
        self.assertEqual(result['status'], ExecutionStatus.REJECTED)
        self.assertEqual(result['reason'], 'price_limit')

        # 检查指标记录
        self.mock_metrics.record_rejection.assert_called_with('risk_rejected')

    def test_order_cancellation(self):
        """测试订单取消"""
        # 先执行订单
        self.engine.execute_order(self.sample_order)

        # 取消订单
        success = self.engine.cancel_order('test_order')

        # 检查结果
        self.assertTrue(success)
        self.assertEqual(
            self.engine.get_execution_status('test_order'),
            ExecutionStatus.CANCELLED
        )

        # 检查指标记录
        self.mock_metrics.record_cancellation.assert_called_once()

    def test_status_tracking(self):
        """测试状态跟踪"""
        # 初始状态应为PENDING
        self.assertEqual(
            self.engine.get_execution_status('nonexistent'),
            ExecutionStatus.PENDING
        )

        # 执行后状态更新
        self.engine.execute_order(self.sample_order)
        self.assertEqual(
            self.engine.get_execution_status('test_order'),
            ExecutionStatus.COMPLETED
        )

if __name__ == '__main__':
    unittest.main()
