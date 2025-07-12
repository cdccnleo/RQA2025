"""执行算法框架测试"""

import unittest
from unittest.mock import MagicMock, patch
from src.trading.execution.execution_algorithm import (
    ExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    IcebergAlgorithm,
    AlgorithmType,
    get_algorithm
)

class TestExecutionAlgorithm(unittest.TestCase):
    """执行算法框架测试类"""

    def setUp(self):
        """测试初始化"""
        self.mock_config = MagicMock()
        self.mock_metrics = MagicMock()
        self.mock_router = MagicMock()

        # 模拟路由结果
        self.sample_order = {
            'symbol': '600000.SH',
            'quantity': 1000,
            'price': 10.5,
            'order_id': 'test_order'
        }
        self.mock_router.route_order.return_value = [self.sample_order]

    def test_twap_algorithm(self):
        """测试TWAP算法"""
        algo = TWAPAlgorithm(self.mock_config, self.mock_metrics)
        algo.router = self.mock_router

        # 执行TWAP算法
        algo_params = {'time_slices': 5, 'slice_interval': 60}
        result = algo.execute({**self.sample_order, 'algo_params': algo_params})

        # 检查结果
        self.assertEqual(len(result), 5)  # 5个时间切片
        self.mock_metrics.record_execution.assert_called_once()

    def test_vwap_algorithm(self):
        """测试VWAP算法"""
        algo = VWAPAlgorithm(self.mock_config, self.mock_metrics)
        algo.router = self.mock_router

        # 执行VWAP算法
        algo_params = {'volume_profile': [0.2, 0.3, 0.5]}
        result = algo.execute({**self.sample_order, 'algo_params': algo_params})

        # 检查结果
        self.assertEqual(len(result), 3)  # 3个成交量切片
        self.mock_metrics.record_execution.assert_called_once()

    def test_iceberg_algorithm(self):
        """测试冰山算法"""
        algo = IcebergAlgorithm(self.mock_config, self.mock_metrics)
        algo.router = self.mock_router

        # 执行冰山算法
        algo_params = {'peak_size': 100}
        large_order = {**self.sample_order, 'quantity': 500}
        result = algo.execute({**large_order, 'algo_params': algo_params})

        # 检查结果
        self.assertEqual(len(result), 5)  # 500/100=5个冰山订单
        self.mock_metrics.record_execution.assert_called_once()

    def test_algorithm_factory(self):
        """测试算法工厂方法"""
        # 测试TWAP
        twap = get_algorithm(AlgorithmType.TWAP)
        self.assertIsInstance(twap, TWAPAlgorithm)

        # 测试VWAP
        vwap = get_algorithm(AlgorithmType.VWAP)
        self.assertIsInstance(vwap, VWAPAlgorithm)

        # 测试ICEBERG
        iceberg = get_algorithm(AlgorithmType.ICEBERG)
        self.assertIsInstance(iceberg, IcebergAlgorithm)

    def test_base_class_abstract(self):
        """测试基类抽象方法"""
        algo = ExecutionAlgorithm(AlgorithmType.TWAP)
        algo.router = self.mock_router

        with self.assertRaises(NotImplementedError):
            algo._apply_algorithm([], {})

if __name__ == '__main__':
    unittest.main()
