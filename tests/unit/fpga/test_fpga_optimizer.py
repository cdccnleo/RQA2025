"""FPGA订单优化器测试"""

import unittest
from unittest.mock import MagicMock, patch
from src.fpga.fpga_optimizer import FPGAOptimizer

class TestFPGAOptimizer(unittest.TestCase):
    """FPGA订单优化器测试类"""

    def setUp(self):
        """测试初始化"""
        self.optimizer = FPGAOptimizer()
        self.optimizer.initialize()

        # 测试数据
        self.sample_order_book = {
            "mid_price": 10.5,
            "volume": 10000,
            "bids": [(10.4, 5000), (10.3, 3000)],
            "asks": [(10.6, 4000), (10.7, 2000)]
        }
        self.sample_order = {
            "symbol": "600000.SH",
            "quantity": 1000,
            "price": 10.5,
            "algo_type": "TWAP"
        }

    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.optimizer.initialized)

    def test_twap_optimization(self):
        """测试TWAP优化"""
        result = self.optimizer.optimize_twap(self.sample_order_book, self.sample_order)
        self.assertIsInstance(result, dict)
        self.assertIn("price", result)
        self.assertIn("quantity", result)
        self.assertIn("timing", result)

    def test_vwap_optimization(self):
        """测试VWAP优化"""
        result = self.optimizer.optimize_vwap(self.sample_order_book, self.sample_order)
        self.assertIsInstance(result, dict)
        self.assertIn("price", result)
        self.assertIn("quantity", result)
        self.assertIn("timing", result)

    def test_iceberg_optimization(self):
        """测试冰山订单优化"""
        result = self.optimizer.optimize_iceberg(self.sample_order_book, self.sample_order)
        self.assertIsInstance(result, dict)
        self.assertIn("price", result)
        self.assertIn("quantity", result)
        self.assertIn("display_quantity", result)

    def test_batch_optimization(self):
        """测试批量优化"""
        order_books = [self.sample_order_book] * 3
        orders = [self.sample_order] * 3
        results = self.optimizer.batch_optimize(order_books, orders)
        self.assertEqual(len(results), len(orders))
        for result in results:
            self.assertIsInstance(result, dict)

    @patch.object(FPGAOptimizer, 'initialize')
    def test_initialization_failure(self, mock_init):
        """测试初始化失败"""
        mock_init.side_effect = Exception("FPGA初始化失败")
        optimizer = FPGAOptimizer()
        with self.assertRaises(Exception):
            optimizer.initialize()
        self.assertFalse(optimizer.initialized)

if __name__ == '__main__':
    unittest.main()
