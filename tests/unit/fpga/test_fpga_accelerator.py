"""FPGA加速模块测试"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.fpga.fpga_accelerator import FPGAAccelerator, SoftwareFallback

class TestFPGAAccelerator(unittest.TestCase):
    """FPGA加速器测试类"""

    def setUp(self):
        """测试初始化"""
        self.accelerator = FPGAAccelerator()
        self.accelerator.initialize()

        # 测试数据
        self.sample_text = "科技公司发布创新产品"
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
        """测试FPGA初始化"""
        self.assertTrue(self.accelerator.initialized)

    def test_sentiment_analysis(self):
        """测试情感分析加速"""
        result = self.accelerator.accelerate_sentiment_analysis(self.sample_text)

        # 检查返回结构
        self.assertIn("positive", result)
        self.assertIn("negative", result)
        self.assertIn("policy_keywords", result)

        # 检查值范围
        self.assertTrue(0 <= result["positive"] <= 1)
        self.assertTrue(0 <= result["negative"] <= 1)
        self.assertIsInstance(result["policy_keywords"], list)

    def test_order_optimization(self):
        """测试订单优化加速"""
        result = self.accelerator.accelerate_order_optimization(self.sample_order_book)

        # 检查返回结构
        self.assertIn("price", result)
        self.assertIn("quantity", result)
        self.assertIn("strategy", result)

        # 检查优化逻辑
        self.assertLessEqual(result["price"], self.sample_order_book["mid_price"])
        self.assertLessEqual(result["quantity"], self.sample_order_book["volume"])

    def test_risk_check(self):
        """测试风控检查加速"""
        result = self.accelerator.accelerate_risk_check(self.sample_order)
        self.assertIsInstance(result, bool)

    @patch.object(FPGAAccelerator, 'initialize')
    def test_initialization_failure(self, mock_init):
        """测试初始化失败"""
        mock_init.side_effect = Exception("FPGA初始化失败")
        accelerator = FPGAAccelerator()
        with self.assertRaises(Exception):
            accelerator.initialize()
        self.assertFalse(accelerator.initialized)

class TestSoftwareFallback(unittest.TestCase):
    """软件降级方案测试类"""

    def setUp(self):
        """测试初始化"""
        self.sample_text = "科技公司发布创新产品"
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

    @patch('transformers.pipeline')
    def test_sentiment_analysis_fallback(self, mock_pipeline):
        """测试情感分析降级"""
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]
        result = SoftwareFallback.sentiment_analysis(self.sample_text)
        self.assertIsInstance(result, list)

    @patch('src.fpga.fpga_accelerator.SmartOrderRouter')
    def test_order_optimization_fallback(self, mock_router):
        """测试订单优化降级"""
        mock_router.return_value.optimize.return_value = {
            "price": 10.4,
            "quantity": 500,
            "strategy": "aggressive"
        }
        result = SoftwareFallback.order_optimization(self.sample_order_book)
        self.assertIsInstance(result, dict)

    @patch('src.fpga.fpga_accelerator.RiskEngine')
    def test_risk_check_fallback(self, mock_engine):
        """测试风控检查降级"""
        mock_engine.return_value.check_order.return_value = True
        result = SoftwareFallback.risk_check(self.sample_order)
        self.assertIsInstance(result, bool)

if __name__ == '__main__':
    unittest.main()
