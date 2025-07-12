import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.fpga import (
    FpgaAccelerator,
    FpgaRiskEngine,
    FpgaOptimizer
)

class TestFpgaModules(unittest.TestCase):
    """统一后的FPGA模块测试"""

    def setUp(self):
        # 初始化各FPGA模块
        self.accelerator = FpgaAccelerator()
        self.risk_engine = FpgaRiskEngine()
        self.optimizer = FpgaOptimizer()

        # 测试数据
        self.market_data = {
            'symbol': '600000.SH',
            'price': 42.50,
            'volume': 100000,
            'order_book': {
                'bids': [(42.49, 200), (42.48, 300)],
                'asks': [(42.51, 150), (42.52, 250)]
            }
        }

        # 模拟硬件连接
        self.accelerator._connect = MagicMock(return_value=True)
        self.risk_engine._connect = MagicMock(return_value=True)
        self.optimizer._connect = MagicMock(return_value=True)

    def test_fpga_connection(self):
        """测试FPGA连接"""
        self.assertTrue(self.accelerator.connect())
        self.assertTrue(self.risk_engine.connect())
        self.assertTrue(self.optimizer.connect())

    # 从test_fpga_accelerator.py合并的测试
    @patch('src.fpga.FpgaAccelerator._send_to_hardware')
    def test_sentiment_analysis(self, mock_send):
        """测试情感分析加速"""
        mock_send.return_value = {'sentiment': 0.85}
        news = ["公司发布利好公告"]
        result = self.accelerator.analyze_sentiment(news)
        self.assertGreater(result['sentiment'], 0.8)

    # 从test_fpga_risk_engine.py合并的测试
    def test_risk_check_performance(self):
        """测试风险检查性能"""
        order = {
            'symbol': '600000.SH',
            'price': 42.50,
            'quantity': 10000
        }

        # 模拟硬件加速
        with patch('src.fpga.FpgaRiskEngine._hardware_check') as mock_check:
            mock_check.return_value = {'approved': True, 'latency': 0.001}
            result = self.risk_engine.check_order(order)
            self.assertTrue(result['approved'])
            self.assertLess(result['latency'], 0.002)  # 应小于2ms

    # 从test_fpga_optimizer.py合并的测试
    def test_order_optimization(self):
        """测试订单优化"""
        large_order = {
            'symbol': '600000.SH',
            'side': 'BUY',
            'quantity': 50000
        }

        optimized = self.optimizer.optimize(large_order, self.market_data)
        self.assertTrue(len(optimized['child_orders']) > 1)
        self.assertEqual(
            sum(o['quantity'] for o in optimized['child_orders']),
            50000
        )

    # 新增降级测试
    @patch('src.fpga.FpgaAccelerator._connect')
    def test_fallback_mechanism(self, mock_connect):
        """测试硬件不可用时的降级机制"""
        mock_connect.return_value = False  # 模拟连接失败

        # 应自动降级到软件实现
        result = self.accelerator.analyze_sentiment(["测试"])
        self.assertIsNotNone(result)
        self.assertTrue(self.accelerator.using_software_fallback)

    # 新增性能对比测试
    def test_performance_improvement(self):
        """测试性能提升"""
        from src.features import SentimentAnalyzer

        # 准备测试数据
        news_data = ["利好新闻"] * 1000

        # 软件基准测试
        software_analyzer = SentimentAnalyzer()
        import time
        start = time.time()
        _ = software_analyzer.analyze_batch(news_data)
        software_time = time.time() - start

        # FPGA加速测试
        start = time.time()
        _ = self.accelerator.analyze_batch(news_data)
        fpga_time = time.time() - start

        # 验证加速效果
        self.assertLess(fpga_time, software_time * 0.5)  # 至少快2倍

if __name__ == '__main__':
    unittest.main()
