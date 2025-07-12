"""FPGA加速模块集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.fpga.fpga_accelerator import FpgaAccelerator, FpgaConfig, FpgaManager
from src.features.feature_engine import FeatureEngine

class TestFpgaAcceleratorIntegration(unittest.TestCase):
    """FPGA加速模块集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建FPGA加速器
        self.config = FpgaConfig(enabled=True, fallback_enabled=True)
        self.accelerator = FpgaAccelerator(self.engine, self.config)

        # 测试数据
        self.test_data = {
            "price": np.array([10.0, 10.1, 10.05, 10.15]),
            "volume": np.array([100, 150, 120, 180]),
            "bid": np.array([[9.99, 200], [9.98, 150]]),
            "ask": np.array([[10.01, 80], [10.02, 120]]),
            "text_data": "证监会发布新政策推动资本市场高质量发展"
        }

        # 批量测试数据
        self.batch_data = [self.test_data] * 10

    def test_feature_registration(self):
        """测试FPGA特征注册"""
        # 检查FPGA特征
        self.assertIn("SENTIMENT_FPGA", self.engine.feature_registry)
        self.assertIn("ORDER_BOOK_FPGA", self.engine.feature_registry)
        self.assertIn("MOMENTUM_FPGA", self.engine.feature_registry)

    def test_fpga_feature_calculation(self):
        """测试FPGA特征计算"""
        # 测试情感分析
        sentiment = self.accelerator.calculate_feature(
            "SENTIMENT_FPGA",
            {"text_data": self.test_data["text_data"]}
        )
        self.assertIn("feature1", sentiment)

        # 测试订单簿分析
        order_book = self.accelerator.calculate_feature(
            "ORDER_BOOK_FPGA",
            {"bid": self.test_data["bid"], "ask": self.test_data["ask"]}
        )
        self.assertIn("feature1", order_book)

        # 测试动量计算
        momentum = self.accelerator.calculate_feature(
            "MOMENTUM_FPGA",
            {"price": self.test_data["price"], "volume": self.test_data["volume"]}
        )
        self.assertIn("feature1", momentum)

    @patch.object(FpgaAccelerator, '_prepare_fpga_data')
    def test_fpga_disabled(self, mock_prepare):
        """测试FPGA禁用时的降级处理"""
        # 禁用FPGA
        self.accelerator.config.enabled = False

        # 调用情感分析
        sentiment = self.accelerator.calculate_feature(
            "SENTIMENT_FPGA",
            {"text_data": self.test_data["text_data"]}
        )

        # 验证降级路径
        self.assertIn("feature1", sentiment)
        mock_prepare.assert_not_called()  # 确保没有调用FPGA准备方法

    @patch.object(FpgaAccelerator, '_prepare_fpga_data')
    def test_fpga_unhealthy(self, mock_prepare):
        """测试FPGA不健康时的降级处理"""
        # 模拟FPGA不健康
        self.accelerator.health_monitor._is_healthy = False

        # 调用订单簿分析
        order_book = self.accelerator.calculate_feature(
            "ORDER_BOOK_FPGA",
            {"bid": self.test_data["bid"], "ask": self.test_data["ask"]}
        )

        # 验证降级路径
        self.assertIn("feature1", order_book)
        mock_prepare.assert_not_called()  # 确保没有调用FPGA准备方法

    def test_batch_processing(self):
        """测试批量特征计算"""
        # 测试批量情感分析
        batch_results = self.accelerator.batch_calculate(
            "SENTIMENT_FPGA",
            [{"text_data": d["text_data"]} for d in self.batch_data]
        )

        # 检查批量结果
        self.assertEqual(len(batch_results), len(self.batch_data))
        for result in batch_results:
            self.assertIn("feature1", result)

    def test_integration_with_feature_engine(self):
        """测试与特征引擎的集成"""
        # 通过特征引擎调用FPGA特征
        features = self.engine.calculate_features({
            "SENTIMENT_FPGA": {"text_data": self.test_data["text_data"]},
            "ORDER_BOOK_FPGA": {
                "bid": self.test_data["bid"],
                "ask": self.test_data["ask"]
            }
        })

        # 检查特征结果
        self.assertIn("SENTIMENT_FPGA", features)
        self.assertIn("ORDER_BOOK_FPGA", features)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch = [{"text_data": d["text_data"]} for d in self.batch_data * 100]

        start = time.time()
        self.accelerator.batch_calculate("SENTIMENT_FPGA", large_batch)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次计算应在1秒内完成
        print(f"FPGA批量处理性能: {1000/elapsed:.2f} 次/秒")

class TestFpgaManager(unittest.TestCase):
    """FPGA管理器测试用例"""

    def setUp(self):
        """测试初始化"""
        self.manager = FpgaManager()
        self.engine = FeatureEngine()
        self.accelerator = FpgaAccelerator(self.engine)

    def test_singleton_pattern(self):
        """测试单例模式"""
        manager2 = FpgaManager()
        self.assertIs(self.manager, manager2)

    def test_accelerator_registration(self):
        """测试加速器注册"""
        self.manager.register_accelerator("SENTIMENT_FPGA", self.accelerator)
        self.assertEqual(
            self.manager.get_accelerator("SENTIMENT_FPGA"),
            self.accelerator
        )

    def test_shutdown(self):
        """测试关闭管理"""
        self.manager.register_accelerator("SENTIMENT_FPGA", self.accelerator)
        self.manager.shutdown()
        self.assertEqual(len(self.manager.accelerators), 0)

if __name__ == '__main__':
    unittest.main()
