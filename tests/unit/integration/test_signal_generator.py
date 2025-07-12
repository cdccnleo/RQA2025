"""实时交易信号生成集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.signal.signal_generator import SignalGenerator, ChinaSignalGenerator
from src.features.feature_engine import FeatureEngine
from src.fpga.fpga_accelerator import FpgaManager

class TestSignalGeneratorIntegration(unittest.TestCase):
    """信号生成器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建信号生成器
        self.generator = SignalGenerator(self.engine)
        self.china_generator = ChinaSignalGenerator(self.engine)

        # 测试数据
        self.test_features = {
            "price": 10.0,
            "momentum": 0.8,
            "sentiment": 0.7,
            "order_flow": 0.6,
            "margin_ratio": 0.5,
            "short_balance": 0.4,
            "institutional_net_buy": 0.9,
            "hot_money_flow": 0.8
        }

        # 批量测试数据
        self.batch_symbols = ["600519.SH", "000001.SZ", "300750.SZ"]
        self.batch_features = [self.test_features] * 3

    def test_basic_signal_generation(self):
        """测试基本信号生成"""
        signal = self.generator.generate("600519.SH", self.test_features)

        # 检查信号结构
        self.assertIn("signal", signal)
        self.assertIn("confidence", signal)
        self.assertIn("target_price", signal)
        self.assertIn("position", signal)

        # 检查信号类型
        self.assertIn(signal["signal"], ["BUY", "SELL", "HOLD"])

    def test_a_share_specific_rules(self):
        """测试A股特定规则"""
        # 测试ST股票
        st_features = {**self.test_features, "is_st": True}
        st_signal = self.china_generator.generate("ST600519", st_features)
        self.assertLessEqual(st_signal["position"], 0.05)  # ST股票仓位应降低

        # 测试科创板
        star_signal = self.china_generator.generate("688981.SH", self.test_features)
        self.assertIsNotNone(star_signal)

    def test_fpga_acceleration(self):
        """测试FPGA加速路径"""
        # 模拟FPGA加速器
        fpga_mock = MagicMock()
        fpga_mock.health_monitor.is_healthy.return_value = True
        fpga_mock._generate_with_fpga.return_value = {
            "signal": "BUY",
            "confidence": 0.9,
            "target_price": 10.1,
            "position": 0.1
        }

        with patch.object(FpgaManager, 'get_accelerator', return_value=fpga_mock):
            signal = self.generator.generate("600519.SH", self.test_features)
            self.assertEqual(signal["confidence"], 0.9)
            fpga_mock._generate_with_fpga.assert_called_once()

    def test_software_fallback(self):
        """测试软件降级路径"""
        # 禁用FPGA
        self.generator.config.use_fpga = False

        signal = self.generator.generate("600519.SH", self.test_features)
        self.assertIsNotNone(signal)

    def test_cool_down_period(self):
        """测试冷却期机制"""
        # 首次生成信号
        signal1 = self.generator.generate("600519.SH", self.test_features)
        self.assertIsNotNone(signal1)

        # 立即再次生成应返回None（在冷却期内）
        signal2 = self.generator.generate("600519.SH", self.test_features)
        self.assertIsNone(signal2)

    def test_batch_generation(self):
        """测试批量信号生成"""
        signals = self.generator.batch_generate(self.batch_symbols, self.batch_features)

        # 检查批量结果
        self.assertEqual(len(signals), len(self.batch_symbols))
        for signal in signals:
            self.assertIsNotNone(signal)

    def test_integration_with_feature_engine(self):
        """测试与特征引擎的集成"""
        # 注册测试特征
        self.engine.register_feature("momentum", lambda x: x.get("momentum", 0))
        self.engine.register_feature("sentiment", lambda x: x.get("sentiment", 0))

        # 通过特征引擎生成信号
        signal = self.generator.generate("600519.SH", self.test_features)
        self.assertIsNotNone(signal)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch_symbols = ["600519.SH"] * 1000
        large_batch_features = [self.test_features] * 1000

        start = time.time()
        self.generator.batch_generate(large_batch_symbols, large_batch_features)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次生成应在1秒内完成
        print(f"信号生成性能: {1000/elapsed:.2f} 次/秒")

if __name__ == '__main__':
    unittest.main()
