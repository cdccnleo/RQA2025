"""高频特征提取优化器集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from src.features.feature_engine import FeatureEngine
from src.features.high_freq_optimizer import HighFreqOptimizer, HighFreqConfig
from src.features.level2_analyzer import Level2Analyzer

class TestHighFreqOptimizerIntegration(unittest.TestCase):
    """高频特征提取优化器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建高频特征优化器
        config = HighFreqConfig(batch_size=100, prealloc_memory=True)
        self.optimizer = HighFreqOptimizer(self.engine, config)

        # 测试数据
        self.test_data = {
            "price": np.array([10.0, 10.1, 10.05, 10.15, 10.2, 10.18, 10.25, 10.3, 10.28, 10.35]),
            "volume": np.array([100, 150, 120, 180, 200, 190, 220, 250, 230, 260]),
            "bid": np.array([
                [9.99, 200], [9.98, 150], [9.97, 300], [9.96, 250], [9.95, 100]
            ]),
            "ask": np.array([
                [10.01, 80], [10.02, 120], [10.03, 200], [10.04, 150], [10.05, 100]
            ]),
            "trade": np.array([
                [10.01, 1, 50],   # [price, direction(1=buy), volume]
                [10.0, -1, 100],
                [10.02, 1, 200],
                [9.99, -1, 50],
                [10.01, 1, 300]
            ])
        }

        # 批量测试数据
        self.batch_data = [self.test_data] * 10

    def test_feature_registration(self):
        """测试特征注册"""
        # 检查高频特征
        self.assertIn("HF_MOMENTUM", self.engine.feature_registry)
        self.assertIn("ORDER_FLOW_IMBALANCE", self.engine.feature_registry)
        self.assertIn("INSTANT_VOLATILITY", self.engine.feature_registry)

    def test_hf_momentum_calculation(self):
        """测试高频动量计算"""
        momentum = self.optimizer.calculate_hf_momentum(
            self.test_data["price"],
            self.test_data["volume"]
        )
        self.assertEqual(len(momentum), len(self.test_data["price"]))
        self.assertTrue(np.all(momentum[:10] == 0))  # 前10个应为0

    def test_order_flow_imbalance(self):
        """测试订单流不平衡计算"""
        imbalance = self.optimizer.calculate_order_flow_imbalance(self.test_data)
        self.assertTrue(-1 <= imbalance <= 1)

    def test_instant_volatility(self):
        """测试瞬时波动率计算"""
        volatility = self.optimizer.calculate_instant_volatility(self.test_data["price"])
        self.assertGreaterEqual(volatility, 0)

    def test_batch_processing(self):
        """测试批量特征计算"""
        batch_results = self.optimizer.batch_calculate_features(self.batch_data)

        # 检查批量结果
        self.assertIn("HF_MOMENTUM", batch_results)
        self.assertIn("ORDER_FLOW_IMBALANCE", batch_results)
        self.assertIn("INSTANT_VOLATILITY", batch_results)

        self.assertEqual(len(batch_results["HF_MOMENTUM"]), len(self.batch_data))

    def test_integration_with_level2(self):
        """测试与Level2分析器的集成"""
        # 创建Level2分析器
        level2_analyzer = Level2Analyzer(self.engine)

        # 计算组合特征
        level2_features = level2_analyzer.calculate_all_features(self.test_data)
        hf_features = self.optimizer.batch_calculate_features([self.test_data])

        # 检查特征组合
        self.assertIn("HIGH_FREQ_PRESSURE", level2_features)
        self.assertIn("HF_MOMENTUM", hf_features)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch = [self.test_data] * 1000

        start = time.time()
        self.optimizer.batch_calculate_features(large_batch)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次计算应在1秒内完成
        print(f"高频特征批量处理性能: {1000/elapsed:.2f} 次/秒")

class TestCppHighFreqOptimizer(unittest.TestCase):
    """C++高频特征优化器测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()
        config = HighFreqConfig()
        self.optimizer = HighFreqOptimizer(self.engine, config)

    def test_cpp_fallback(self):
        """测试C++降级处理"""
        # 当C++扩展不可用时应自动降级到Python处理
        features = self.optimizer.batch_calculate_features([{
            "price": np.array([10.0, 10.1]),
            "volume": np.array([100, 150]),
            "bid": np.array([[9.99, 200]]),
            "ask": np.array([[10.01, 100]])
        }])
        self.assertIn("HF_MOMENTUM", features)

if __name__ == '__main__':
    unittest.main()
