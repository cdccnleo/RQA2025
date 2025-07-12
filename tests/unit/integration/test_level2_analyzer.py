"""Level2行情分析器集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from src.features.feature_engine import FeatureEngine
from src.features.level2_analyzer import Level2Analyzer, Level2Config

class TestLevel2AnalyzerIntegration(unittest.TestCase):
    """Level2行情分析器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建Level2分析器
        config = Level2Config(depth=5, window_size=10, tick_buffer_size=1000)
        self.analyzer = Level2Analyzer(self.engine, config)

        # 测试数据
        self.test_order_book = {
            "bid": np.array([
                [10.0, 100], [9.99, 200], [9.98, 150], [9.97, 300], [9.96, 250]
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
            ]),
            "upper_limit": 11.0,  # A股特有数据
            "lower_limit": 9.0
        }

        # 填充tick缓冲区
        for i in range(20):
            direction = 1 if i % 2 == 0 else -1
            self.analyzer.update_tick_buffer({
                "price": 10.0 + i*0.01,
                "volume": 100 + i*10,
                "timestamp": i,
                "direction": "buy" if direction > 0 else "sell"
            })

    def test_feature_registration(self):
        """测试特征注册"""
        # 检查Level2特征
        self.assertIn("HIGH_FREQ_PRESSURE", self.engine.feature_registry)
        self.assertIn("INSTANT_LIQUIDITY", self.engine.feature_registry)
        self.assertIn("LARGE_ORDER_TRACKING", self.engine.feature_registry)

    def test_high_freq_pressure(self):
        """测试高频买卖压力计算"""
        pressure = self.analyzer.calculate_high_freq_pressure()
        self.assertTrue(-1 <= pressure <= 1)

    def test_instant_liquidity(self):
        """测试瞬时流动性计算"""
        liquidity = self.analyzer.calculate_instant_liquidity(self.test_order_book)
        self.assertGreater(liquidity, 0)

    def test_large_order_tracking(self):
        """测试大单追踪"""
        large_orders = self.analyzer.track_large_orders(self.test_order_book["trade"])
        self.assertIn("large_buy_vol", large_orders)
        self.assertIn("large_sell_vol", large_orders)

    def test_a_share_features(self):
        """测试A股特有特征"""
        features = self.analyzer.calculate_a_share_features(self.test_order_book)
        self.assertIn("limit_order_pressure", features)

    def test_integration_with_order_book(self):
        """测试与订单簿分析器的集成"""
        features = self.analyzer.calculate_all_features(self.test_order_book)

        # 检查订单簿特征
        self.assertIn("ORDER_BOOK_IMBALANCE", features)

        # 检查Level2特征
        self.assertIn("HIGH_FREQ_PRESSURE", features)
        self.assertIn("INSTANT_LIQUIDITY", features)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        start = time.time()
        for _ in range(1000):
            self.analyzer.calculate_all_features(self.test_order_book)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次计算应在1秒内完成
        print(f"Level2分析性能: {1000/elapsed:.2f} 次/秒")

class TestCppLevel2Analyzer(unittest.TestCase):
    """C++ Level2分析器测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()
        config = Level2Config()
        self.analyzer = Level2Analyzer(self.engine, config)

    def test_cpp_fallback(self):
        """测试C++降级处理"""
        # 当C++扩展不可用时应自动降级到Python处理
        features = self.analyzer.calculate_all_features({
            "bid": np.array([[10.0, 100]]),
            "ask": np.array([[10.01, 100]])
        })
        self.assertIn("HIGH_FREQ_PRESSURE", features)

if __name__ == '__main__':
    unittest.main()
