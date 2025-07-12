"""订单簿分析器集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from src.features.feature_engine import FeatureEngine
from src.features.order_book_analyzer import OrderBookAnalyzer, OrderBookConfig

class TestOrderBookAnalyzerIntegration(unittest.TestCase):
    """订单簿分析器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建订单簿分析器
        config = OrderBookConfig(depth=5, window_size=10, a_share_specific=True)
        self.analyzer = OrderBookAnalyzer(self.engine, config)

        # 测试数据
        self.test_order_book = {
            "bid": np.array([
                [10.0, 100], [9.99, 200], [9.98, 150], [9.97, 300], [9.96, 250]
            ]),
            "ask": np.array([
                [10.01, 80], [10.02, 120], [10.03, 200], [10.04, 150], [10.05, 100]
            ]),
            "bid_volume": np.array([100, 200, 150, 300, 250]),
            "ask_volume": np.array([80, 120, 200, 150, 100]),
            "trade": np.array([
                [10.01, 50], [10.0, -100], [10.02, 200], [9.99, -50], [10.01, 300]
            ])
        }

        # A股测试数据
        self.a_share_order_book = {
            "bid": np.array([
                [11.0, 500], [10.99, 1000], [10.98, 800], [10.97, 1200], [10.96, 900]
            ]),
            "ask": np.array([
                [11.1, 400], [11.11, 600], [11.12, 800], [11.13, 500], [11.14, 300]
            ]),
            "trade": np.array([
                [11.05, 200000], [11.04, -150000], [11.06, 300000],
                [11.03, -100000], [11.07, 50000]
            ])
        }

    def test_feature_registration(self):
        """测试特征注册"""
        # 检查基础特征
        self.assertIn("ORDER_BOOK_IMBALANCE", self.engine.feature_registry)
        self.assertIn("LARGE_ORDER_IMPACT", self.engine.feature_registry)

        # 检查A股特有特征
        self.assertIn("PRICE_LIMIT_PRESSURE", self.engine.feature_registry)
        self.assertIn("MAIN_CAPITAL_FLOW", self.engine.feature_registry)

    def test_basic_feature_calculation(self):
        """测试基础特征计算"""
        features = self.analyzer.calculate_all_features(self.test_order_book)

        # 检查订单簿不平衡度
        self.assertIn("ORDER_BOOK_IMBALANCE", features)
        imbalance = features["ORDER_BOOK_IMBALANCE"]["value"]
        self.assertTrue(-1 <= imbalance <= 1)

        # 检查大单冲击成本
        self.assertIn("LARGE_ORDER_IMPACT", features)
        impact = features["LARGE_ORDER_IMPACT"]
        self.assertIn("bid_impact", impact)
        self.assertIn("ask_impact", impact)

    def test_a_share_feature_calculation(self):
        """测试A股特有特征计算"""
        features = self.analyzer.calculate_all_features(
            self.a_share_order_book, prev_close=10.0
        )

        # 检查涨跌停压力
        self.assertIn("PRICE_LIMIT_PRESSURE", features)
        pressure = features["PRICE_LIMIT_PRESSURE"]["value"]
        self.assertTrue(-1 <= pressure <= 1)

        # 检查主力资金流向
        self.assertIn("MAIN_CAPITAL_FLOW", features)
        flow = features["MAIN_CAPITAL_FLOW"]["value"]
        self.assertTrue(-1 <= flow <= 1)

    def test_hidden_liquidity_detection(self):
        """测试隐藏流动性检测"""
        features = self.analyzer.calculate_all_features(self.test_order_book)
        self.assertIn("HIDDEN_LIQUIDITY", features)
        hidden = features["HIDDEN_LIQUIDITY"]["value"]
        self.assertTrue(0 <= hidden <= 1)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量订单簿测试
        large_order_books = [self.test_order_book] * 1000

        start = time.time()
        for ob in large_order_books:
            self.analyzer.calculate_all_features(ob)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次计算应在1秒内完成
        print(f"订单簿分析性能: {1000/elapsed:.2f} 次/秒")

class TestCppOrderBookAnalyzer(unittest.TestCase):
    """C++订单簿分析器测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()
        config = OrderBookConfig()
        self.analyzer = OrderBookAnalyzer(self.engine, config)

    def test_cpp_fallback(self):
        """测试C++降级处理"""
        # 当C++扩展不可用时应自动降级到Python处理
        features = self.analyzer.calculate_all_features({
            "bid": np.array([[10.0, 100]]),
            "ask": np.array([[10.01, 100]])
        })
        self.assertIn("ORDER_BOOK_IMBALANCE", features)

if __name__ == '__main__':
    unittest.main()
