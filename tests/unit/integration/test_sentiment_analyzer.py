"""情感分析器集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from src.features.feature_engine import FeatureEngine
from src.features.sentiment_analyzer import SentimentAnalyzer, SentimentConfig

class TestSentimentAnalyzerIntegration(unittest.TestCase):
    """情感分析器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 配置政策关键词和行业术语
        config = SentimentConfig(
            policy_keywords=["货币政策", "财政政策", "碳中和"],
            industry_terms={
                "新能源": ["光伏", "风电", "锂电池"],
                "科技": ["半导体", "芯片", "人工智能"]
            }
        )

        # 创建情感分析器
        self.analyzer = SentimentAnalyzer(self.engine, config)

        # 测试数据
        self.test_texts = [
            "央行发布新的货币政策，利好市场流动性",
            "财政政策加码，基建投资有望提速",
            "碳中和目标推动新能源行业发展",
            "半导体行业面临技术封锁挑战"
        ]

    def test_sentiment_feature_registration(self):
        """测试情感特征注册"""
        # 检查基础情感特征
        self.assertIn("SENTIMENT_SCORE", self.engine.feature_registry)

        # 检查政策情感特征
        self.assertIn("POLICY_SENTIMENT", self.engine.feature_registry)

        # 检查行业情感特征
        self.assertIn("新能源_SENTIMENT", self.engine.feature_registry)
        self.assertIn("科技_SENTIMENT", self.engine.feature_registry)

    def test_basic_sentiment_analysis(self):
        """测试基础情感分析"""
        scores = self.analyzer.calculate_sentiment_features(self.test_texts)

        # 检查返回结构
        self.assertIn("SENTIMENT_SCORE", scores)
        self.assertEqual(len(scores["SENTIMENT_SCORE"]), len(self.test_texts))

        # 检查情感分数范围
        for score in scores["SENTIMENT_SCORE"]:
            self.assertTrue(0 <= score <= 1)

    def test_policy_sentiment_analysis(self):
        """测试政策情感分析"""
        scores = self.analyzer.calculate_sentiment_features(self.test_texts)

        # 检查政策情感分数
        self.assertIn("POLICY_SENTIMENT", scores)

        # 验证包含政策关键词的文本有情感分数
        self.assertNotEqual(scores["POLICY_SENTIMENT"][0], 0)  # 货币政策
        self.assertNotEqual(scores["POLICY_SENTIMENT"][1], 0)  # 财政政策
        self.assertNotEqual(scores["POLICY_SENTIMENT"][2], 0)  # 碳中和

    def test_industry_sentiment_analysis(self):
        """测试行业情感分析"""
        scores = self.analyzer.calculate_sentiment_features(self.test_texts)

        # 检查行业情感分数
        self.assertIn("新能源_SENTIMENT", scores)
        self.assertIn("科技_SENTIMENT", scores)

        # 验证新能源行业文本有情感分数
        self.assertNotEqual(scores["新能源_SENTIMENT"][2], 0)  # 碳中和相关

        # 验证科技行业文本有情感分数
        self.assertNotEqual(scores["科技_SENTIMENT"][3], 0)  # 半导体相关

    def test_a_share_specific_features(self):
        """测试A股特有情感特征"""
        # 检查特征配置中的A股标记
        self.assertTrue(self.engine.feature_registry["POLICY_SENTIMENT"].a_share_specific)
        self.assertTrue(self.engine.feature_registry["新能源_SENTIMENT"].a_share_specific)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量文本测试
        large_texts = ["这是一条测试文本" + str(i) for i in range(1000)]

        # 测试情感分析性能
        start = time.time()
        scores = self.analyzer.calculate_sentiment_features(large_texts)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 5.0)  # 1000条文本应在5秒内完成
        print(f"情感分析性能: {len(large_texts)/elapsed:.2f} 条/秒")

class TestFpgaSentimentAnalyzer(unittest.TestCase):
    """FPGA情感分析器测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()
        config = SentimentConfig()
        self.analyzer = SentimentAnalyzer(self.engine, config)

    def test_fpga_fallback(self):
        """测试FPGA降级处理"""
        # 当FPGA不可用时应自动降级到CPU处理
        scores = self.analyzer.calculate_sentiment_features(["测试文本"])
        self.assertIn("SENTIMENT_SCORE", scores)

if __name__ == '__main__':
    unittest.main()
