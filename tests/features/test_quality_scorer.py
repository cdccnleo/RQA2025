"""
测试质量评分工具
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.quality.quality_scorer import (
    get_feature_quality_score,
    get_quality_category,
    calculate_quality_scores,
    FEATURE_QUALITY_MAP
)


class TestQualityScorer(unittest.TestCase):
    """测试质量评分工具"""

    def test_trend_indicators(self):
        """测试趋势类指标评分"""
        self.assertEqual(get_feature_quality_score("SMA_5"), 0.90)
        self.assertEqual(get_feature_quality_score("SMA_10"), 0.90)
        self.assertEqual(get_feature_quality_score("EMA_5"), 0.90)
        self.assertEqual(get_feature_quality_score("EMA_20"), 0.90)
        self.assertEqual(get_feature_quality_score("WMA_5"), 0.88)

    def test_momentum_indicators(self):
        """测试动量类指标评分"""
        self.assertEqual(get_feature_quality_score("RSI"), 0.85)
        self.assertEqual(get_feature_quality_score("RSI_14"), 0.85)
        self.assertEqual(get_feature_quality_score("MACD"), 0.85)
        self.assertEqual(get_feature_quality_score("CCI"), 0.83)

    def test_volatility_indicators(self):
        """测试波动率类指标评分"""
        self.assertEqual(get_feature_quality_score("BOLL_upper"), 0.80)
        self.assertEqual(get_feature_quality_score("BOLL_middle"), 0.80)
        self.assertEqual(get_feature_quality_score("BOLL_lower"), 0.80)
        self.assertEqual(get_feature_quality_score("ATR"), 0.80)

    def test_complex_indicators(self):
        """测试复杂指标评分"""
        self.assertEqual(get_feature_quality_score("KDJ_K"), 0.82)
        self.assertEqual(get_feature_quality_score("KDJ_D"), 0.82)
        self.assertEqual(get_feature_quality_score("KDJ_J"), 0.82)
        self.assertEqual(get_feature_quality_score("STOCH"), 0.81)

    def test_volume_indicators(self):
        """测试成交量类指标评分"""
        self.assertEqual(get_feature_quality_score("OBV"), 0.78)
        self.assertEqual(get_feature_quality_score("VWAP"), 0.78)
        self.assertEqual(get_feature_quality_score("ADL"), 0.77)

    def test_default_score(self):
        """测试默认评分"""
        self.assertEqual(get_feature_quality_score("UNKNOWN"), 0.80)
        self.assertEqual(get_feature_quality_score("CUSTOM"), 0.80)
        self.assertEqual(get_feature_quality_score("XYZ_123"), 0.80)

    def test_quality_category(self):
        """测试质量等级分类"""
        self.assertEqual(get_quality_category(0.95), "优秀")
        self.assertEqual(get_quality_category(0.90), "优秀")
        self.assertEqual(get_quality_category(0.85), "良好")
        self.assertEqual(get_quality_category(0.80), "良好")
        self.assertEqual(get_quality_category(0.75), "一般")
        self.assertEqual(get_quality_category(0.70), "一般")
        self.assertEqual(get_quality_category(0.65), "较差")
        self.assertEqual(get_quality_category(0.60), "较差")
        self.assertEqual(get_quality_category(0.55), "差")
        self.assertEqual(get_quality_category(0.50), "差")

    def test_calculate_quality_scores(self):
        """测试批量计算评分"""
        features = ["SMA_5", "RSI", "BOLL_upper", "UNKNOWN"]
        scores = calculate_quality_scores(features)

        self.assertEqual(len(scores), 4)
        self.assertEqual(scores["SMA_5"], 0.90)
        self.assertEqual(scores["RSI"], 0.85)
        self.assertEqual(scores["BOLL_upper"], 0.80)
        self.assertEqual(scores["UNKNOWN"], 0.80)

    def test_case_insensitive(self):
        """测试大小写不敏感"""
        self.assertEqual(get_feature_quality_score("sma_5"), 0.90)
        self.assertEqual(get_feature_quality_score("Sma_5"), 0.90)
        self.assertEqual(get_feature_quality_score("rsi"), 0.85)
        self.assertEqual(get_feature_quality_score("Rsi"), 0.85)

    def test_feature_quality_map_completeness(self):
        """测试评分映射表的完整性"""
        # 确保所有必要的键都存在
        self.assertIn('SMA', FEATURE_QUALITY_MAP)
        self.assertIn('EMA', FEATURE_QUALITY_MAP)
        self.assertIn('RSI', FEATURE_QUALITY_MAP)
        self.assertIn('MACD', FEATURE_QUALITY_MAP)
        self.assertIn('BOLL', FEATURE_QUALITY_MAP)
        self.assertIn('KDJ', FEATURE_QUALITY_MAP)
        self.assertIn('DEFAULT', FEATURE_QUALITY_MAP)

        # 确保评分在合理范围内
        for key, score in FEATURE_QUALITY_MAP.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestQualityDistribution(unittest.TestCase):
    """测试质量分布统计"""

    def test_distribution_calculation(self):
        """测试分布计算"""
        features = [
            "SMA_5",      # 0.90 - 优秀
            "EMA_10",     # 0.90 - 优秀
            "RSI",        # 0.85 - 良好
            "MACD",       # 0.85 - 良好
            "BOLL_upper", # 0.80 - 良好
            "KDJ_K",      # 0.82 - 良好
            "OBV",        # 0.78 - 一般
            "UNKNOWN",    # 0.80 - 良好
        ]

        scores = calculate_quality_scores(features)

        # 统计各等级的数量
        categories = {}
        for score in scores.values():
            category = get_quality_category(score)
            categories[category] = categories.get(category, 0) + 1

        # 验证分布
        self.assertEqual(categories.get("优秀", 0), 2)  # SMA, EMA
        self.assertEqual(categories.get("良好", 0), 5)  # RSI, MACD, BOLL, KDJ, UNKNOWN
        self.assertEqual(categories.get("一般", 0), 1)  # OBV


if __name__ == '__main__':
    unittest.main()
