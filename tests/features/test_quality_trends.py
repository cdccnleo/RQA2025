"""
测试质量趋势分析
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.quality import (
    QualityTrendAnalyzer,
    TrendDirection,
    analyze_quality_trends,
    get_trend_analyzer
)


class TestQualityTrendAnalyzer(unittest.TestCase):
    """测试质量趋势分析器"""

    def setUp(self):
        """测试前准备"""
        self.analyzer = QualityTrendAnalyzer()

    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 少于3个数据点
        history = [
            {'recorded_at': datetime.now(), 'quality_score': 0.9},
            {'recorded_at': datetime.now() + timedelta(days=1), 'quality_score': 0.85}
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")
        self.assertIsNone(result)

    def test_stable_trend(self):
        """测试稳定趋势"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.85 + (i * 0.001)}
            for i in range(5)
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")

        self.assertIsNotNone(result)
        self.assertEqual(result.trend, TrendDirection.STABLE)
        self.assertLess(result.trend_strength, 0.5)

    def test_improving_trend(self):
        """测试改善趋势"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.70 + (i * 0.05)}
            for i in range(5)
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")

        self.assertIsNotNone(result)
        self.assertEqual(result.trend, TrendDirection.IMPROVING)
        self.assertGreater(result.change_percent, 0)

    def test_declining_trend(self):
        """测试下降趋势"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.90 - (i * 0.05)}
            for i in range(5)
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")

        self.assertIsNotNone(result)
        self.assertEqual(result.trend, TrendDirection.DECLINING)
        self.assertLess(result.change_percent, 0)

    def test_anomaly_detection(self):
        """测试异常检测"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': score}
            for i, score in enumerate([0.85, 0.86, 0.84, 0.95, 0.85])  # 0.95是异常值
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")

        self.assertIsNotNone(result)
        self.assertGreater(len(result.anomalies), 0)

    def test_statistics_calculation(self):
        """测试统计信息计算"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.80 + (i * 0.02)}
            for i in range(5)
        ]

        result = self.analyzer.analyze_trends(history, "SMA_5")

        self.assertIsNotNone(result)
        self.assertIn('count', result.statistics)
        self.assertIn('mean', result.statistics)
        self.assertIn('std', result.statistics)
        self.assertEqual(result.statistics['count'], 5)

    def test_volatility_calculation(self):
        """测试波动性计算"""
        base_time = datetime.now()

        # 低波动性数据
        stable_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.85 + (i * 0.001)}
            for i in range(5)
        ]
        stable_result = self.analyzer.analyze_trends(stable_history, "SMA_5")

        # 高波动性数据
        volatile_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': score}
            for i, score in enumerate([0.70, 0.90, 0.75, 0.95, 0.80])
        ]
        volatile_result = self.analyzer.analyze_trends(volatile_history, "RSI")

        self.assertIsNotNone(stable_result)
        self.assertIsNotNone(volatile_result)
        self.assertLess(stable_result.volatility, volatile_result.volatility)

    def test_trend_report_generation(self):
        """测试趋势报告生成"""
        base_time = datetime.now()

        # 创建多个分析结果
        analyses = []

        # 稳定趋势
        stable_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.85}
            for i in range(5)
        ]
        analyses.append(self.analyzer.analyze_trends(stable_history, "SMA_5"))

        # 下降趋势
        declining_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.90 - (i * 0.05)}
            for i in range(5)
        ]
        analyses.append(self.analyzer.analyze_trends(declining_history, "RSI"))

        report = self.analyzer.generate_trend_report(analyses)

        self.assertIn('summary', report)
        self.assertIn('total_features', report)
        self.assertIn('trend_distribution', report)
        self.assertIn('concerning_features', report)
        self.assertIn('recommendations', report)
        self.assertEqual(report['total_features'], 2)

    def test_concerning_features_identification(self):
        """测试关注特征识别"""
        base_time = datetime.now()

        # 低质量特征
        low_quality_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.60}
            for i in range(5)
        ]
        low_quality_analysis = self.analyzer.analyze_trends(low_quality_history, "OBV")

        # 下降趋势特征
        declining_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.90 - (i * 0.08)}
            for i in range(5)
        ]
        declining_analysis = self.analyzer.analyze_trends(declining_history, "KDJ")

        analyses = [low_quality_analysis, declining_analysis]
        report = self.analyzer.generate_trend_report(analyses)

        # 应该识别出2个需要关注的特征
        self.assertEqual(len(report['concerning_features']), 2)

    def test_recommendations_generation(self):
        """测试建议生成"""
        base_time = datetime.now()

        # 创建多个分析结果，包含各种问题
        analyses = []

        # 下降趋势
        declining_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.90 - (i * 0.05)}
            for i in range(5)
        ]
        analyses.append(self.analyzer.analyze_trends(declining_history, "SMA_5"))

        # 低质量
        low_quality_history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.60}
            for i in range(5)
        ]
        analyses.append(self.analyzer.analyze_trends(low_quality_history, "RSI"))

        report = self.analyzer.generate_trend_report(analyses)

        # 应该生成相应的建议
        self.assertGreater(len(report['recommendations']), 0)

    def test_empty_data_handling(self):
        """测试空数据处理"""
        report = self.analyzer.generate_trend_report([])

        self.assertEqual(report['total_features'], 0)
        self.assertEqual(report['summary'], '无数据')


class TestAnalyzeQualityTrends(unittest.TestCase):
    """测试便捷函数"""

    def test_analyze_quality_trends(self):
        """测试analyze_quality_trends便捷函数"""
        base_time = datetime.now()
        history = [
            {'recorded_at': base_time + timedelta(days=i), 'quality_score': 0.85 + (i * 0.01)}
            for i in range(5)
        ]

        result = analyze_quality_trends(history, "EMA_10")

        self.assertIsNotNone(result)
        self.assertEqual(result.feature_name, "EMA_10")


if __name__ == '__main__':
    unittest.main()
