"""
测试数据质量检查工具
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.quality import (
    DataQualityChecker,
    DataQualityMetrics,
    calculate_final_quality_score,
    calculate_quality_scores_with_data_quality,
    get_quality_monitor,
    QualityAlert
)


class TestDataQualityChecker(unittest.TestCase):
    """测试数据质量检查器"""

    def setUp(self):
        """测试前准备"""
        self.checker = DataQualityChecker()
        self.checker.clear_cache()

    def test_perfect_data(self):
        """测试完美数据"""
        # 创建无缺失、无异常的数据
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        metrics = self.checker.check_data_quality(data)

        self.assertEqual(metrics.completeness, 1.0)
        self.assertEqual(metrics.stability, 1.0)
        self.assertEqual(metrics.calculation_success, 1.0)
        self.assertEqual(metrics.overall_factor, 1.0)

    def test_data_with_missing_values(self):
        """测试有缺失值的数据"""
        # 创建有缺失值的数据（10%缺失）
        data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10])

        metrics = self.checker.check_data_quality(data)

        # 缺失率10%，超过5%阈值，应该被惩罚
        self.assertLess(metrics.completeness, 1.0)
        self.assertGreater(metrics.completeness, 0.0)

    def test_data_with_outliers(self):
        """测试有异常值的数据"""
        # 创建有异常值的数据
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100是异常值

        metrics = self.checker.check_data_quality(data)

        # 有异常值，稳定性应该下降
        self.assertLess(metrics.stability, 1.0)

    def test_empty_data(self):
        """测试空数据"""
        data = pd.Series([])

        metrics = self.checker.check_data_quality(data)

        self.assertEqual(metrics.completeness, 0.0)
        self.assertEqual(metrics.calculation_success, 0.0)
        self.assertEqual(metrics.overall_factor, 0.0)

    def test_time_coverage(self):
        """测试时间覆盖率"""
        # 创建带时间索引的数据
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.Series(range(10), index=dates)

        # 测试完全覆盖
        metrics = self.checker.check_data_quality(
            data,
            expected_start_date=datetime(2024, 1, 1),
            expected_end_date=datetime(2024, 1, 10)
        )
        self.assertEqual(metrics.time_coverage, 1.0)

        # 测试部分覆盖
        metrics = self.checker.check_data_quality(
            data,
            expected_start_date=datetime(2024, 1, 1),
            expected_end_date=datetime(2024, 1, 20)
        )
        self.assertLess(metrics.time_coverage, 1.0)

    def test_cache_functionality(self):
        """测试缓存功能"""
        data = pd.Series([1, 2, 3, 4, 5])

        # 第一次检查
        metrics1 = self.checker.check_data_quality(data, use_cache=True)

        # 第二次检查（应该使用缓存）
        metrics2 = self.checker.check_data_quality(data, use_cache=True)

        # 结果应该相同
        self.assertEqual(metrics1.overall_factor, metrics2.overall_factor)

    def test_overall_factor_calculation(self):
        """测试综合质量因子计算"""
        # 创建中等质量的数据
        data = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 100])

        metrics = self.checker.check_data_quality(data)

        # 验证综合因子在合理范围内
        self.assertGreaterEqual(metrics.overall_factor, 0.0)
        self.assertLessEqual(metrics.overall_factor, 1.0)

        # 验证综合因子是加权平均
        expected_factor = (
            metrics.completeness * 0.30 +
            metrics.stability * 0.25 +
            metrics.calculation_success * 0.25 +
            metrics.time_coverage * 0.20
        )
        self.assertAlmostEqual(metrics.overall_factor, round(expected_factor, 3))


class TestFinalQualityScore(unittest.TestCase):
    """测试最终质量评分计算"""

    def test_calculate_final_quality_score(self):
        """测试最终质量评分计算"""
        # SMA的基础评分是0.90
        score = calculate_final_quality_score("SMA_5", 1.0)
        self.assertEqual(score, 0.90)

        # 质量因子0.95
        score = calculate_final_quality_score("SMA_5", 0.95)
        self.assertEqual(score, 0.855)  # 0.90 * 0.95 = 0.855

        # 质量因子0.88
        score = calculate_final_quality_score("RSI", 0.88)
        self.assertEqual(score, 0.748)  # 0.85 * 0.88 = 0.748

    def test_quality_factor_bounds(self):
        """测试质量因子边界"""
        # 质量因子超过1.0应该被截断
        score = calculate_final_quality_score("SMA_5", 1.5)
        self.assertEqual(score, 0.90)  # 0.90 * 1.0 = 0.90

        # 质量因子低于0应该被截断
        score = calculate_final_quality_score("SMA_5", -0.5)
        self.assertEqual(score, 0.0)  # 0.90 * 0.0 = 0.0

    def test_calculate_quality_scores_with_data_quality(self):
        """测试批量计算质量评分（含数据质量因子）"""
        features = ["SMA_5", "RSI", "BOLL_upper"]

        # 创建模拟的数据质量指标
        mock_metrics = {
            "SMA_5": type('MockMetrics', (), {'overall_factor': 0.95})(),
            "RSI": type('MockMetrics', (), {'overall_factor': 0.88})(),
            "BOLL_upper": type('MockMetrics', (), {'overall_factor': 1.0})()
        }

        scores = calculate_quality_scores_with_data_quality(features, mock_metrics)

        # 验证计算结果
        self.assertEqual(scores["SMA_5"], 0.855)  # 0.90 * 0.95
        self.assertEqual(scores["RSI"], 0.748)    # 0.85 * 0.88
        self.assertEqual(scores["BOLL_upper"], 0.80)  # 0.80 * 1.0

    def test_calculate_quality_scores_without_data_quality(self):
        """测试批量计算质量评分（无数据质量因子）"""
        features = ["SMA_5", "RSI"]

        scores = calculate_quality_scores_with_data_quality(features, None)

        # 没有数据质量因子，应该使用基础评分
        self.assertEqual(scores["SMA_5"], 0.90)
        self.assertEqual(scores["RSI"], 0.85)


class TestQualityMonitor(unittest.TestCase):
    """测试质量监控器"""

    def setUp(self):
        """测试前准备"""
        self.monitor = get_quality_monitor()
        self.monitor.clear_history()

    def test_quality_drop_alert(self):
        """测试质量下降告警"""
        # 质量下降15%（超过10%阈值）
        alert = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="SMA_5",
            old_score=0.90,
            new_score=0.75
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, 'quality_drop')
        self.assertEqual(alert.severity, 'warning')

    def test_critical_quality_drop_alert(self):
        """测试严重质量下降告警"""
        # 质量下降30%（超过20%严重阈值），但新评分仍在正常范围(>=0.7)
        alert = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="SMA_5",
            old_score=1.0,
            new_score=0.70
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, 'quality_drop')
        self.assertEqual(alert.severity, 'critical')

    def test_low_quality_alert(self):
        """测试低质量告警"""
        alert = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="RSI",
            old_score=0.85,
            new_score=0.65
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, 'low_quality')
        self.assertEqual(alert.severity, 'warning')

    def test_critical_low_quality_alert(self):
        """测试严重低质量告警"""
        alert = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="RSI",
            old_score=0.85,
            new_score=0.45
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, 'low_quality')
        self.assertEqual(alert.severity, 'critical')

    def test_no_alert_for_good_quality(self):
        """测试高质量无告警"""
        alert = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="SMA_5",
            old_score=0.90,
            new_score=0.88
        )

        self.assertIsNone(alert)

    def test_alert_cooldown(self):
        """测试告警冷却期"""
        # 第一次触发告警
        alert1 = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="SMA_5",
            old_score=0.90,
            new_score=0.75
        )
        self.assertIsNotNone(alert1)

        # 立即再次触发（应该在冷却期内）
        alert2 = self.monitor.check_quality_change(
            feature_id=1,
            feature_name="SMA_5",
            old_score=0.90,
            new_score=0.75
        )
        self.assertIsNone(alert2)  # 冷却期内不生成新告警

    def test_get_recent_alerts(self):
        """测试获取最近告警"""
        # 生成几个告警
        self.monitor.check_quality_change(1, "SMA_5", 0.90, 0.75)
        self.monitor.check_quality_change(2, "RSI", 0.85, 0.65)

        # 获取最近告警
        alerts = self.monitor.get_recent_alerts(hours=1)

        self.assertEqual(len(alerts), 2)

    def test_get_alert_statistics(self):
        """测试获取告警统计"""
        # 生成几个告警
        self.monitor.check_quality_change(1, "SMA_5", 0.90, 0.75)
        self.monitor.check_quality_change(2, "RSI", 0.85, 0.45)

        stats = self.monitor.get_alert_statistics(days=1)

        self.assertEqual(stats['total_alerts'], 2)
        self.assertIn('quality_drop', stats['type_distribution'])
        self.assertIn('low_quality', stats['type_distribution'])


if __name__ == '__main__':
    unittest.main()
