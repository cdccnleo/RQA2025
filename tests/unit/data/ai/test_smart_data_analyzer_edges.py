import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest

from src.data.ai.smart_data_analyzer import SmartDataAnalyzer, SummaryStats


def test_summarize_empty_list():
    """测试空列表的汇总统计"""
    analyzer = SmartDataAnalyzer()
    
    stats = analyzer.summarize([])
    
    assert stats.count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0
    assert stats.minimum == 0.0
    assert stats.maximum == 0.0


def test_summarize_single_value():
    """测试单值列表的汇总统计"""
    analyzer = SmartDataAnalyzer()
    
    stats = analyzer.summarize([42.0])
    
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.std == 0.0
    assert stats.minimum == 42.0
    assert stats.maximum == 42.0


def test_summarize_non_list_raises():
    """测试非列表输入抛出异常"""
    analyzer = SmartDataAnalyzer()
    
    with pytest.raises(TypeError, match="series must be a list"):
        analyzer.summarize("not a list")
    
    with pytest.raises(TypeError, match="series must be a list"):
        analyzer.summarize(123)


def test_summarize_multiple_values():
    """测试多值列表的汇总统计"""
    analyzer = SmartDataAnalyzer()
    
    stats = analyzer.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert stats.count == 5
    assert stats.mean == 3.0
    assert stats.minimum == 1.0
    assert stats.maximum == 5.0
    assert stats.std > 0.0


def test_detect_outliers_empty_list():
    """测试空列表的异常检测"""
    analyzer = SmartDataAnalyzer()
    
    outliers = analyzer.detect_outliers([])
    
    assert outliers == []


def test_detect_outliers_zero_std():
    """测试标准差为0的异常检测"""
    analyzer = SmartDataAnalyzer()
    
    # 所有值相同，标准差为0
    outliers = analyzer.detect_outliers([1.0, 1.0, 1.0])
    
    assert outliers == []


def test_detect_outliers_with_outliers():
    """测试包含异常值的异常检测"""
    analyzer = SmartDataAnalyzer()
    
    # 使用较小的阈值来确保能检测到异常值
    # 正常值在 1-5 范围内，异常值 100
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    
    # 使用较小的阈值（1.0）来确保能检测到
    outliers = analyzer.detect_outliers(series, z_threshold=1.0)
    
    # 100.0 应该是异常值（使用较小阈值）
    assert len(outliers) > 0
    assert 5 in outliers  # 索引5是100.0


def test_detect_outliers_custom_threshold():
    """测试自定义阈值的异常检测"""
    analyzer = SmartDataAnalyzer()
    
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    
    # 使用较小的阈值
    outliers_low = analyzer.detect_outliers(series, z_threshold=1.0)
    
    # 使用较大的阈值
    outliers_high = analyzer.detect_outliers(series, z_threshold=5.0)
    
    # 较小阈值应该检测到更多异常值
    assert len(outliers_low) >= len(outliers_high)


def test_detect_outliers_no_outliers():
    """测试没有异常值的情况"""
    analyzer = SmartDataAnalyzer()
    
    # 所有值都在正常范围内
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    outliers = analyzer.detect_outliers(series, z_threshold=3.0)
    
    assert outliers == []


def test_compute_trend_insufficient_data():
    """测试数据不足时的趋势计算"""
    analyzer = SmartDataAnalyzer()
    
    # 空列表
    trend = analyzer.compute_trend([])
    assert trend == "flat"
    
    # 单值
    trend = analyzer.compute_trend([1.0])
    assert trend == "flat"


def test_compute_trend_upward():
    """测试上升趋势"""
    analyzer = SmartDataAnalyzer()
    
    # 左半段均值 < 右半段均值
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    trend = analyzer.compute_trend(series)
    
    assert trend == "up"


def test_compute_trend_downward():
    """测试下降趋势"""
    analyzer = SmartDataAnalyzer()
    
    # 左半段均值 > 右半段均值
    series = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    
    trend = analyzer.compute_trend(series)
    
    assert trend == "down"


def test_compute_trend_flat():
    """测试平稳趋势"""
    analyzer = SmartDataAnalyzer()
    
    # 左半段均值 ≈ 右半段均值
    series = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    
    trend = analyzer.compute_trend(series)
    
    assert trend == "flat"


def test_compute_trend_odd_length():
    """测试奇数长度列表的趋势计算"""
    analyzer = SmartDataAnalyzer()
    
    # 奇数长度：7个值，mid = 3
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    trend = analyzer.compute_trend(series)
    
    # 应该能计算趋势
    assert trend in ["up", "down", "flat"]


def test_compute_trend_edge_case_small_difference():
    """测试趋势边界情况（差值很小）"""
    analyzer = SmartDataAnalyzer()
    
    # 差值小于 eps (1e-9)
    series = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    
    trend = analyzer.compute_trend(series)
    
    # 应该返回 "flat"（因为差值小于 eps）
    assert trend == "flat"


def test_analyze_combines_all_methods():
    """测试 analyze 方法组合所有分析"""
    analyzer = SmartDataAnalyzer()
    
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    
    result = analyzer.analyze(series)
    
    assert "stats" in result
    assert "trend" in result
    assert "outliers" in result
    assert isinstance(result["stats"], SummaryStats)
    assert isinstance(result["trend"], str)
    assert isinstance(result["outliers"], list)


def test_analyze_custom_threshold():
    """测试 analyze 方法使用自定义阈值"""
    analyzer = SmartDataAnalyzer()
    
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    
    result_low = analyzer.analyze(series, z_threshold=1.0)
    result_high = analyzer.analyze(series, z_threshold=5.0)
    
    # 较小阈值应该检测到更多异常值
    assert len(result_low["outliers"]) >= len(result_high["outliers"])


def test_summarize_negative_values():
    """测试负值的汇总统计"""
    analyzer = SmartDataAnalyzer()
    
    stats = analyzer.summarize([-5.0, -2.0, 0.0, 2.0, 5.0])
    
    assert stats.count == 5
    assert stats.mean == 0.0
    assert stats.minimum == -5.0
    assert stats.maximum == 5.0


def test_detect_outliers_negative_outliers():
    """测试负异常值的检测"""
    analyzer = SmartDataAnalyzer()
    
    # 使用较小的阈值来确保能检测到异常值
    # 正常值在 1-5 范围内，负异常值 -100
    series = [1.0, 2.0, 3.0, 4.0, 5.0, -100.0]
    
    # 使用较小的阈值（1.0）来确保能检测到
    outliers = analyzer.detect_outliers(series, z_threshold=1.0)
    
    # -100.0 应该是异常值（使用较小阈值）
    assert len(outliers) > 0
    assert 5 in outliers  # 索引5是-100.0


def test_compute_trend_single_point_difference():
    """测试趋势计算（单点差异）"""
    analyzer = SmartDataAnalyzer()
    
    # 只有2个值
    trend = analyzer.compute_trend([1.0, 2.0])
    
    # 应该能计算趋势
    assert trend in ["up", "down", "flat"]


def test_summarize_float_precision():
    """测试浮点数精度的汇总统计"""
    analyzer = SmartDataAnalyzer()
    
    stats = analyzer.summarize([1.1, 2.2, 3.3, 4.4, 5.5])
    
    assert stats.count == 5
    assert abs(stats.mean - 3.3) < 0.01
    assert stats.minimum == 1.1
    assert stats.maximum == 5.5


def test_detect_outliers_all_outliers():
    """测试所有值都是异常值的情况"""
    analyzer = SmartDataAnalyzer()
    
    # 所有值都相同，但有一个明显不同的值
    series = [1.0, 1.0, 1.0, 1.0, 100.0]
    
    outliers = analyzer.detect_outliers(series, z_threshold=2.0)
    
    # 100.0 应该是异常值
    assert len(outliers) > 0


def test_compute_trend_equal_halves():
    """测试左右半段均值相等的情况"""
    analyzer = SmartDataAnalyzer()
    
    # 左半段和右半段均值相等
    series = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    
    trend = analyzer.compute_trend(series)
    
    # 应该返回 "flat"
    assert trend == "flat"

