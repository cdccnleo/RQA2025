"""
边界测试：smart_data_analyzer.py
测试边界情况和异常场景
"""
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


def test_smart_data_analyzer_init():
    """测试 SmartDataAnalyzer（初始化）"""
    analyzer = SmartDataAnalyzer()
    
    assert analyzer is not None


def test_summary_stats_init():
    """测试 SummaryStats（初始化）"""
    stats = SummaryStats(count=10, mean=5.0, std=2.0, minimum=1.0, maximum=9.0)
    
    assert stats.count == 10
    assert stats.mean == 5.0
    assert stats.std == 2.0
    assert stats.minimum == 1.0
    assert stats.maximum == 9.0


def test_smart_data_analyzer_summarize_empty():
    """测试 SmartDataAnalyzer（汇总，空列表）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([])
    
    assert result.count == 0
    assert result.mean == 0.0
    assert result.std == 0.0
    assert result.minimum == 0.0
    assert result.maximum == 0.0


def test_smart_data_analyzer_summarize_single():
    """测试 SmartDataAnalyzer（汇总，单个元素）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([5.0])
    
    assert result.count == 1
    assert result.mean == 5.0
    assert result.std == 0.0
    assert result.minimum == 5.0
    assert result.maximum == 5.0


def test_smart_data_analyzer_summarize_multiple():
    """测试 SmartDataAnalyzer（汇总，多个元素）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert result.count == 5
    assert result.mean == 3.0
    assert result.minimum == 1.0
    assert result.maximum == 5.0
    assert result.std > 0


def test_smart_data_analyzer_summarize_negative():
    """测试 SmartDataAnalyzer（汇总，负数）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    
    assert result.count == 6
    assert result.mean == 0.0
    assert result.minimum == -5.0
    assert result.maximum == 5.0


def test_smart_data_analyzer_summarize_zero():
    """测试 SmartDataAnalyzer（汇总，零值）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([0.0, 0.0, 0.0])
    
    assert result.count == 3
    assert result.mean == 0.0
    assert result.std == 0.0


def test_smart_data_analyzer_summarize_large():
    """测试 SmartDataAnalyzer（汇总，大数值）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.summarize([1e10, 2e10, 3e10])
    
    assert result.count == 3
    assert result.mean == 2e10
    assert result.minimum == 1e10
    assert result.maximum == 3e10


def test_smart_data_analyzer_summarize_invalid_type():
    """测试 SmartDataAnalyzer（汇总，无效类型）"""
    analyzer = SmartDataAnalyzer()
    
    with pytest.raises(TypeError, match="series must be a list"):
        analyzer.summarize("not a list")


def test_smart_data_analyzer_detect_outliers_empty():
    """测试 SmartDataAnalyzer（检测异常，空列表）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.detect_outliers([])
    
    assert result == []


def test_smart_data_analyzer_detect_outliers_single():
    """测试 SmartDataAnalyzer（检测异常，单个元素）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.detect_outliers([5.0])
    
    assert result == []


def test_smart_data_analyzer_detect_outliers_no_outliers():
    """测试 SmartDataAnalyzer（检测异常，无异常）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.detect_outliers([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert result == []


def test_smart_data_analyzer_detect_outliers_with_outliers():
    """测试 SmartDataAnalyzer（检测异常，有异常）"""
    analyzer = SmartDataAnalyzer()
    
    # 添加一个明显异常的值，使用较小的阈值确保能检测到
    result = analyzer.detect_outliers([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], z_threshold=2.0)
    
    assert len(result) > 0
    assert 5 in result  # 100.0 的索引


def test_smart_data_analyzer_detect_outliers_custom_threshold():
    """测试 SmartDataAnalyzer（检测异常，自定义阈值）"""
    analyzer = SmartDataAnalyzer()
    
    # 使用较小的阈值
    result = analyzer.detect_outliers([1.0, 2.0, 3.0, 4.0, 5.0], z_threshold=1.0)
    
    assert len(result) > 0


def test_smart_data_analyzer_detect_outliers_zero_std():
    """测试 SmartDataAnalyzer（检测异常，零标准差）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.detect_outliers([5.0, 5.0, 5.0, 5.0])
    
    assert result == []


def test_smart_data_analyzer_compute_trend_empty():
    """测试 SmartDataAnalyzer（计算趋势，空列表）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([])
    
    assert result == "flat"


def test_smart_data_analyzer_compute_trend_single():
    """测试 SmartDataAnalyzer（计算趋势，单个元素）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([5.0])
    
    assert result == "flat"


def test_smart_data_analyzer_compute_trend_up():
    """测试 SmartDataAnalyzer（计算趋势，上升）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    assert result == "up"


def test_smart_data_analyzer_compute_trend_down():
    """测试 SmartDataAnalyzer（计算趋势，下降）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    
    assert result == "down"


def test_smart_data_analyzer_compute_trend_flat():
    """测试 SmartDataAnalyzer（计算趋势，平稳）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    
    assert result == "flat"


def test_smart_data_analyzer_compute_trend_odd_length():
    """测试 SmartDataAnalyzer（计算趋势，奇数长度）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.compute_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert result in ["up", "down", "flat"]


def test_smart_data_analyzer_analyze():
    """测试 SmartDataAnalyzer（分析）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.analyze([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert "stats" in result
    assert "trend" in result
    assert "outliers" in result
    assert isinstance(result["stats"], SummaryStats)
    assert isinstance(result["trend"], str)
    assert isinstance(result["outliers"], list)


def test_smart_data_analyzer_analyze_with_outliers():
    """测试 SmartDataAnalyzer（分析，有异常）"""
    analyzer = SmartDataAnalyzer()
    
    # 使用较小的阈值确保能检测到异常
    result = analyzer.analyze([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], z_threshold=2.0)
    
    assert len(result["outliers"]) > 0


def test_smart_data_analyzer_analyze_custom_threshold():
    """测试 SmartDataAnalyzer（分析，自定义阈值）"""
    analyzer = SmartDataAnalyzer()
    
    result = analyzer.analyze([1.0, 2.0, 3.0, 4.0, 5.0], z_threshold=1.0)
    
    assert len(result["outliers"]) > 0
