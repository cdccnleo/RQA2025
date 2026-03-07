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


from src.data.ai.smart_data_analyzer import SmartDataAnalyzer, SummaryStats
import pytest


def test_summarize_and_trend():
    analyzer = SmartDataAnalyzer()
    series = [1, 2, 3, 4, 5, 6]
    stats = analyzer.summarize(series)
    assert isinstance(stats, SummaryStats)
    assert stats.count == 6 and stats.minimum == 1 and stats.maximum == 6
    # 趋势判定：右半段均值 > 左半段
    assert analyzer.compute_trend(series) == "up"
    # 扁平 / 短序列
    assert analyzer.compute_trend([1]) == "flat"


def test_outliers_and_analyze():
    analyzer = SmartDataAnalyzer()
    series = [10, 10, 10, 10, 100]  # 100 应视为离群
    outliers = analyzer.detect_outliers(series, z_threshold=2.0)
    assert len(outliers) >= 1
    rep = analyzer.analyze(series, z_threshold=2.0)
    assert rep["trend"] in {"up", "down", "flat"}
    assert isinstance(rep["stats"], SummaryStats)
    assert isinstance(rep["outliers"], list)


def test_summarize_type_errors_and_empty():
    analyzer = SmartDataAnalyzer()
    with pytest.raises(TypeError):
        analyzer.summarize("not-a-list")  # type: ignore
    empty_stats = analyzer.summarize([])
    assert empty_stats.count == 0 and empty_stats.std == 0.0


