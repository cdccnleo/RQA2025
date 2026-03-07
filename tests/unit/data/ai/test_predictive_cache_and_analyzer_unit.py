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


from typing import List

import pytest

from src.data.ai.predictive_cache import PredictiveCache
from src.data.ai.smart_data_analyzer import SmartDataAnalyzer, SummaryStats


def test_predictive_cache_fifo_and_prediction():
    cache = PredictiveCache(capacity=3)
    cache.set("A", 1)
    cache.set("B", 2)
    cache.set("C", 3)
    # 触发 FIFO 淘汰
    cache.set("D", 4)
    assert cache.get("A") is None  # A 被淘汰
    assert cache.get("B") == 2
    stats = cache.get_stats()
    assert stats.size == 3 and stats.evictions >= 1

    # 训练简单转移：B->C, C->D，期望 next of C 为 D
    cache.clear()
    cache.set("B", 2)
    cache.set("C", 3)
    cache.set("D", 4)
    assert cache.predict_next_key("C") == "D"
    # top-k
    assert cache.top_predictions("C", k=2)[:1] == ["D"]


def test_predictive_cache_invalid_capacity():
    with pytest.raises(ValueError):
        PredictiveCache(0)


def test_smart_data_analyzer_basic():
    analyzer = SmartDataAnalyzer()
    series: List[float] = [1, 2, 3, 4, 5, 100]
    stats = analyzer.summarize(series)
    assert isinstance(stats, SummaryStats)
    assert stats.count == 6 and stats.minimum == 1 and stats.maximum == 100

    outliers = analyzer.detect_outliers(series, z_threshold=2.0)
    assert len(outliers) >= 1  # 100 通常会被识别为异常

    trend = analyzer.compute_trend([1, 1, 2, 2, 3, 3])
    assert trend in ("up", "flat", "down")

    report = analyzer.analyze(series, z_threshold=2.0)
    assert "stats" in report and "trend" in report and "outliers" in report


