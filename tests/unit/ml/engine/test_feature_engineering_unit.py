import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.ml.engine.feature_engineering import (
    FeatureEngineer,
    FeatureType,
)


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("src.ml.engine.feature_engineering.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


def build_sample_data():
    return pd.DataFrame(
        {
            "numeric": [1.0, 2.0, None],
            "category": ["A", "B", "A"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
        }
    )


def test_define_feature_and_pipeline_info():
    engineer = FeatureEngineer(config={"enable_caching": True})
    engineer.define_feature("numeric", FeatureType.NUMERIC, data_type="float")
    engineer.define_feature("category", FeatureType.CATEGORICAL, data_type="string")

    steps = [
        {"type": "handle_missing", "method": "fill", "fill_value": 0},
        {"type": "custom_transformation", "function": "log"},
        {"type": "handle_missing", "method": "drop"},
    ]
    engineer.create_pipeline("basic_pipeline", steps, ["numeric", "category"])

    info = engineer.get_pipeline_info("basic_pipeline")
    assert info["name"] == "basic_pipeline"
    assert info["steps"] == 3
    assert info["input_features"] == 2


def test_process_data_with_cache():
    engineer = FeatureEngineer(config={"enable_caching": True, "cache_max_size": 2})
    engineer.create_pipeline(
        "cache_pipeline",
        [{"type": "handle_missing", "method": "fill", "fill_value": 0}],
        ["numeric"],
    )

    data = build_sample_data()[["numeric"]]
    first_result = engineer.process_data(data, "cache_pipeline")

    assert engineer.stats["pipelines_executed"] == 1
    assert engineer.stats["cache_hits"] == 0
    assert first_result["numeric"].isna().sum() == 0

    second_result = engineer.process_data(data, "cache_pipeline")
    assert engineer.stats["cache_hits"] == 1
    pd.testing.assert_frame_equal(first_result, second_result)


def test_create_temporal_features():
    engineer = FeatureEngineer(config={"enable_caching": False})
    engineer.create_pipeline(
        "time_pipeline",
        [{"type": "create_temporal_features"}],
        ["numeric", "timestamp"],
    )

    data = build_sample_data()
    result = engineer.process_data(data, "time_pipeline")

    expected_columns = {"hour", "day_of_week", "hour_sin", "hour_cos"}
    assert expected_columns.issubset(set(result.columns))


def test_custom_transformation_log_transforms_numeric_only():
    engineer = FeatureEngineer(config={"enable_caching": False})
    engineer.create_pipeline(
        "log_pipeline",
        [
            {"type": "custom_transformation", "function": "log"},
        ],
        ["numeric", "category"],
    )
    data = build_sample_data()
    result = engineer.process_data(data, "log_pipeline")
    # numeric column should be transformed, category preserved
    assert result["category"].equals(data["category"])
    assert pytest.approx(result["numeric"].iloc[0]) == pytest.approx(0.6931, abs=1e-3)


def test_handle_missing_drop_removes_rows():
    engineer = FeatureEngineer(config={"enable_caching": False})
    engineer.create_pipeline(
        "drop_pipeline",
        [{"type": "handle_missing", "method": "drop"}],
        ["numeric"],
    )
    data = build_sample_data()[["numeric"]]
    result = engineer.process_data(data, "drop_pipeline")
    assert len(result) == 2


def test_unknown_pipeline_raises():
    engineer = FeatureEngineer()
    with pytest.raises(ValueError, match="管道 unknown 不存在"):
        engineer.process_data(pd.DataFrame({"value": [1]}), "unknown")


def test_create_temporal_features_without_timestamp():
    engineer = FeatureEngineer(config={"enable_caching": False})
    data = pd.DataFrame({"value": [1, 2]})
    result = engineer._create_temporal_features(data)
    assert result.equals(data)


def test_cache_eviction_when_exceeding_limit():
    engineer = FeatureEngineer(config={"enable_caching": True, "cache_max_size": 2})
    engineer.create_pipeline(
        "evict_pipeline",
        [{"type": "handle_missing", "method": "fill", "fill_value": 0}],
        ["numeric"],
    )
    engineer.process_data(pd.DataFrame({"numeric": [1]}), "evict_pipeline")
    engineer.process_data(pd.DataFrame({"numeric": [2]}), "evict_pipeline")
    engineer.process_data(pd.DataFrame({"numeric": [3]}), "evict_pipeline")
    assert len(engineer._pipeline_cache) == 2


def test_get_pipeline_info_returns_none_for_missing_pipeline():
    """测试get_pipeline_info对不存在的管道返回None（101行）"""
    engineer = FeatureEngineer()
    result = engineer.get_pipeline_info("nonexistent_pipeline")
    assert result is None


def test_apply_step_unknown_type_returns_data_unchanged():
    """测试_apply_step处理未知step_type时返回原始数据（157行）"""
    engineer = FeatureEngineer(config={"enable_caching": False})
    engineer.create_pipeline(
        "unknown_step_pipeline",
        [{"type": "unknown_step_type", "param": "value"}],
        ["numeric"],
    )
    
    data = pd.DataFrame({"numeric": [1.0, 2.0, 3.0]})
    result = engineer.process_data(data, "unknown_step_pipeline")
    
    # 未知step_type应该返回原始数据不变
    pd.testing.assert_frame_equal(result, data)
