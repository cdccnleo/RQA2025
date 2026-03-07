import pandas as pd
import pytest
from unittest.mock import MagicMock


def test_process_features_full_pipeline(feature_engine_factory, sample_price_frame, feature_config_basic):
    """
    验证在启用选择、标准化和持久化的情况下，特征引擎会按照完整流程执行并更新统计信息。
    """
    engine = feature_engine_factory(feature_config_basic)

    engineered = sample_price_frame.assign(engineered_flag=1.0)
    processed = engineered.assign(processed_flag=2.0)
    selected = processed[["processed_flag"]].copy()
    standardized = selected.assign(standardized_flag=3.0)

    engine._engineer_features = MagicMock(return_value=engineered)
    engine._process_features = MagicMock(return_value=processed)
    engine._selector = MagicMock()
    engine._selector.select_features = MagicMock(return_value=selected)
    engine._standardizer = MagicMock()
    engine._standardizer.standardize_features = MagicMock(return_value=standardized)
    engine._saver = MagicMock()

    config = feature_config_basic
    config.enable_feature_saving = True

    result = engine.process_features(sample_price_frame, config)

    engine._engineer_features.assert_called_once_with(sample_price_frame, config)
    engine._process_features.assert_called_once_with(engineered, config)
    engine._selector.select_features.assert_called_once_with(processed, config=config)
    engine._standardizer.standardize_features.assert_called_once_with(selected, config=config)
    engine._saver.save_features.assert_called_once_with(standardized, config=config)

    assert result.equals(standardized)
    assert engine.stats["processed_features"] == standardized.shape[1]
    assert engine.stats["errors"] == 0
    assert engine.stats["processing_time"] >= 0


def test_process_features_without_optional_steps(
    feature_engine_factory,
    sample_price_frame,
    feature_config_without_selection,
):
    """
    验证关闭特征选择和标准化后，流程会直接返回处理结果且不会访问对应组件。
    """
    engine = feature_engine_factory(feature_config_without_selection)

    engineered = sample_price_frame.assign(engineered_flag=1.0)
    processed = engineered.assign(processed_flag=2.0)

    engineer_mock = MagicMock(return_value=engineered)
    process_mock = MagicMock(return_value=processed)

    engine._engineer_features = engineer_mock
    engine._process_features = process_mock

    result = engine.process_features(sample_price_frame, feature_config_without_selection)

    engineer_mock.assert_called_once()
    process_mock.assert_called_once()
    assert result.equals(processed)
    assert engine.stats["processed_features"] == processed.shape[1]
    assert engine.stats["errors"] == 0


def test_process_features_invalid_data(feature_engine_factory, feature_config_basic):
    """
    验证输入数据缺失必要字段或为空时会抛出 ValueError 并记录错误计数。
    """
    engine = feature_engine_factory(feature_config_basic)
    invalid_df = pd.DataFrame()

    with pytest.raises(ValueError):
        engine.process_features(invalid_df, feature_config_basic)

    assert engine.stats["errors"] == 1


def test_validate_data_checks_types(feature_engine_factory, feature_config_basic):
    """
    验证 validate_data 会拒绝缺失列或非数值列的数据。
    """
    engine = feature_engine_factory(feature_config_basic)

    df_missing = pd.DataFrame({"close": [1, 2, 3]})
    assert engine.validate_data(df_missing) is False

    df_non_numeric = pd.DataFrame(
        {"close": [1, 2, 3], "high": ["a", "b", "c"], "low": [1, 2, 3], "volume": [1, 2, 3]}
    )
    assert engine.validate_data(df_non_numeric) is False

    df_valid = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0], "low": [0.5, 1.5, 2.5], "volume": [100, 200, 300]}
    )
    assert engine.validate_data(df_valid) is True


def test_register_processor_requires_base_class(feature_engine_factory, passthrough_processor):
    """
    验证 register_processor 会限制处理器类型，且成功注册后可查询到。
    """
    engine = feature_engine_factory()

    with pytest.raises(ValueError):
        engine.register_processor("invalid", object())

    engine.register_processor("passthrough", passthrough_processor)
    assert "passthrough" in engine.list_processors()
    assert engine.get_processor("passthrough") is passthrough_processor


def test_stats_reset(feature_engine_factory, sample_price_frame, feature_config_basic):
    """
    验证 reset_stats 会将统计信息恢复为初始状态。
    """
    engine = feature_engine_factory(feature_config_basic)

    engine.stats["processed_features"] = 10
    engine.stats["processing_time"] = 5.5
    engine.stats["errors"] = 2

    engine.reset_stats()

    assert engine.stats == {"processed_features": 0, "processing_time": 0.0, "errors": 0}

