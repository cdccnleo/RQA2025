import pytest

from src.features.core import exceptions as exc_module


def test_data_validation_error_str_includes_details():
    error = exc_module.FeatureDataValidationError(
        "数据校验失败",
        missing_columns=["close", "volume"],
        invalid_types=["str"],
        data_shape=(120, 5),
    )
    message = str(error)
    assert "缺失列" in message
    assert "无效类型" in message
    assert "数据形状: (120, 5)" in message
    assert error.error_type is exc_module.FeatureErrorType.DATA_VALIDATION


def test_processing_error_str_with_original_exception():
    original = RuntimeError("底层错误")
    error = exc_module.FeatureProcessingError(
        "处理失败",
        processor_name="demo",
        step="fit",
        original_error=original,
        feature_name="alpha",
    )
    rendered = str(error)
    assert "处理器: demo" in rendered
    assert "步骤: fit" in rendered
    assert "特征: alpha" in rendered
    assert "底层错误" in rendered


def test_exception_factory_creates_concrete_instances():
    factory = exc_module.FeatureExceptionFactory()
    assert isinstance(factory.create_data_validation_error("bad"), exc_module.FeatureDataValidationError)
    assert isinstance(factory.create_config_validation_error("bad"), exc_module.FeatureConfigValidationError)
    assert isinstance(factory.create_processing_error("bad"), exc_module.FeatureProcessingError)
    assert isinstance(factory.create_standardization_error("bad"), exc_module.FeatureStandardizationError)
    assert isinstance(factory.create_selection_error("bad"), exc_module.FeatureSelectionError)
    assert isinstance(factory.create_sentiment_error("bad"), exc_module.FeatureSentimentError)
    assert isinstance(factory.create_technical_error("bad"), exc_module.FeatureTechnicalError)
    assert isinstance(factory.create_general_error("bad"), exc_module.FeatureGeneralError)


def test_exception_handler_enhances_known_errors_and_tracks_history(monkeypatch):
    handler = exc_module.FeatureExceptionHandler()
    data_error = exc_module.FeatureDataValidationError("数据不完整")
    enhanced = handler.handle_exception(data_error, {"data_shape": (10, 3)})
    assert enhanced.data_shape == (10, 3)

    config_error = exc_module.FeatureConfigValidationError("配置错误")
    enhanced_config = handler.handle_exception(config_error, {"config": {"key": "value"}})
    assert enhanced_config.config_dict == {"key": "value"}

    generic_error = ValueError("通用错误")
    returned = handler.handle_exception(generic_error)
    assert returned is generic_error

    summary = handler.get_error_summary()
    assert summary["total_errors"] == 3
    assert summary["error_types"]["FeatureDataValidationError"] == 1
    assert summary["error_types"]["FeatureConfigValidationError"] == 1
    assert summary["error_types"]["ValueError"] == 1
    assert summary["recent_errors"]  # 最近错误列表不为空

    handler.clear_history()
    assert handler.error_count == 0
    assert handler.get_error_summary()["recent_errors"] == []


def test_exception_handler_enhances_processing_error_with_context():
    handler = exc_module.FeatureExceptionHandler()
    processing_error = exc_module.FeatureProcessingError("执行失败", processor_name=None, step=None)
    enhanced = handler.handle_exception(processing_error, {"processor": "selector", "step": "transform"})
    assert enhanced.processor_name == "selector"
    assert enhanced.step == "transform"


def test_handle_feature_exception_decorator_uses_global_handler(monkeypatch):
    custom_handler = exc_module.FeatureExceptionHandler()
    monkeypatch.setattr(exc_module, "feature_exception_handler", custom_handler)

    @exc_module.handle_feature_exception
    def failing_function(arg1, *, flag):
        raise exc_module.FeatureSentimentError("情感分析失败", model_type="bert")

    with pytest.raises(exc_module.FeatureSentimentError) as excinfo:
        failing_function("text", flag=True)

    assert custom_handler.error_count == 1
    history_entry = custom_handler.error_history[-1]
    assert history_entry["error_type"] == "FeatureSentimentError"
    assert history_entry["context"]["function"] == "failing_function"
    assert excinfo.value.model_type == "bert"


def test_exception_handler_returns_features_on_assessment_failure():
    handler = exc_module.FeatureExceptionHandler()
    error = exc_module.FeatureProcessingError("quality failed")
    result = handler.handle_exception(error, {"processor": "quality", "step": "filter"})
    assert result.processor_name == "quality"
    assert result.step == "filter"

