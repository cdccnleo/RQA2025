import pandas as pd
import pytest

from src.features.core.engine import FeatureEngine
from src.features.processors.base_processor import BaseFeatureProcessor, ProcessorConfig


@pytest.fixture
def feature_request_stub(monkeypatch):
    """
    覆盖 FeatureRequest，避免真实基础设施依赖并兼容旧参数名称。
    """

    class StubFeatureRequest:
        def __init__(self, data, features=None, params=None, feature_names=None, config=None, metadata=None):
            self.data = data
            self.features = features or feature_names or []
            self.params = params or config or {}
            self.metadata = metadata or {}

    monkeypatch.setattr(
        "src.infrastructure.interfaces.standard_interfaces.FeatureRequest",
        StubFeatureRequest,
        raising=False,
    )


class _SimpleProcessor(BaseFeatureProcessor):
    """
    测试用处理器，直接在数据上附加标识列。
    """

    def __init__(self, name: str, value: float = 1.0, raise_error: bool = False):
        super().__init__(ProcessorConfig(processor_type=name, feature_params={name: {}}))
        self._value = value
        self._raise = raise_error

    def process(self, request):  # type: ignore[override]
        if self._raise:
            raise RuntimeError(f"{self.processor_type} failed")
        frame = request.data.copy()
        frame[f"{self.processor_type}_flag"] = self._value
        return frame

    def _compute_feature(self, data: pd.DataFrame, feature_name: str, params):  # type: ignore[override]
        return data.iloc[:, 0]

    def _get_feature_metadata(self, feature_name: str):  # type: ignore[override]
        return {"name": feature_name}

    def _get_available_features(self):  # type: ignore[override]
        return ["flag"]


@pytest.fixture
def engine_without_defaults(monkeypatch):
    """
    返回禁用默认处理器注册的 FeatureEngine，便于精确控制。
    """
    monkeypatch.setattr(FeatureEngine, "_register_default_processors", lambda self: None)
    return FeatureEngine()


def test_engineer_features_combines_processors(
    feature_request_stub,
    engine_without_defaults,
    sample_price_frame,
    feature_config_with_sentiment,
):
    technical = _SimpleProcessor("technical")
    sentiment = _SimpleProcessor("sentiment", value=2.0)

    engine_without_defaults.processors = {
        "technical": technical,
        "sentiment": sentiment,
    }

    combined = engine_without_defaults._engineer_features(sample_price_frame, feature_config_with_sentiment)

    assert "technical_flag" in combined.columns
    assert "sentiment_flag" in combined.columns
    expected_tech = pd.Series([1.0] * len(combined), index=combined.index, name="technical_flag")
    expected_sent = pd.Series([2.0] * len(combined), index=combined.index, name="sentiment_flag")
    pd.testing.assert_series_equal(combined["technical_flag"], expected_tech)
    pd.testing.assert_series_equal(combined["sentiment_flag"], expected_sent)


def test_engineer_features_failure_returns_input(
    feature_request_stub,
    engine_without_defaults,
    sample_price_frame,
    feature_config_basic,
):
    engine_without_defaults.processors = {"technical": _SimpleProcessor("technical", raise_error=True)}
    result = engine_without_defaults._engineer_features(sample_price_frame, feature_config_basic)
    assert result.equals(sample_price_frame)


def test_process_features_with_general_processor(
    feature_request_stub,
    engine_without_defaults,
    sample_price_frame,
    feature_config_basic,
):
    engine_without_defaults.processors = {"general": _SimpleProcessor("general", value=3.0)}
    processed = engine_without_defaults._process_features(sample_price_frame, feature_config_basic)
    assert "general_flag" in processed.columns


def test_process_features_without_processor_returns_original(
    feature_request_stub,
    engine_without_defaults,
    sample_price_frame,
    feature_config_basic,
):
    processed = engine_without_defaults._process_features(sample_price_frame, feature_config_basic)
    assert processed.equals(sample_price_frame)


def test_process_with_processor_success(
    feature_request_stub,
    engine_without_defaults,
    sample_price_frame,
    feature_config_basic,
):
    engine_without_defaults.processors = {"passthrough": _SimpleProcessor("passthrough", value=5.0)}
    result = engine_without_defaults.process_with_processor(sample_price_frame, "passthrough", feature_config_basic)
    assert "passthrough_flag" in result.columns


def test_process_with_processor_missing(engine_without_defaults, sample_price_frame, feature_config_basic):
    with pytest.raises(ValueError):
        engine_without_defaults.process_with_processor(sample_price_frame, "missing", feature_config_basic)


def test_list_processors_reflects_registration(engine_without_defaults, feature_request_stub):
    engine_without_defaults.register_processor("technical", _SimpleProcessor("technical"))
    assert engine_without_defaults.list_processors() == ["technical"]

