from types import SimpleNamespace

import pandas as pd
import pytest

from src.features.processors.base_processor import BaseFeatureProcessor, ProcessorConfig


class DummyProcessor(BaseFeatureProcessor):
    def __init__(self):
        super().__init__(ProcessorConfig("dummy", feature_params={"alpha": {"window": 3}}))
        self.available_calls = 0
        self.metadata_calls = 0

    def _compute_feature(self, data, feature_name, params):
        return pd.Series(range(len(data)), index=data.index, name=f"{feature_name}_values")

    def _get_feature_metadata(self, feature_name):
        self.metadata_calls += 1
        return {"name": feature_name, "params": self.config.feature_params.get(feature_name, {})}

    def _get_available_features(self):
        self.available_calls += 1
        return {"alpha": {"window": 3}, "beta": {"window": 5}}


@pytest.fixture
def sample_request():
    frame = pd.DataFrame({"close": [1, 2, 3]})
    return SimpleNamespace(data=frame, features=["alpha"], params={"multiplier": 2})


def test_process_raises_for_empty_data():
    processor = DummyProcessor()
    request = SimpleNamespace(data=pd.DataFrame(), features=["alpha"], params={})
    with pytest.raises(ValueError, match="输入数据为空"):
        processor.process(request)


def test_process_checks_unknown_features(sample_request):
    processor = DummyProcessor()
    request = SimpleNamespace(
        data=sample_request.data,
        features=["unknown"],
        params={},
    )
    with pytest.raises(ValueError, match="不支持的特征"):
        processor.process(request)


def test_process_runs_all_features_when_unspecified(sample_request):
    processor = DummyProcessor()
    request = SimpleNamespace(
        data=sample_request.data,
        features=[],
        params={},
    )
    result = processor.process(request)
    assert "feature_alpha" in result.columns
    assert "feature_beta" in result.columns
    assert processor.available_calls == 1


def test_list_features_uses_cache(sample_request):
    processor = DummyProcessor()
    first = processor.list_features()
    second = processor.list_features()
    assert first == second
    assert processor.available_calls == 1


def test_get_feature_info_cached(sample_request):
    processor = DummyProcessor()
    info_first = processor.get_feature_info("alpha")
    info_second = processor.get_feature_info("alpha")
    assert info_first == info_second
    assert processor.metadata_calls == 1


def test_clear_cache_resets_internal_state():
    processor = DummyProcessor()
    processor.list_features()
    processor.get_feature_info("alpha")
    processor.clear_cache()
    processor.list_features()
    processor.get_feature_info("alpha")
    assert processor.available_calls == 2
    assert processor.metadata_calls == 2


def test_validate_config_detects_missing_fields():
    processor = DummyProcessor()
    assert processor.validate_config() is True
    processor.config.processor_type = None  # type: ignore[assignment]
    assert processor.validate_config() is False


def test_feature_params_helpers():
    processor = DummyProcessor()
    assert processor.get_feature_params("alpha") == {"window": 3}
    processor.set_feature_params("beta", {"window": 10})
    assert processor.get_feature_params("beta")["window"] == 10


