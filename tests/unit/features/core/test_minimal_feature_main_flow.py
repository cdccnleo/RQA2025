import sys
import numpy as np
import pandas as pd
import pytest
from importlib import import_module

# 修正最上层相对导入
sys.modules.setdefault("src.feature_config", import_module("src.features.core.feature_config"))
sys.modules.setdefault("engine", import_module("src.features.core.engine"))

from src.features.core import minimal_feature_main_flow as main_flow


@pytest.fixture(autouse=True)
def stub_random(monkeypatch):
    class SecretsStub:
        @staticmethod
        def uniform(low, high, size):
            return np.linspace(low, high, num=size)

    monkeypatch.setattr(main_flow.np, "secrets", SecretsStub(), raising=False)


def _patch_dependencies(monkeypatch, *, validate=True, process=None, stats=None, processors=None):
    class StubFeatureConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubFeatureEngine:
        def __init__(self, config):
            self.config = config

        def validate_data(self, data):
            return validate

        def process_features(self, data, config):
            if callable(process):
                return process(data, config)
            return data.assign(feature=data["close"] * 2)

        def get_stats(self):
            if stats is None:
                return {"processed": 100}
            return stats

        def list_processors(self):
            return processors if processors is not None else ["mock"]

    class StubFeatureType:
        TECHNICAL = "TECHNICAL"

    monkeypatch.setattr(main_flow, "FeatureConfig", StubFeatureConfig)
    monkeypatch.setattr(main_flow, "FeatureEngine", StubFeatureEngine)
    monkeypatch.setattr(main_flow, "FeatureType", StubFeatureType)

    return StubFeatureEngine


def test_setup_environment_success(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()

    assert flow.setup_environment() is True
    assert flow.engine is not None
    assert flow.config is not None
    assert isinstance(flow.test_data, pd.DataFrame)
    assert len(flow.test_data) == 100
    assert (flow.test_data["high"] >= flow.test_data["open"]).all()
    assert (flow.test_data["high"] >= flow.test_data["close"]).all()


def test_setup_environment_engine_failure(monkeypatch):
    _patch_dependencies(monkeypatch)

    class RaisingEngine:
        def __init__(self, *_):
            raise RuntimeError("broken")

    monkeypatch.setattr(main_flow, "FeatureEngine", RaisingEngine)
    flow = main_flow.MinimalFeatureMainFlow()

    assert flow.setup_environment() is False


def test_create_test_data_generates_consistent_frame(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()
    flow._create_test_data()

    assert list(flow.test_data.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert flow.test_data["high"].ge(flow.test_data["low"]).all()
    assert flow.test_data["high"].ge(flow.test_data["open"]).all()


def test_execute_feature_processing_happy_path(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()
    flow.setup_environment()

    result = flow.execute_feature_processing()
    assert not result.empty
    assert "feature" in result.columns


def test_execute_feature_processing_validation_failure(monkeypatch):
    _patch_dependencies(monkeypatch, validate=False)
    flow = main_flow.MinimalFeatureMainFlow()
    flow.setup_environment()

    with pytest.raises(ValueError):
        flow.execute_feature_processing()


def test_validate_results_drops_empty_columns(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()
    data = pd.DataFrame(
        {
            "filled": [1, 2, 3],
            "empty": [np.nan, np.nan, np.nan],
        }
    )

    assert flow.validate_results(data) is True


def test_validate_results_returns_false_when_all_empty(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()
    data = pd.DataFrame({"empty": [np.nan, np.nan]})

    assert flow.validate_results(data) is False


def test_run_complete_flow_success(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()

    assert flow.run_complete_flow() is True


def test_run_complete_flow_environment_failure(monkeypatch):
    flow = main_flow.MinimalFeatureMainFlow()
    monkeypatch.setattr(flow, "setup_environment", lambda: False)

    assert flow.run_complete_flow() is False


def test_run_complete_flow_processing_failure(monkeypatch):
    _patch_dependencies(monkeypatch)
    flow = main_flow.MinimalFeatureMainFlow()
    flow.setup_environment()
    monkeypatch.setattr(
        flow,
        "execute_feature_processing",
        lambda: (_ for _ in ()).throw(ValueError("fail")),
    )

    assert flow.run_complete_flow() is False


def test_output_statistics_handles_engine_methods(monkeypatch, caplog):
    stats = {"processed": 5}
    processors = ["p1", "p2"]
    StubEngine = _patch_dependencies(monkeypatch, stats=stats, processors=processors)
    flow = main_flow.MinimalFeatureMainFlow()
    flow.engine = StubEngine(None)
    flow.test_data = pd.DataFrame({"value": [1, 2]})
    processed = pd.DataFrame({"feature": [3, 4]})

    with caplog.at_level("INFO"):
        flow._output_statistics(processed)

    assert "输出特征行数" in caplog.text
    assert "引擎统计" in caplog.text
    assert "注册的处理器" in caplog.text


def test_main_success(monkeypatch):
    class StubFlow:
        def __init__(self):
            pass

        def run_complete_flow(self):
            return True

    monkeypatch.setattr(main_flow, "MinimalFeatureMainFlow", StubFlow)

    with pytest.raises(SystemExit) as exc:
        main_flow.main()

    assert exc.value.code == 0


def test_main_failure(monkeypatch):
    class StubFlow:
        def __init__(self):
            pass

        def run_complete_flow(self):
            return False

    monkeypatch.setattr(main_flow, "MinimalFeatureMainFlow", StubFlow)

    with pytest.raises(SystemExit) as exc:
        main_flow.main()

    assert exc.value.code == 1

