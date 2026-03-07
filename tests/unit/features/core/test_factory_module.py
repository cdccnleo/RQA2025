from types import SimpleNamespace

import pytest

from src.features.core import factory as factory_module
from src.features.core.feature_config import FeatureType


class DummyParallelProcessor:
    def __init__(self, parallel_config):
        self.parallel_config = parallel_config


class DummyManagerProcessor:
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def process_features(self, data, config):
        return {"manager": data, "config": config}

    def get_stats(self):
        return {"status": "manager"}

    def close(self):
        self.closed = True


class DummyFeatureProcessor:
    def __init__(self, processor_config):
        self.processor_config = processor_config


def test_register_processor_adds_new_entry(caplog):
    factory = factory_module.FeatureProcessorFactory()

    class CustomProcessor:
        pass

    factory.register_processor("custom", CustomProcessor)
    info = factory.get_processor_info("custom")
    assert info["class_name"] == "CustomProcessor"
    assert info["module"] == CustomProcessor.__module__


def test_create_processor_handles_unknown_type():
    factory = factory_module.FeatureProcessorFactory()
    with pytest.raises(ValueError):
        factory.create_processor("unknown")


def test_create_processor_branches(monkeypatch):
    factory = factory_module.FeatureProcessorFactory()
    factory._processors["parallel"]["class"] = DummyParallelProcessor
    factory._processors["manager"]["class"] = DummyManagerProcessor
    factory._processors["feature"]["class"] = DummyFeatureProcessor

    parallel = factory.create_processor("parallel", {"n_jobs": 3, "chunk_size": 12})
    assert isinstance(parallel, DummyParallelProcessor)
    assert parallel.parallel_config.n_jobs == 3
    assert parallel.parallel_config.chunk_size == 12

    manager = factory.create_processor(
        "manager",
        {
            "feature_types": [FeatureType.TECHNICAL, FeatureType.SENTIMENT],
            "technical_indicators": ["sma"],
            "enable_feature_selection": True,
            "enable_standardization": False,
            "max_workers": 8,  # 应被过滤
        },
    )
    assert isinstance(manager, DummyManagerProcessor)
    assert manager.feature_config.enable_feature_selection is True
    assert manager.feature_config.enable_standardization is False
    assert FeatureType.SENTIMENT in manager.feature_config.feature_types

    feature = factory.create_processor(
        "feature",
        {"feature_params": {"foo": {"window": 5}}},
    )
    assert isinstance(feature, DummyFeatureProcessor)
    assert feature.processor_config.processor_type == "feature"
    assert feature.processor_config.feature_params["foo"]["window"] == 5

    factory._processors["optimized"]["class"] = lambda cfg: cfg
    optimized_proc = factory.create_processor("optimized", {"foo": 1})
    assert optimized_proc == {"foo": 1}


def test_create_processor_logs_and_raises_on_failure(monkeypatch):
    factory = factory_module.FeatureProcessorFactory()

    class FailingProcessor:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    factory.register_processor("failing", FailingProcessor)
    with pytest.raises(RuntimeError):
        factory.create_processor("failing")


def test_unified_feature_manager_initializes_and_processes(monkeypatch):
    created = {}

    class DummyFactory:
        def create_processor(self, processor_type, config=None, **_kwargs):
            if processor_type == "manager":
                proc = DummyManagerProcessor(config)
                created[processor_type] = proc
                return proc
            if processor_type == "parallel":
                proc = SimpleNamespace(
                    process_features_parallel=lambda data, cfg: {"parallel": data, "cfg": cfg},
                    get_stats=lambda: {"status": "parallel"},
                    close=lambda: None,
                )
                created[processor_type] = proc
                return proc
            if processor_type == "optimized":
                proc = SimpleNamespace(get_stats=lambda: {"status": "optimized"}, close=lambda: setattr(proc, "closed", True))
                created[processor_type] = proc
                return proc
            raise ValueError("unsupported")

    monkeypatch.setattr(factory_module, "FeatureProcessorFactory", lambda: DummyFactory())
    manager = factory_module.UnifiedFeatureManager(
        {
            "manager": {"feature_types": [FeatureType.TECHNICAL]},
            "parallel": {"n_jobs": 2},
            "optimized": {"enable_parallel": False},
        }
    )
    assert {"manager", "parallel", "optimized"} <= set(created.keys())
    assert "optimized" in manager._active_processors

    results = manager.process_features(
        data={"close": [1, 2]},
        feature_configs=[
            {"processor_type": "manager"},
            {"processor_type": "parallel"},
            {"processor_type": "missing"},
        ],
    )
    assert results["manager"]["manager"] == {"close": [1, 2]}
    assert results["parallel"]["parallel"] == {"close": [1, 2]}

    stats = manager.get_performance_stats()
    assert stats["parallel"]["status"] == "parallel"
    assert stats["manager"]["status"] == "manager"

    manager.close()
    assert manager._active_processors == {}
    assert getattr(created["optimized"], "closed", False) is True


def test_unified_feature_manager_create_processor_handles_error(monkeypatch):
    class DummyFactory:
        def create_processor(self, *_args, **_kwargs):
            raise RuntimeError("factory failure")

    monkeypatch.setattr(factory_module, "FeatureProcessorFactory", lambda: DummyFactory())
    manager = factory_module.UnifiedFeatureManager({})
    assert manager.create_processor("any") is None


def test_create_feature_processor_delegates(monkeypatch):
    calls = []

    class DummyFactory:
        def create_processor(self, processor_type, config=None, **kwargs):
            calls.append((processor_type, config, kwargs))
            return "created"

    monkeypatch.setattr(factory_module, "feature_processor_factory", DummyFactory())
    result = factory_module.create_feature_processor("base", {"foo": 1}, enabled=True)
    assert result == "created"
    assert calls[0][0] == "base"
    assert calls[0][1] == {"foo": 1}
    assert calls[0][2] == {"enabled": True}


def test_get_unified_feature_manager_returns_instance():
    manager = factory_module.get_unified_feature_manager({})
    assert isinstance(manager, factory_module.UnifiedFeatureManager)

