import logging
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from src.features.processors.distributed.distributed_feature_processor import DistributedFeatureProcessor


def _make_config(**overrides):
    base = {
        "use_distributed": True,
        "max_workers": 2,
        "chunk_size": 2,
        "memory_limit_mb": 512,
        "timeout_seconds": 5,
        "fallback_to_sequential": True,
    }
    base.update(overrides)
    return base



class _SequentialFuture:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class _SequentialExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _SequentialFuture(fn, *args, **kwargs)


def _patch_executor(monkeypatch):
    target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{target}.ProcessPoolExecutor", _SequentialExecutor)
    monkeypatch.setattr(
        f"{target}.as_completed",
        lambda futures, timeout=None: list(futures) if isinstance(futures, dict) else list(futures),
    )


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    target = "src.features.processors.distributed.distributed_feature_processor.logger"
    monkeypatch.setattr(target, logging.getLogger(__name__))


@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {"close": [10 + i for i in range(6)], "volume": [100, 120, 130, 140, 150, 160]},
        index=pd.RangeIndex(6),
    )
    target = pd.Series(range(6), index=data.index, name="target")
    return data, target


@pytest.fixture
def distributed_config():
    return _make_config()


def test_chunk_data_respects_size(sample_data):
    data, _ = sample_data
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False, chunk_size=3))
    chunks = processor._chunk_data(data, chunk_size=3)
    assert len(chunks) == 2
    assert chunks[0].equals(data.iloc[:3])
    assert chunks[1].equals(data.iloc[3:])


def test_distributed_technical_processing_combines_chunks(monkeypatch, sample_data, distributed_config):
    data, _ = sample_data
    target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{target}.DISTRIBUTED_AVAILABLE", True)
    _patch_executor(monkeypatch)

    def fake_process_chunk(self, chunk, indicators, params):
        return chunk.assign(processed=chunk["close"] + params.get("offset", 0))

    monkeypatch.setattr(DistributedFeatureProcessor, "_process_chunk_technical", fake_process_chunk)
    config = dict(distributed_config)
    config["max_workers"] = None
    processor = DistributedFeatureProcessor(config=config)

    result = processor.calculate_distributed_technical_features(
        data, indicators=["sma"], params={"offset": 1}, use_gpu=False
    )

    expected = data.assign(processed=data["close"] + 1)
    pd.testing.assert_frame_equal(result[["processed"]], expected[["processed"]])


def test_initialize_distributed_failure(monkeypatch, distributed_config):
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)
    def fake_cpu_count():
        raise RuntimeError("no cpu")

    monkeypatch.setattr(f"{module_target}.mp.cpu_count", fake_cpu_count)
    config = dict(distributed_config)
    config["max_workers"] = None
    processor = DistributedFeatureProcessor(config=config)
    assert processor.distributed_available is False


def test_distributed_technical_processing_fallback_on_error(monkeypatch, sample_data, distributed_config):
    data, _ = sample_data
    target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{target}.DISTRIBUTED_AVAILABLE", True)
    _patch_executor(monkeypatch)

    def flaky_process(self, chunk, indicators, params):
        if chunk.index.start == 2:
            raise RuntimeError("chunk failed")
        return chunk.assign(flag="ok")

    def sequential_fallback(self, data, indicators, params, use_gpu):
        return data.assign(flag="fallback")

    monkeypatch.setattr(DistributedFeatureProcessor, "_process_chunk_technical", flaky_process)
    monkeypatch.setattr(
        DistributedFeatureProcessor, "_calculate_sequential_technical_features", sequential_fallback
    )
    processor = DistributedFeatureProcessor(config=dict(distributed_config))

    result = processor.calculate_distributed_technical_features(
        data, indicators=["sma"], params={}, use_gpu=False
    )

    assert (result.loc[:1, "flag"] == "ok").all()
    assert (result.loc[2:3, "flag"] == "fallback").all()
    assert (result.loc[4:, "flag"] == "ok").all()


def test_distributed_technical_processing_global_failure(monkeypatch, sample_data, distributed_config):
    data, _ = sample_data
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)

    def boom(_, *__):
        raise RuntimeError("split fail")

    monkeypatch.setattr(DistributedFeatureProcessor, "_chunk_data", boom)
    fallback_called = {}

    def fallback(self, frame, indicators, params, use_gpu):
        fallback_called["called"] = True
        return frame.assign(flag="sequential")

    monkeypatch.setattr(
        DistributedFeatureProcessor,
        "_calculate_sequential_technical_features",
        fallback,
    )
    processor = DistributedFeatureProcessor(config=dict(distributed_config))
    result = processor.calculate_distributed_technical_features(
        data, indicators=["sma"], params={}, use_gpu=False
    )
    assert fallback_called.get("called") is True
    assert "flag" in result.columns


def test_distributed_quality_processing_uses_fallback(monkeypatch, sample_data, distributed_config):
    data, target = sample_data
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)
    _patch_executor(monkeypatch)

    call_state = {"attempts": 0}

    def failing_quality(self, chunk, target_chunk, metrics):
        call_state["attempts"] += 1
        if call_state["attempts"] == 1:
            raise ValueError("boom")
        return pd.DataFrame({"importance_score": 0.3}, index=chunk.index)

    def sequential_quality(self, chunk, target_chunk, metrics):
        return pd.DataFrame({"importance_score": 0.1}, index=chunk.index)

    monkeypatch.setattr(DistributedFeatureProcessor, "_process_chunk_quality", failing_quality)
    monkeypatch.setattr(
        DistributedFeatureProcessor, "_calculate_sequential_quality_features", sequential_quality
    )

    processor = DistributedFeatureProcessor(config=dict(distributed_config))
    result = processor.calculate_distributed_quality_features(data, target, quality_metrics=["importance"])

    assert result.loc[:1, "importance_score"].eq(0.1).all()
    assert result.loc[2:, "importance_score"].eq(0.3).all()


def test_calculate_sequential_technical_features_gpu(monkeypatch, sample_data):
    data, _ = sample_data
    fake_gpu = SimpleNamespace(
        calculate_multiple_indicators_gpu=lambda df, indicators, params: df.assign(gpu=len(indicators))
    )
    gpu_package = ModuleType("gpu_package")
    gpu_package.__path__ = []
    gpu_module = ModuleType("gpu_module")
    gpu_module.GPUTechnicalProcessor = lambda: fake_gpu
    monkeypatch.setitem(sys.modules, "src.features.gpu", gpu_package)
    monkeypatch.setitem(sys.modules, "src.features.gpu.gpu_technical_processor", gpu_module)
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    result = processor._calculate_sequential_technical_features(
        data, indicators=["sma", "ema"], params={}, use_gpu=True
    )
    assert "gpu" in result.columns
    assert result["gpu"].iloc[0] == 2


def test_calculate_sequential_technical_features_failure(monkeypatch, sample_data):
    data, _ = sample_data
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))

    class BoomProcessor:
        def calculate_multiple_indicators(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    module = ModuleType("tech_module")
    module.TechnicalProcessor = BoomProcessor
    monkeypatch.setitem(sys.modules, "src.features.technical.technical_processor", module)

    result = processor._calculate_sequential_technical_features(data, indicators=["sma"], params=None, use_gpu=False)
    assert result.empty


def test_process_chunk_gpu_fallbacks_to_cpu(monkeypatch, sample_data):
    data, _ = sample_data
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))

    class BoomGPU:
        def calculate_multiple_indicators_gpu(self, *_args, **_kwargs):
            raise RuntimeError("gpu down")

    gpu_module = ModuleType("gpu_module")
    gpu_module.GPUTechnicalProcessor = BoomGPU
    monkeypatch.setitem(sys.modules, "src.features.processors.gpu.gpu_technical_processor", gpu_module)

    result = processor._process_chunk_gpu(data, ["sma"], {"sma_window": 2})
    assert not result.empty
    assert "sma" in result.columns


def test_process_chunk_gpu_success(monkeypatch, sample_data):
    data, _ = sample_data
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))

    class StubGPU:
        def calculate_multiple_indicators_gpu(self, frame, indicators, params=None):
            return frame.assign(gpu_flag=1)

    gpu_package = ModuleType("src.features.gpu")
    gpu_package.__path__ = []
    gpu_module = ModuleType("src.features.gpu.gpu_technical_processor")
    gpu_module.GPUTechnicalProcessor = lambda: StubGPU()
    monkeypatch.setitem(sys.modules, "src.features.gpu", gpu_package)
    monkeypatch.setitem(sys.modules, "src.features.gpu.gpu_technical_processor", gpu_module)

    result = processor._process_chunk_gpu(data, ["sma"], {"sma_window": 2})
    assert "gpu_flag" in result.columns
    assert result["gpu_flag"].eq(1).all()


def test_process_chunk_technical_unknown_indicator_returns_empty(sample_data):
    data, _ = sample_data
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    result = processor._process_chunk_technical(data.drop(columns=["close"]), ["sma"], {})
    assert result.empty
    assert len(result.index) == len(data.index)


def test_calculate_sequential_quality_features_calls_analyzers(monkeypatch, sample_data):
    data, target = sample_data
    importance_module = ModuleType("importance_module")
    correlation_module = ModuleType("correlation_module")
    stability_module = ModuleType("stability_module")

    class StubImportance:
        def analyze_feature_importance(self, frame, y, task):
            return {"combined_importance": [0.5] * len(frame)}

    class StubCorrelation:
        def analyze_feature_correlation(self, frame):
            return {"dummy": 1}

    class StubStability:
        def analyze_feature_stability(self, frame):
            return {"combined_stability": [0.2] * len(frame)}

    importance_module.FeatureImportanceAnalyzer = StubImportance
    correlation_module.FeatureCorrelationAnalyzer = StubCorrelation
    stability_module.FeatureStabilityAnalyzer = StubStability

    monkeypatch.setitem(sys.modules, "src.features.feature_importance", importance_module)
    monkeypatch.setitem(sys.modules, "src.features.feature_correlation", correlation_module)
    monkeypatch.setitem(sys.modules, "src.features.feature_stability", stability_module)

    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    result = processor._calculate_sequential_quality_features(data, target, quality_metrics=None)

    assert {"importance_score", "correlation_score", "stability_score"} <= set(result.columns)
    assert result["importance_score"].iloc[0] == 0.5
    assert result["stability_score"].iloc[0] == 0.2
    assert result["correlation_score"].iloc[0] == 0.5


def test_process_chunk_quality_failure_returns_empty(monkeypatch, sample_data):
    data, target = sample_data
    importance_module = ModuleType("importance_module")

    class BoomImportance:
        def analyze_feature_importance(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    importance_module.FeatureImportanceAnalyzer = BoomImportance
    monkeypatch.setitem(sys.modules, "src.features.feature_importance", importance_module)
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    result = processor._process_chunk_quality(data, target, ["importance"])
    assert result.empty
    assert len(result.index) == len(data.index)


def test_calculate_distributed_technical_features_use_gpu(monkeypatch, sample_data, distributed_config):
    data, _ = sample_data
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)
    _patch_executor(monkeypatch)

    called = {}

    def fake_gpu(self, chunk, indicators, params):
        called.setdefault("count", 0)
        called["count"] += 1
        return chunk.assign(gpu=1)

    monkeypatch.setattr(DistributedFeatureProcessor, "_process_chunk_gpu", fake_gpu)
    processor = DistributedFeatureProcessor(config=dict(distributed_config))

    result = processor.calculate_distributed_technical_features(data, indicators=["sma"], params={}, use_gpu=True)
    assert called.get("count", 0) > 0
    assert "gpu" in result.columns


def test_calculate_distributed_quality_features_unavailable(monkeypatch, sample_data):
    data, target = sample_data
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", False)

    sequential_called = {}

    def fallback(self, frame, target_series, metrics):
        sequential_called["called"] = True
        return frame.assign(fake=1)

    monkeypatch.setattr(
        DistributedFeatureProcessor,
        "_calculate_sequential_quality_features",
        fallback,
    )

    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    result = processor.calculate_distributed_quality_features(data, target, quality_metrics=["importance"])
    assert sequential_called.get("called") is True
    assert "fake" in result.columns


def test_calculate_distributed_quality_features_global_failure(monkeypatch, sample_data, distributed_config):
    data, target = sample_data
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)

    def boom_process(self, *_args, **_kwargs):
        raise RuntimeError("quality fail")

    monkeypatch.setattr(DistributedFeatureProcessor, "_chunk_data", lambda self, frame: [frame])
    monkeypatch.setattr(
        DistributedFeatureProcessor,
        "_calculate_sequential_quality_features",
        lambda self, frame, tgt, metrics: frame.assign(fallback=1),
    )
    monkeypatch.setattr(
        DistributedFeatureProcessor,
        "_process_chunk_quality",
        boom_process,
    )
    processor = DistributedFeatureProcessor(config=dict(distributed_config))
    result = processor.calculate_distributed_quality_features(data, target, quality_metrics=["importance"])
    assert "fallback" in result.columns


def test_get_distributed_info_unavailable(monkeypatch):
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", False)
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    info = processor.get_distributed_info()
    assert info["available"] is False
    assert "reason" in info


def test_get_distributed_info_failure(monkeypatch):
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)
    def fake_cpu_count():
        raise RuntimeError("cpus missing")

    monkeypatch.setattr(f"{module_target}.mp.cpu_count", fake_cpu_count)
    processor = DistributedFeatureProcessor(config=_make_config())
    info = processor.get_distributed_info()
    assert info["available"] is False
    assert "cpus missing" in info["reason"]


def test_estimate_processing_time_with_speedup(monkeypatch, distributed_config):
    module_target = "src.features.processors.distributed.distributed_feature_processor"
    monkeypatch.setattr(f"{module_target}.DISTRIBUTED_AVAILABLE", True)
    processor = DistributedFeatureProcessor(config=dict(distributed_config))
    processor.distributed_available = True
    result = processor.estimate_processing_time(data_size=1000, indicators_count=4, use_gpu=False)
    assert result["speedup_factor"] == distributed_config["max_workers"]
    assert result["estimated_time_seconds"] > 0


def test_estimate_processing_time_no_distributed(distributed_config):
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False))
    processor.distributed_available = False
    result = processor.estimate_processing_time(data_size=500, indicators_count=2, use_gpu=True)
    assert result["speedup_factor"] == 1
    assert result["estimated_time_seconds"] > 0


def test_optimize_chunk_size_applies_bounds(distributed_config):
    processor = DistributedFeatureProcessor(config=_make_config(use_distributed=False, chunk_size=1000, max_workers=1))
    optimized_small = processor.optimize_chunk_size(data_size=50, memory_limit_mb=1)
    optimized_large = processor.optimize_chunk_size(data_size=50_000, memory_limit_mb=500)
    assert 100 <= optimized_small <= 10000
    assert optimized_large <= 10000
