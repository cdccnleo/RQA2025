import types
import sys
import numpy as np
import pandas as pd
import pytest

from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor


class _StubLogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


@pytest.fixture(autouse=True)
def stub_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.get_logger",
        lambda *_args, **_kwargs: _StubLogger(),
    )


def _stub_cp():
    module = types.SimpleNamespace()

    class _CudaDevice:
        def __init__(self):
            pass

        def use(self):
            return None

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        Device = _CudaDevice

        class runtime:
            @staticmethod
            def memGetInfo():
                return 1024, 2048

        class mempool:
            @staticmethod
            def free_all_blocks():
                return None

    module.cuda = _Cuda

    def _zeros(size, dtype=None):
        return np.zeros(size, dtype=dtype)

    module.zeros = _zeros
    module.zeros_like = np.zeros_like
    module.ones = np.ones
    module.asarray = np.asarray
    module.array = np.array
    module.concatenate = np.concatenate
    module.diff = np.diff
    module.convolve = np.convolve
    module.abs = np.abs
    module.mean = np.mean
    module.maximum = np.maximum
    module.minimum = np.minimum

    class _MemoryPool:
        def set_limit(self, size=0):
            return None

    module.get_default_memory_pool = lambda: _MemoryPool()

    return module


def _stub_cp_available():
    module = _stub_cp()

    class _Pool:
        def __init__(self):
            self.limit = None

        def set_limit(self, size=0):
            self.limit = size

    pool = _Pool()

    module.cuda.is_available = staticmethod(lambda: True)
    module.cuda.runtime.memGetInfo = staticmethod(lambda: (1024 * 1024 * 512, 1024 * 1024 * 1024))
    module.get_default_memory_pool = lambda: pool
    module.zeros = np.zeros
    module.full = np.full
    module._pool = pool
    return module


def _sample_frame(size=10):
    return pd.DataFrame(
        {
            "close": np.linspace(10, 10 + size, num=size),
            "high": np.linspace(11, 11 + size, num=size),
            "low": np.linspace(9, 9 + size, num=size),
        }
    )


@pytest.fixture
def cpu_only_config(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        False,
    )
    return {"use_gpu": True, "fallback_to_cpu": True}


def test_should_use_gpu_respects_threshold(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )

    class StubProcessor(GPUTechnicalProcessor):
        def __init__(self):
            super().__init__(config={"optimization_level": "balanced"})
            self.logger = _StubLogger()

    processor = StubProcessor()
    monkeypatch.setattr(processor, "get_gpu_info", lambda: {"memory_usage": 10, "total_memory_gb": 8})
    assert processor._should_use_gpu(1000) is True
    monkeypatch.setattr(processor, "get_gpu_info", lambda: {"memory_usage": 90, "total_memory_gb": 8})
    assert processor._should_use_gpu(1000) is False


def test_clear_gpu_memory_handles_exception(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})

    class BoomPool:
        @staticmethod
        def free_all_blocks():
            raise RuntimeError("boom")

    class BoomDevice:
        def use(self):
            return None

    stub_cp = _stub_cp()
    stub_cp.cuda.Device = BoomDevice
    stub_cp.cuda.mempool.free_all_blocks = staticmethod(BoomPool.free_all_blocks)

    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        stub_cp,
    )
    processor.clear_gpu_memory()


def test_calculate_rsi_gpu_small_dataset_falls_back(monkeypatch, cpu_only_config):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config=cpu_only_config)
    frame = _sample_frame(size=5)
    result = processor.calculate_rsi_gpu(frame, window=14)
    assert isinstance(result, pd.Series)
    assert result.name in {"RSI_14", "close"}


def test_calculate_macd_gpu_falls_back_to_cpu(monkeypatch, cpu_only_config):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config=cpu_only_config)
    frame = _sample_frame(size=50)
    result = processor.calculate_macd_gpu(frame)
    assert {"macd", "signal", "histogram"} <= set(result.columns)


def test_calculate_bollinger_gpu_cpu_path(monkeypatch, cpu_only_config):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config=cpu_only_config)
    frame = _sample_frame(size=40)
    result = processor.calculate_bollinger_bands_gpu(frame, window=20, num_std=2)
    expected = {"upper", "middle", "lower"}
    assert expected <= set(result.columns)


def test_calculate_atr_gpu_cpu_fallback(monkeypatch, cpu_only_config):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config=cpu_only_config)
    frame = _sample_frame(size=40)
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1
    result = processor.calculate_atr_gpu(frame, window=14)
    assert isinstance(result, pd.Series)


def test_initialize_gpu_success_sets_limit(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp_available()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    assert processor.gpu_available is True
    assert cp_stub._pool.limit is not None


def test_initialize_gpu_returns_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp()
    cp_stub.cuda.is_available = staticmethod(lambda: False)
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": False})
    assert processor._initialize_gpu() is False


def test_preallocate_memory_blocks_handles_failure(monkeypatch):
    cp_stub = _stub_cp_available()
    calls = {"count": 0}

    def failing_zeros(size, dtype=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fail allocation")
        return np.zeros(size, dtype=dtype)

    cp_stub.zeros = failing_zeros
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    processor._preallocate_memory_blocks()  # should swallow failure


def test_optimize_memory_access_handles_pool_error(monkeypatch):
    cp_stub = _stub_cp_available()

    class BoomPool:
        def set_limit(self, size=0):
            raise RuntimeError("limit fail")

    cp_stub.get_default_memory_pool = lambda: BoomPool()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    processor._optimize_memory_access()


def test_should_use_gpu_large_dataset_low_memory(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp_available()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )

    processor = GPUTechnicalProcessor(config={"optimization_level": "aggressive"})

    def stub_info():
        return {"memory_usage": 10, "total_memory_gb": 2}

    monkeypatch.setattr(processor, "get_gpu_info", stub_info)
    assert processor._should_use_gpu(200000) is False


def test_get_gpu_info_handles_exception(monkeypatch):
    cp_stub = _stub_cp()

    class FailingDevice:
        def __init__(self):
            raise RuntimeError("device error")

    cp_stub.cuda.Device = FailingDevice
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda *_: "Stub",
            device_count=lambda: 0,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    processor = GPUTechnicalProcessor(config={"use_gpu": False})
    info = processor.get_gpu_info()
    assert info["available"] is False


def test_initialize_gpu_failure_logs_and_disables(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )

    class BoomDevice:
        def __init__(self):
            raise RuntimeError("device boom")

    cp_stub = _stub_cp()
    cp_stub.cuda.Device = BoomDevice
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    assert processor._initialize_gpu() is False


def test_calculate_multiple_indicators_switches_to_cpu_on_failure(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp_available()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)

    def boom_process(*_args, **_kwargs):
        raise RuntimeError("gpu fail")

    processor._calculate_multiple_indicators_gpu = boom_process  # type: ignore[attr-defined]
    monkeypatch.setattr(
        processor,
        "_calculate_multiple_indicators_cpu",
        lambda data, indicators, params=None: data.assign(cpu=True),
    )
    frame = _sample_frame(size=20)
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1
    result = processor.calculate_multiple_indicators_gpu(frame, ["sma"])
    assert "cpu" in result.columns


def test_clear_gpu_memory_skips_when_no_gpu(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": False})
    processor.gpu_available = False
    processor.clear_gpu_memory()  # should be no-op without errors


def test_calculate_multiple_indicators_gpu_uses_gpu_path(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp_available()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )

    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)
    frame = _sample_frame(size=30)
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1

    indicators = ["sma", "ema", "macd", "bollinger", "atr"]
    params = {
        "sma_window": 5,
        "ema_window": 5,
        "macd_fast": 3,
        "macd_slow": 6,
        "macd_signal": 3,
        "bb_window": 5,
        "bb_std": 2,
        "atr_window": 5,
    }

    result = processor.calculate_multiple_indicators_gpu(frame, indicators, params)
    expected_cols = {
        "sma_5",
        "ema_5",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr_5",
    }
    assert expected_cols <= set(result.columns)


def test_calculate_multiple_indicators_gpu_logs_on_general_failure(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    cp_stub = _stub_cp_available()
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        cp_stub,
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda *_: True)

    def boom_gpu(*_args, **_kwargs):
        raise RuntimeError("gpu process fail")

    processor._calculate_multiple_indicators_gpu = boom_gpu  # type: ignore[attr-defined]
    monkeypatch.setattr(
        processor,
        "_calculate_multiple_indicators_cpu",
        lambda data, indicators, params=None: data.assign(cpu=True),
    )
    frame = _sample_frame(size=20)
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1
    result = processor.calculate_multiple_indicators_gpu(frame, ["sma"])
    assert "cpu" in result.columns


def test_calculate_macd_gpu_handles_exception(monkeypatch, cpu_only_config, caplog):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _stub_cp_available(),
        raising=False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": True, "fallback_to_cpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)

    def boom_macd(*_args, **_kwargs):
        raise RuntimeError("macd fail")

    def cpu_macd(data, fast=12, slow=26, signal=9):
        index = data.index
        return pd.DataFrame({"macd": pd.Series(0.0, index=index), "signal": pd.Series(0.0, index=index), "histogram": pd.Series(0.0, index=index)})

    monkeypatch.setattr(processor, "_calculate_macd_cpu", cpu_macd)
    frame = _sample_frame(size=40)
    with caplog.at_level("WARNING"):
        result = processor.calculate_multiple_indicators_gpu(frame, ["macd"])
    assert {"macd_signal", "macd_histogram"} <= set(result.columns)


