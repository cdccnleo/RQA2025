import logging

import numpy as np
import pandas as pd
import pytest

from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor


class _StubGPUArray(np.ndarray):
    pass


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.get_logger",
        lambda name: logging.getLogger(name),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.logger",
        logging.getLogger(__name__),
    )


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "close": np.linspace(1.0, 10.0, num=50),
            "high": np.linspace(1.2, 10.2, num=50),
            "low": np.linspace(0.8, 9.8, num=50),
        }
    )


def _force_cpu(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        None,
        raising=False,
    )


def test_calculate_sma_gpu_falls_back_to_cpu(sample_frame, monkeypatch):
    _force_cpu(monkeypatch)

    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    result = processor.calculate_sma_gpu(sample_frame, window=5)

    expected = sample_frame["close"].rolling(window=5).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_rsi_gpu_with_small_dataset_uses_cpu(sample_frame, monkeypatch):
    _force_cpu(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True, "gpu_threshold": 999})

    rsi = processor.calculate_rsi_gpu(sample_frame, window=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_frame)
    finite_rsi = rsi.dropna()
    assert finite_rsi.between(0, 100, inclusive="both").all()


def test_calculate_macd_gpu_cpu_path(sample_frame, monkeypatch):
    _force_cpu(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})

    macd_df = processor.calculate_macd_gpu(sample_frame)
    assert {"macd", "signal", "histogram"} <= set(macd_df.columns)
    assert len(macd_df) == len(sample_frame)


def test_calculate_bollinger_bands_gpu_cpu_path(sample_frame, monkeypatch):
    _force_cpu(monkeypatch)
    processor = GPUTechnicalProcessor()

    bands = processor.calculate_bollinger_bands_gpu(sample_frame, window=10, num_std=2)
    assert {"upper", "middle", "lower"} <= set(bands.columns)
    assert len(bands) == len(sample_frame)


class _CpStub:
    float32 = np.float32
    nan = np.nan
    float64 = np.float64

    @staticmethod
    def asarray(arr, dtype=None):
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def zeros_like(arr, dtype=None):
        return np.zeros_like(arr, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def asnumpy(arr):
        return np.asarray(arr)

    @staticmethod
    def ones(size, dtype=None):
        return np.ones(size, dtype=dtype)

    @staticmethod
    def convolve(a, v, mode="full"):
        return np.convolve(a, v, mode)

    @staticmethod
    def full(shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def concatenate(arrays):
        return np.concatenate(arrays)

    @staticmethod
    def diff(a):
        return np.diff(a)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def roll(a, shift):
        return np.roll(a, shift)

    @staticmethod
    def mean(a):
        return np.mean(a)

    @staticmethod
    def sqrt(a):
        return np.sqrt(a)

    @staticmethod
    def get_default_memory_pool():
        class _Pool:
            def set_limit(self, size=0):
                return None

        return _Pool()


def _enable_gpu_with_stub(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _CpStub,
        raising=False,
    )
    def _init_stub(self):
        self.gpu_available = True
        return True

    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPUTechnicalProcessor._initialize_gpu",
        _init_stub,
    )


def test_calculate_multiple_indicators_gpu_matches_cpu(sample_frame, monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)

    indicators = ["sma", "ema", "rsi", "macd", "bollinger", "atr"]
    params = {
        "sma_window": 5,
        "ema_window": 5,
        "rsi_window": 5,
        "macd_fast": 3,
        "macd_slow": 6,
        "macd_signal": 3,
        "bb_window": 5,
        "bb_std": 2,
        "atr_window": 5,
    }

    gpu_df = processor.calculate_multiple_indicators_gpu(sample_frame, indicators, params)
    cpu_df = processor._calculate_multiple_indicators_cpu(sample_frame, indicators, params)

    assert set(cpu_df.columns) <= set(gpu_df.columns)
    assert len(gpu_df) == len(sample_frame)
    pd.testing.assert_series_equal(
        gpu_df[f"sma_{params['sma_window']}"],
        cpu_df[f"sma_{params['sma_window']}"],
        check_exact=False,
        atol=1e-6,
    )


def test_should_use_gpu_considers_memory(monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True, "optimization_level": "balanced"})

    monkeypatch.setattr(processor, "get_gpu_info", lambda: {"memory_usage": 90, "total_memory_gb": 8})
    assert processor._should_use_gpu(1000) is False

    monkeypatch.setattr(processor, "get_gpu_info", lambda: {"memory_usage": 10, "total_memory_gb": 2})
    assert processor._should_use_gpu(200_000) is False

    monkeypatch.setattr(processor, "get_gpu_info", lambda: {"memory_usage": 10, "total_memory_gb": 8})
    assert processor._should_use_gpu(600) is True


def test_get_gpu_info_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        False,
    )
    processor = GPUTechnicalProcessor(config={"use_gpu": False})
    info = processor.get_gpu_info()
    assert info["available"] is False


def test_calculate_atr_gpu_falls_back_to_cpu(sample_frame, monkeypatch):
    _force_cpu(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    data = sample_frame.copy()
    atr = processor.calculate_atr_gpu(data, window=5)
    expected = processor._calculate_atr_cpu(data, window=5)
    pd.testing.assert_series_equal(atr, expected, check_names=False)


def test_calculate_ema_gpu_matches_cpu(sample_frame, monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)

    ema_gpu = processor.calculate_ema_gpu(sample_frame, window=5)
    ema_cpu = processor._calculate_ema_cpu(sample_frame, window=5)
    assert np.allclose(ema_gpu.values, ema_cpu.values, atol=1e-1)


def test_get_gpu_info_handles_error(monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})

    class FailingCp(_CpStub):
        class cuda:
            @staticmethod
            def Device():
                raise RuntimeError("device error")

    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        FailingCp,
        raising=False,
    )
    info = processor.get_gpu_info()
    assert info["available"] is False


def test_clear_gpu_memory_invokes_optimizer(monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    called = {}

    def fake_optimize():
        called["done"] = True

    processor.gpu_available = True
    monkeypatch.setattr(processor, "_optimize_memory_usage", fake_optimize, raising=False)
    processor.clear_gpu_memory()
    assert called.get("done") is True


def test_initialize_gpu_failure_sets_cpu_mode(monkeypatch):
    class _FailingPool:
        def set_limit(self, size=0):
            return None

    class _FailingCp:
        class cuda:
            @staticmethod
            def is_available():
                return True

            class Device:
                def __str__(self):
                    return "StubDevice"

        class runtime:
            @staticmethod
            def memGetInfo():
                raise RuntimeError("no mem info")

        @staticmethod
        def get_default_memory_pool():
            return _FailingPool()

    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _FailingCp,
        raising=False,
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        True,
    )

    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    assert processor.gpu_available is False


def test_optimize_memory_access_handles_internal_errors(monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})

    def fail_preallocate():
        raise RuntimeError("preallocate boom")

    monkeypatch.setattr(processor, "_preallocate_memory_blocks", fail_preallocate)

    processor._optimize_memory_access()  # 不应抛出异常


def test_calculate_multiple_indicators_gpu_handles_exception(sample_frame, monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True})
    monkeypatch.setattr(processor, "_should_use_gpu", lambda size: True)

    class _FailingCp(_CpStub):
        @staticmethod
        def asarray(*args, **kwargs):
            raise RuntimeError("gpu failure")

    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.cp",
        _FailingCp,
        raising=False,
    )

    called = {}

    def fake_cpu(data, indicators, params=None):
        called["cpu"] = True
        return pd.DataFrame({"cpu_result": data["close"]})

    monkeypatch.setattr(
        processor,
        "_calculate_multiple_indicators_cpu",
        fake_cpu,
    )

    result = processor.calculate_multiple_indicators_gpu(sample_frame, ["sma"])
    assert called.get("cpu") is True
    assert "cpu_result" in result.columns


def test_should_use_gpu_returns_false_on_high_memory(monkeypatch):
    _enable_gpu_with_stub(monkeypatch)
    processor = GPUTechnicalProcessor(config={"use_gpu": True, "optimization_level": "balanced"})
    monkeypatch.setattr(
        processor,
        "get_gpu_info",
        lambda: {"memory_usage": 95, "total_memory_gb": 16},
    )
    assert processor._should_use_gpu(1000) is False


def test_calculate_multiple_indicators_cpu_returns_expected_columns(sample_frame):
    processor = GPUTechnicalProcessor(config={"use_gpu": False})
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
    cpu_df = processor._calculate_multiple_indicators_cpu(sample_frame, indicators, params)
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
    assert expected_cols <= set(cpu_df.columns)

