import logging
import sys
import types
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.features.core import parallel_feature_processor as parallel


class _SequentialFuture:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._done = False
        self._result = None

    def result(self, timeout=None):
        if not self._done:
            self._result = self._fn(*self._args, **self._kwargs)
            self._done = True
        return self._result

    def done(self):
        return self._done


class _SequentialExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.submissions: List[_SequentialFuture] = []
        self.shutdown_called = False

    def submit(self, fn, *args, **kwargs):
        future = _SequentialFuture(fn, *args, **kwargs)
        self.submissions.append(future)
        return future

    def shutdown(self, wait=True):
        self.shutdown_called = True


def _sequential_as_completed(futures, timeout=None):
    if isinstance(futures, dict):
        iterator = list(futures.keys())
    else:
        iterator = list(futures)
    for future in iterator:
        yield future


class FeatureTypeStub:
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    HIGH_FREQUENCY = "high_frequency"
    CUSTOM = "custom"


class SecretsStub:
    @staticmethod
    def normal(loc, scale, size):
        return np.full(size, loc, dtype=float)

    @staticmethod
    def uniform(low, high, size):
        return np.linspace(low, high, num=size, dtype=float)


@pytest.fixture(autouse=True)
def patch_parallel_environment(monkeypatch):
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", _SequentialExecutor)
    monkeypatch.setattr(parallel, "as_completed", _sequential_as_completed)
    monkeypatch.setattr(parallel, "logger", logging.getLogger("parallel-test"))
    monkeypatch.setattr(parallel, "logger_unified", logging.getLogger("parallel-test"))
    monkeypatch.setattr(parallel, "FeatureType", FeatureTypeStub)
    monkeypatch.setattr(parallel.np, "secrets", SecretsStub(), raising=False)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "close": np.linspace(10, 15, 12),
            "open": np.linspace(9, 14, 12),
            "high": np.linspace(11, 16, 12),
            "low": np.linspace(8, 13, 12),
            "volume": np.arange(100, 112),
        }
    )


def _make_config(feature_type, *, technical=None, technical_params=None, sentiment=None):
    technical_params = technical_params or types.SimpleNamespace(
        sma_periods=[3],
        rsi_period=3,
        macd_fast=2,
        macd_slow=3,
        macd_signal=1,
    )
    return types.SimpleNamespace(
        feature_types=[feature_type],
        technical_indicators=technical or [],
        technical_params=technical_params,
        sentiment_types=sentiment or [],
    )


def test_process_features_parallel_combines_chunks(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(
        feature_engine=object(),
        config=parallel.ParallelConfig(n_jobs=2, chunk_size=5, timeout=5),
    )

    monkeypatch.setattr(
        processor,
        "_process_chunk",
        lambda chunk, configs: chunk.assign(processed=chunk["close"] + 1),
    )

    result = processor.process_features_parallel(sample_data, feature_configs=[_make_config(FeatureTypeStub.CUSTOM)])

    assert "processed" in result.columns
    assert len(result) == len(sample_data)
    assert processor.stats["success_count"] == 1
    assert processor.stats["error_count"] == 0


def test_process_features_parallel_returns_original_on_empty(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(
        feature_engine=object(),
        config=parallel.ParallelConfig(n_jobs=2, chunk_size=5, timeout=5),
    )

    monkeypatch.setattr(
        processor,
        "_process_chunk",
        lambda chunk, configs: pd.DataFrame(),  # 返回空结果触发fallback
    )

    result = processor.process_features_parallel(sample_data, feature_configs=[_make_config(FeatureTypeStub.CUSTOM)])

    assert result.equals(sample_data)
    assert processor.stats["error_count"] == 1


def test_split_data_warns_and_defaults(monkeypatch, sample_data, caplog):
    processor = parallel.ParallelFeatureProcessor(feature_engine=object(), config=parallel.ParallelConfig(chunk_size=0))

    with caplog.at_level("WARNING"):
        chunks = processor._split_data(sample_data, chunk_size=0)

    assert len(chunks) == len(sample_data)  # chunk_size降级为1
    assert "分块大小不能为0" in caplog.text


def test_process_chunk_routes_to_specialized_calculators(sample_data):
    extra_rows = pd.DataFrame([sample_data.iloc[-1]] * 15).reset_index(drop=True)
    extended = pd.concat([sample_data.reset_index(drop=True), extra_rows], ignore_index=True)

    technical = _make_config(FeatureTypeStub.TECHNICAL, technical=["SMA", "MACD"])
    sentiment = _make_config(FeatureTypeStub.SENTIMENT, sentiment=["sentiment_main"])
    hf = _make_config(FeatureTypeStub.HIGH_FREQUENCY, sentiment=["momentum", "volatility"])
    generic = _make_config(FeatureTypeStub.CUSTOM)

    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    result = processor._process_chunk(extended, [technical, sentiment, hf, generic])

    expected_cols = {
        "SMA_sma",
        "MACD_macd",
        "MACD_signal",
        "MACD_histogram",
        "sentiment_main_score",
        "sentiment_main_confidence",
        "momentum_20",
        "volatility_20",
        "price_mean",
    }
    assert expected_cols <= set(result.columns)


def test_batch_process_symbols_handles_failures(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(feature_engine=object(), config=parallel.ParallelConfig(chunk_size=4))

    def fake_symbol(symbol, data, configs):
        if symbol == "FAIL":
            raise ValueError("boom")
        return data.assign(done=True)

    monkeypatch.setattr(processor, "_process_single_symbol", fake_symbol)

    data_dict = {"AAA": sample_data.copy(), "FAIL": sample_data.copy()}
    results = processor.batch_process_symbols(["AAA", "FAIL"], data_dict, feature_configs=[_make_config(FeatureTypeStub.CUSTOM)])

    assert "AAA" in results and "FAIL" in results
    assert "done" in results["AAA"].columns
    assert results["FAIL"].equals(sample_data)


def test_get_performance_stats_computes_averages():
    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    processor.stats = {
        "total_processed": 4,
        "total_time": 8.0,
        "success_count": 3,
        "error_count": 1,
    }

    stats = processor.get_performance_stats()
    assert stats["avg_time_per_record"] == 2.0
    assert stats["success_rate"] == pytest.approx(0.75)


def test_close_shuts_executor():
    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    processor.close()
    assert getattr(processor.executor, "shutdown_called", False) is True


def test_process_features_parallel_handles_chunk_exception(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(
        feature_engine=object(),
        config=parallel.ParallelConfig(n_jobs=2, chunk_size=4, timeout=5),
    )

    def boom(chunk, configs):
        raise RuntimeError("boom")

    monkeypatch.setattr(processor, "_process_chunk", boom)

    result = processor.process_features_parallel(sample_data, feature_configs=[_make_config(FeatureTypeStub.CUSTOM)])

    assert result.equals(sample_data)
    assert processor.stats["success_count"] == 0
    assert processor.stats["error_count"] == 1


def test_process_features_parallel_timeout_triggers_fallback(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(
        feature_engine=object(),
        config=parallel.ParallelConfig(n_jobs=2, chunk_size=6, timeout=1),
    )

    monkeypatch.setattr(
        processor,
        "_process_chunk",
        lambda chunk, configs: chunk.assign(flag=True),
    )

    def raise_timeout(futures, timeout=None):
        raise TimeoutError("timeout")

    monkeypatch.setattr(parallel, "as_completed", raise_timeout)

    result = processor.process_features_parallel(sample_data, feature_configs=[_make_config(FeatureTypeStub.CUSTOM)])

    assert "flag" in result.columns
    assert processor.stats["success_count"] == 1


def test_calculate_technical_features_warning_on_missing_columns(sample_data, caplog):
    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    technical = _make_config(FeatureTypeStub.TECHNICAL, technical=["SMA"])
    bad_data = sample_data.drop(columns=["close"])

    with caplog.at_level("WARNING"):
        features = processor._calculate_technical_features(bad_data, technical)

    assert features == {}
    assert "计算技术指标" in caplog.text


def test_calculate_generic_features_handles_error(sample_data, caplog):
    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    with caplog.at_level("WARNING"):
        features = processor._calculate_generic_features(sample_data[["close"]], _make_config(FeatureTypeStub.CUSTOM))
    assert features == {}
    assert "计算通用特征失败" in caplog.text


def test_process_single_symbol_logs_error(monkeypatch, caplog, sample_data):
    processor = parallel.ParallelFeatureProcessor(feature_engine=object())
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(processor, "process_features_parallel", boom)

    with caplog.at_level("ERROR"):
        result = processor._process_single_symbol("AAA", sample_data, [_make_config(FeatureTypeStub.CUSTOM)])

    assert result is None
    assert "处理股票AAA特征失败" in caplog.text


def test_batch_process_symbols_timeout_returns_original(sample_data, monkeypatch):
    processor = parallel.ParallelFeatureProcessor(
        feature_engine=object(),
        config=parallel.ParallelConfig(n_jobs=2, chunk_size=6, timeout=1),
    )

    monkeypatch.setattr(
        processor,
        "_process_single_symbol",
        lambda symbol, data, configs: data.assign(symbol=symbol),
    )

    def raise_timeout(futures, timeout=None):
        raise TimeoutError("timeout")

    monkeypatch.setattr(parallel, "as_completed", raise_timeout)

    data_dict = {"AAA": sample_data.copy(), "BBB": sample_data.copy()}
    results = processor.batch_process_symbols(["AAA", "BBB"], data_dict, [_make_config(FeatureTypeStub.CUSTOM)])

    assert set(results.keys()) == {"AAA", "BBB"}
    assert all("symbol" in df.columns for df in results.values())

