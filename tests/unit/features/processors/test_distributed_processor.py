import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from src.features.processors.distributed_processor import (
    CachingProcessor,
    DistributedFeatureProcessor,
    MemoryOptimizedProcessor,
)


class _SimpleProcessor:
    def __init__(self):
        self.calls = 0

    def process(self, data, config=None):
        self.calls += 1
        return data.assign(processed=data["value"] + 1)

    def get_processor_type(self):
        return "simple"


@pytest.fixture(autouse=True)
def mock_unified_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.distributed_processor.get_unified_logger",
        lambda name: logging.getLogger(name),
    )


@pytest.fixture
def sample_frame():
    return pd.DataFrame({"value": range(6)})


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


def _patch_sequential_executor(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.distributed_processor.ProcessPoolExecutor",
        _SequentialExecutor,
    )
    monkeypatch.setattr(
        "src.features.processors.distributed_processor.as_completed",
        lambda futures, timeout=None: list(futures),
    )

def test_process_in_parallel_batches_and_stats(monkeypatch, sample_frame):
    processor = DistributedFeatureProcessor(max_workers=2, chunk_size=2)
    worker = _SimpleProcessor()
    _patch_sequential_executor(monkeypatch)

    result = processor.process_in_parallel(sample_frame, worker)

    assert len(result) == len(sample_frame)
    assert "processed" in result.columns
    assert processor.stats["chunks_processed"] == 3
    assert processor.stats["errors"] == 0
    assert processor.stats["total_processed"] == len(result)


def test_process_in_parallel_handles_chunk_error(monkeypatch, sample_frame):
    processor = DistributedFeatureProcessor(max_workers=2, chunk_size=2)

    class FailingProcessor(_SimpleProcessor):
        def process(self, data, config=None):
            super().process(data, config)
            if 2 in data["value"].values:
                raise ValueError("boom")
            return data.assign(tag="ok")

    _patch_sequential_executor(monkeypatch)

    worker = FailingProcessor()
    result = processor.process_in_parallel(sample_frame, worker)

    assert not result.empty
    assert len(result) == 4  # 一个数据块失败，剩余两块成功
    assert processor.stats["errors"] == 0
    assert processor.stats["chunks_processed"] == 3


def test_process_chunk_failure_returns_empty(sample_frame):
    processor = DistributedFeatureProcessor()

    class BadProcessor:
        def process(self, data, config=None):
            raise RuntimeError("failure")

    chunk = sample_frame.iloc[:2]
    result = processor._process_chunk(chunk, BadProcessor())
    assert result.empty


def test_process_in_parallel_global_failure(monkeypatch, sample_frame):
    processor = DistributedFeatureProcessor()
    monkeypatch.setattr(processor, "_split_data", lambda *_: (_ for _ in ()).throw(RuntimeError("split fail")))
    result = processor.process_in_parallel(sample_frame, _SimpleProcessor())
    assert result.empty
    assert processor.stats["errors"] == 1


def test_memory_optimizer_uses_batches_when_exceeding_limit(monkeypatch, sample_frame):
    optimizer = MemoryOptimizedProcessor(max_memory_mb=1)
    worker = _SimpleProcessor()

    monkeypatch.setattr(
        optimizer,
        "_estimate_memory_usage",
        lambda data: 10.0,
    )

    captured = {}

    def fake_batches(data, processor, config):
        captured["called"] = True
        return processor.process(data, config)

    monkeypatch.setattr(optimizer, "_process_in_batches", fake_batches)

    result = optimizer.process_with_memory_optimization(sample_frame, worker)
    assert captured.get("called") is True
    assert not result.empty


def test_memory_optimizer_direct_process_when_within_limit(monkeypatch, sample_frame):
    optimizer = MemoryOptimizedProcessor(max_memory_mb=1000)
    worker = _SimpleProcessor()

    monkeypatch.setattr(
        optimizer,
        "_estimate_memory_usage",
        lambda data: 5.0,
    )

    result = optimizer.process_with_memory_optimization(sample_frame, worker)
    assert worker.calls == 1
    assert "processed" in result.columns


def test_memory_optimizer_returns_empty_on_exception(monkeypatch, sample_frame):
    optimizer = MemoryOptimizedProcessor(max_memory_mb=1000)
    worker = _SimpleProcessor()

    def boom(*_args, **_kwargs):
        raise RuntimeError("estimate fail")

    monkeypatch.setattr(optimizer, "_estimate_memory_usage", boom)
    result = optimizer.process_with_memory_optimization(sample_frame, worker)
    assert result.empty


def test_caching_processor_hits_cache(tmp_path, sample_frame):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=100)
    worker = _SimpleProcessor()

    class SimpleConfig:
        def to_dict(self):
            return {"mode": "test"}

    config = SimpleConfig()

    first = cache.process_with_cache(sample_frame, worker, config=config)
    second = cache.process_with_cache(sample_frame, worker, config=config)

    assert len(first) == len(sample_frame)
    assert len(second) == len(sample_frame)
    assert cache.cache_stats["misses"] == 1
    assert cache.cache_stats["hits"] == 1
    assert worker.calls == 1  # 第二次命中缓存


def test_caching_processor_load_failure(tmp_path, sample_frame):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=100)
    worker = _SimpleProcessor()

    class SimpleConfig:
        def to_dict(self):
            return {"mode": "test"}

    config = SimpleConfig()
    cache.process_with_cache(sample_frame, worker, config=config)

    # 写入坏数据触发加载失败
    for cache_file in tmp_path.glob("*.pkl"):
        cache_file.write_bytes(b"corrupt")

    result = cache.process_with_cache(sample_frame, worker, config=config)
    assert not result.empty
    assert cache.cache_stats["misses"] == 2
    assert cache.cache_stats["hits"] == 0
    assert worker.calls == 2


def test_caching_processor_empty_input(tmp_path):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=100)
    worker = _SimpleProcessor()
    result = cache.process_with_cache(pd.DataFrame(), worker)
    assert result.empty


def test_process_in_parallel_with_empty_frame_returns_empty():
    processor = DistributedFeatureProcessor()
    worker = _SimpleProcessor()
    empty = pd.DataFrame({"value": []})
    result = processor.process_in_parallel(empty, worker)
    assert result.empty
    assert processor.stats["chunks_processed"] == 0


def test_distributed_stats_helpers(monkeypatch, sample_frame):
    processor = DistributedFeatureProcessor(chunk_size=3)
    worker = _SimpleProcessor()
    _patch_sequential_executor(monkeypatch)
    processor.process_in_parallel(sample_frame, worker)
    stats = processor.get_performance_stats()
    assert stats["total_processed"] == 6
    assert stats["throughput"] >= 0
    assert stats["success_rate"] >= 0
    processor.reset_stats()
    reset = processor.get_performance_stats()
    assert reset["total_processed"] == 0
    assert reset["chunks_processed"] == 0
    assert reset["success_rate"] == 0


def test_split_data_respects_chunk_size(sample_frame):
    processor = DistributedFeatureProcessor(chunk_size=4)
    chunks = processor._split_data(sample_frame)
    assert len(chunks) == 2
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 2


def test_memory_optimizer_fallback_estimate(monkeypatch, sample_frame):
    optimizer = MemoryOptimizedProcessor(max_memory_mb=1000)
    worker = _SimpleProcessor()

    def boom(self, deep=False):
        raise BaseException("fail")

    monkeypatch.setattr(pd.DataFrame, "memory_usage", boom, raising=True)
    result = optimizer.process_with_memory_optimization(sample_frame, worker)
    assert "processed" in result.columns


def test_memory_optimizer_batch_error_is_skipped(monkeypatch, sample_frame):
    optimizer = MemoryOptimizedProcessor(max_memory_mb=1)

    class FailingProcessor(_SimpleProcessor):
        def process(self, data, config=None):
            super().process(data, config)
            if len(data) < 3:
                raise RuntimeError("batch fail")
            return data.assign(ok=True)

    monkeypatch.setattr(optimizer, "_estimate_memory_usage", lambda data: 10.0)
    result = optimizer.process_with_memory_optimization(sample_frame, FailingProcessor())
    assert result.empty


def test_caching_processor_cleanup_triggered(tmp_path, sample_frame, monkeypatch):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=1)
    worker = _SimpleProcessor()
    cache.cache_stats["size_mb"] = 2
    called = {}

    def fake_cleanup():
        called["done"] = True

    monkeypatch.setattr(cache, "_cleanup_cache", fake_cleanup)
    cache.process_with_cache(sample_frame, worker)
    assert called.get("done") is True


def test_caching_processor_cleanup_removes_old_files(tmp_path):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=100)
    for idx in range(12):
        file = tmp_path / f"cache_{idx}.pkl"
        file.write_bytes(b"data")
        os.utime(file, (idx, idx))

    cache._cleanup_cache()
    remaining = list(tmp_path.glob("*.pkl"))
    assert len(remaining) <= 10


def test_caching_processor_get_stats_hit_rate(tmp_path, sample_frame):
    cache = CachingProcessor(cache_dir=tmp_path, max_cache_size_mb=100)
    worker = _SimpleProcessor()
    cache.process_with_cache(sample_frame, worker)
    cache.process_with_cache(sample_frame, worker)
    stats = cache.get_cache_stats()
    assert 0.0 <= stats["hit_rate"] <= 1.0
