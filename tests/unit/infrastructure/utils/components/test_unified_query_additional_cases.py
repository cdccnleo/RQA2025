#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_query 执行与异常分支补充测试：覆盖实时/历史/聚合/跨存储查询以及批量异步查询主流程。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.infrastructure.utils.components.unified_query import (
    QueryRequest,
    QueryResult,
    QueryType,
    StorageType,
    UnifiedQueryInterface,
)


class _RealtimeAdapter:
    def __init__(self, df: Optional[pd.DataFrame] = None, raise_error: bool = False) -> None:
        self._df = df if df is not None else pd.DataFrame({"symbol": ["AAA"], "value": [1]})
        self._raise_error = raise_error

    def query_realtime(self, **_: Any) -> pd.DataFrame:
        if self._raise_error:
            raise RuntimeError("realtime boom")
        return self._df

    def query_aggregated(self, **__: Any) -> pd.DataFrame:
        if self._raise_error:
            raise RuntimeError("aggregate boom")
        return self._df


class _HistoricalAdapter:
    def __init__(self, df: Optional[pd.DataFrame] = None, raise_error: bool = False) -> None:
        self._df = df if df is not None else pd.DataFrame({"symbol": ["BBB"], "value": [2]})
        self._raise_error = raise_error

    def query_historical(self, **_: Any) -> pd.DataFrame:
        if self._raise_error:
            raise RuntimeError("historical boom")
        return self._df

    def query_aggregated(self, **__: Any) -> pd.DataFrame:
        if self._raise_error:
            raise RuntimeError("aggregate boom")
        return self._df


def _make_request(query_type: QueryType, **extra: Any) -> QueryRequest:
    now = datetime.now()
    return QueryRequest(
        query_id=f"{query_type.value}_exec",
        query_type=query_type,
        data_type=extra.get("data_type", "tick"),
        symbols=extra.get("symbols", ["AAA"]),
        start_time=extra.get("start_time", now - timedelta(minutes=1)),
        end_time=extra.get("end_time", now),
        aggregation=extra.get("aggregation"),
        filters=extra.get("filters"),
        limit=extra.get("limit"),
    )


@pytest.fixture
def unified_interface() -> UnifiedQueryInterface:
    interface = UnifiedQueryInterface(
        {
            "cache_enabled": False,
            "max_concurrent_queries": 2,
            "max_async_workers": 2,
        }
    )
    # 强制走内部缓存 fallback 逻辑，避免依赖外部组件
    interface._cache_manager = None
    yield interface
    interface.query_executor.shutdown(wait=True)
    interface.async_executor.shutdown(wait=True)


def test_execute_realtime_query_success_and_missing_adapter(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)

    # 成功路径
    unified_interface.storage_adapters = {StorageType.INFLUXDB: _RealtimeAdapter()}
    result = unified_interface._execute_realtime_query(request)
    assert result.success is True
    assert result.record_count == 1
    assert result.data_source == "influxdb"

    # 无适配器路径
    unified_interface.storage_adapters = {}
    missing = unified_interface._execute_realtime_query(request)
    assert missing.success is False
    assert missing.error_message == "未找到可用的数据源适配器"
    assert missing.data is not None and missing.data.empty


def test_execute_historical_query_handles_error(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.HISTORICAL)

    # 缺失适配器 -> 转换为失败结果
    unified_interface.storage_adapters = {}
    missing = unified_interface._execute_historical_query(request)
    assert missing.success is False
    assert missing.error_message == "Parquet适配器未初始化"
    assert isinstance(missing.data, pd.DataFrame)

    # 适配器抛错 -> 捕获并转换为失败结果
    unified_interface.storage_adapters = {StorageType.PARQUET: _HistoricalAdapter(raise_error=True)}
    failed = unified_interface._execute_historical_query(request)
    assert failed.success is False
    assert failed.error_message == "historical boom"
    assert failed.record_count == 0
    assert isinstance(failed.data, pd.DataFrame)


def test_execute_aggregated_query_branch_selection(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.AGGREGATED, aggregation={"real_time": True})
    unified_interface.storage_adapters = {
        StorageType.INFLUXDB: _RealtimeAdapter(),
        StorageType.PARQUET: _HistoricalAdapter(),
    }

    realtime_res = unified_interface._execute_aggregated_query(request)
    assert realtime_res.success is True
    assert realtime_res.data_source == "influxdb"

    # 切换到历史分支并触发异常
    request.aggregation = {"real_time": False}
    unified_interface.storage_adapters[StorageType.PARQUET] = _HistoricalAdapter(raise_error=True)
    aggregated_fail = unified_interface._execute_aggregated_query(request)
    assert aggregated_fail.success is False
    assert aggregated_fail.error_message == "aggregate boom"
    assert aggregated_fail.record_count == 0


def test_execute_cross_storage_success_and_all_fail(unified_interface: UnifiedQueryInterface) -> None:
    df_realtime = pd.DataFrame({"symbol": ["AAA"], "timestamp": [1], "value": [10]})
    df_history = pd.DataFrame({"symbol": ["AAA"], "timestamp": [2], "value": [20]})
    request = _make_request(QueryType.CROSS_STORAGE)

    unified_interface.storage_adapters = {
        StorageType.INFLUXDB: _RealtimeAdapter(df_realtime),
        StorageType.PARQUET: _HistoricalAdapter(df_history),
    }

    success = unified_interface._execute_cross_storage_query(request)
    assert success.success is True
    assert success.data_source == "cross_storage"
    assert success.record_count == 2
    assert list(success.data["value"]) == [10, 20]

    # 所有存储失败 -> 返回失败结果
    unified_interface.storage_adapters = {
        StorageType.INFLUXDB: _RealtimeAdapter(raise_error=True),
    }
    failure = unified_interface._execute_cross_storage_query(request)
    assert failure.success is False
    assert failure.error_message == "所有存储查询都失败"
    assert failure.record_count == 0


def test_query_single_storage_unknown_and_exception(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    adapter = _RealtimeAdapter()

    # 不支持的存储类型返回 None
    none_result = unified_interface._query_single_storage(adapter, StorageType.REDIS, request)
    assert none_result is None

    # 异常路径也返回 None
    boom_result = unified_interface._query_single_storage(
        _RealtimeAdapter(raise_error=True), StorageType.INFLUXDB, request
    )
    assert boom_result is None


@pytest.mark.asyncio
async def test_query_multiple_data_async_success_flow(unified_interface: UnifiedQueryInterface) -> None:
    requests = [
        _make_request(QueryType.REALTIME, symbols=["AAA"]),
        _make_request(QueryType.HISTORICAL, symbols=["BBB"]),
    ]

    async def _fake_query(request: QueryRequest) -> QueryResult:
        df = pd.DataFrame({"symbol": request.symbols, "value": [len(request.symbols)]})
        return QueryResult(query_id=request.query_id, success=True, data=df, record_count=len(df))

    unified_interface.query_data_async = _fake_query  # type: ignore[assignment]
    results = await unified_interface.query_multiple_data_async(requests)
    assert len(results) == 2
    assert all(result.success for result in results)


@pytest.mark.asyncio
async def test_query_multiple_data_async_empty_requests(unified_interface: UnifiedQueryInterface) -> None:
    # 空列表会直接返回 []
    assert await unified_interface.query_multiple_data_async([]) == []


def test_process_query_results_index_mismatch(unified_interface: UnifiedQueryInterface) -> None:
    request_a = _make_request(QueryType.REALTIME, symbols=["AAA"])
    request_b = _make_request(QueryType.REALTIME, symbols=["BBB"])
    grouped = {"realtime": [request_a, request_b]}
    raw_results = [
        QueryResult(query_id=request_a.query_id, success=True, data=pd.DataFrame(), record_count=0)
    ]
    processed = unified_interface._process_query_results(raw_results, grouped)
    assert len(processed) == 2
    assert processed[0].success is True
    assert processed[1].success is False
    assert "结果数量不足" in processed[1].error_message


def test_process_query_results_nested_non_query_result(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    grouped = {"realtime": [request]}
    raw_results = [[QueryResult(query_id=request.query_id, success=True, data=pd.DataFrame(), record_count=0)]]
    processed = unified_interface._process_query_results(raw_results, grouped)
    assert len(processed) == 1
    assert processed[0].success is False
    assert "[" in processed[0].error_message


def test_process_query_results_handles_none(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    grouped = {"realtime": [request]}
    processed = unified_interface._process_query_results(None, grouped)  # type: ignore[arg-type]
    assert len(processed) == 1
    assert processed[0].success is False
    assert processed[0].error_message == "None"


def test_execute_cross_storage_query_future_timeout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    unified_interface: UnifiedQueryInterface,
) -> None:
    class _FakeFuture:
        def result(self, timeout: float | None = None):
            raise TimeoutError("wait timeout")

    class _FakeExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, *args: Any, **kwargs: Any):
            return _FakeFuture()

    monkeypatch.setattr(
        "src.infrastructure.utils.components.unified_query.ThreadPoolExecutor", _FakeExecutor
    )
    unified_interface.storage_adapters = {StorageType.INFLUXDB: _RealtimeAdapter()}
    request = _make_request(QueryType.CROSS_STORAGE)

    caplog.set_level("WARNING")
    result = unified_interface._execute_cross_storage_query(request)
    assert result.success is False
    assert result.error_message == "所有存储查询都失败"
    assert any("wait timeout" in message for message in caplog.messages)


def test_execute_cross_storage_query_merge_failure(
    monkeypatch: pytest.MonkeyPatch, unified_interface: UnifiedQueryInterface
) -> None:
    df_realtime = pd.DataFrame({"symbol": ["AAA"], "timestamp": [1], "value": [10]})
    df_history = pd.DataFrame({"symbol": ["AAA"], "timestamp": [2], "value": [20]})
    unified_interface.storage_adapters = {
        StorageType.INFLUXDB: _RealtimeAdapter(df_realtime),
        StorageType.PARQUET: _HistoricalAdapter(df_history),
    }
    request = _make_request(QueryType.CROSS_STORAGE)

    def _boom_merge(results: List[Any]) -> pd.DataFrame:
        raise ValueError("merge failed")

    monkeypatch.setattr(
        unified_interface,
        "_merge_cross_storage_results",
        _boom_merge,  # type: ignore[assignment]
    )

    failure = unified_interface._execute_cross_storage_query(request)
    assert failure.success is False
    assert failure.record_count == 0
    assert failure.error_message == "merge failed"


def test_cache_result_fallback_on_cache_manager_error(
    caplog: pytest.LogCaptureFixture, unified_interface: UnifiedQueryInterface
) -> None:
    request = _make_request(QueryType.REALTIME)
    result = QueryResult(query_id=request.query_id, success=True, data=pd.DataFrame(), record_count=0)

    class _BrokenCacheManager:
        def cache_result(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("cache write fail")

        def cleanup_expired_cache(self) -> None:
            raise AssertionError("should not be called")

    unified_interface.cache_enabled = True
    unified_interface.cache_ttl = 1
    unified_interface._cache_manager = _BrokenCacheManager()
    unified_interface.query_cache["stale"] = (result, time.time() - 10)

    caplog.set_level("WARNING")
    unified_interface._cache_result(request, result)
    cache_key = unified_interface._generate_cache_key(request)

    assert cache_key in unified_interface.query_cache
    assert "stale" not in unified_interface.query_cache
    assert any("缓存管理器写入失败" in message for message in caplog.messages)


def test_log_query_success_handles_empty_processed(unified_interface: UnifiedQueryInterface, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    unified_interface._log_query_success([], [], {})
    assert any("批量异步查询完成: 0/0 成功" in message for message in caplog.messages)


def test_handle_query_exception_without_requests(unified_interface: UnifiedQueryInterface) -> None:
    results = unified_interface._handle_query_exception(RuntimeError("boom"), [])
    assert results == []


def test_get_query_statistics_without_cache_manager(unified_interface: UnifiedQueryInterface) -> None:
    unified_interface._cache_manager = None
    unified_interface.query_cache.clear()
    stats = unified_interface.get_query_statistics()
    assert stats["cache_size"] == 0
    assert stats["cache_hit_rate"] == 0.0

