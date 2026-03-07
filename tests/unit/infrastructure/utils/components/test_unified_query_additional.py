#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_query 额外单测，聚焦缓存、分组与并发辅助逻辑。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.infrastructure.utils.components.unified_query import (
    QueryRequest,
    QueryResult,
    QueryType,
    StorageType,
    UnifiedQueryInterface,
)


@pytest.fixture
def unified_interface() -> UnifiedQueryInterface:
    interface = UnifiedQueryInterface(
        {
            "cache_enabled": True,
            "cache_ttl": 1,
            "max_concurrent_queries": 2,
            "max_async_workers": 2,
        }
    )
    # 使用回退缓存逻辑，避免依赖外部组件
    interface._cache_manager = None
    yield interface
    interface.query_executor.shutdown(wait=True)
    interface.async_executor.shutdown(wait=True)


def _make_request(
    query_type: QueryType,
    start: datetime | None = None,
    end: datetime | None = None,
    **extra: Any,
) -> QueryRequest:
    now = datetime.now()
    return QueryRequest(
        query_id=f"{query_type.value}_req",
        query_type=query_type,
        data_type=extra.get("data_type", "tick"),
        symbols=extra.get("symbols", ["AAA", "BBB"]),
        start_time=start or now - timedelta(minutes=1),
        end_time=end or now,
        aggregation=extra.get("aggregation"),
        filters=extra.get("filters"),
        storage_preference=extra.get("storage_preference"),
        limit=extra.get("limit"),
    )


def _make_result(query_id: str, success: bool = True) -> QueryResult:
    df = pd.DataFrame({"symbol": ["AAA"], "value": [1]})
    return QueryResult(query_id=query_id, success=success, data=df, record_count=len(df))


def test_generate_cache_key_stable(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(
        QueryType.REALTIME,
        aggregation={"func": "sum", "field": "value"},
        filters={"exchange": "SSE", "limit": 10},
    )
    key1 = unified_interface._generate_cache_key(request)
    key2 = unified_interface._generate_cache_key(request)
    assert key1 == key2


def test_cache_result_and_cleanup_expired(unified_interface: UnifiedQueryInterface) -> None:
    unified_interface.cache_ttl = 0  # 立即过期便于验证清理
    request = _make_request(QueryType.REALTIME)
    result = _make_result(request.query_id)
    unified_interface.query_cache["old"] = (result, time.time() - 10)

    unified_interface._cache_result(request, result)

    cache_key = unified_interface._generate_cache_key(request)
    assert "old" not in unified_interface.query_cache
    assert cache_key in unified_interface.query_cache


def test_get_cached_result_fallback_respects_ttl(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    result = _make_result(request.query_id)
    unified_interface.cache_ttl = 10
    cache_key = unified_interface._generate_cache_key(request)
    unified_interface.query_cache[cache_key] = (result, time.time())

    cached = unified_interface._get_cached_result(request)
    assert cached is result

    unified_interface.cache_ttl = 0
    unified_interface.query_cache[cache_key] = (result, time.time() - 5)
    expired = unified_interface._get_cached_result(request)
    assert expired is None
    assert cache_key not in unified_interface.query_cache


def test_merge_cross_storage_results_deduplicates(unified_interface: UnifiedQueryInterface) -> None:
    df1 = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "timestamp": [1, 2],
            "value": [10, 20],
        }
    )
    df2 = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "timestamp": [2],
            "value": [25],
        }
    )
    merged = unified_interface._merge_cross_storage_results(
        [
            (StorageType.INFLUXDB, df1),
            (StorageType.PARQUET, df2),
        ]
    )
    assert list(merged["data_source"]) == ["influxdb", "parquet"]
    assert merged.loc[merged["timestamp"] == 2, "value"].iloc[0] == 25


def test_group_requests_by_type(unified_interface: UnifiedQueryInterface) -> None:
    requests = [
        _make_request(QueryType.REALTIME),
        _make_request(QueryType.HISTORICAL),
        _make_request(QueryType.AGGREGATED),
        _make_request(QueryType.CROSS_STORAGE),
    ]
    grouped = unified_interface._group_requests_by_type(requests)
    assert set(grouped.keys()) == {"realtime", "historical", "aggregated", "cross_storage"}
    assert all(len(group) == 1 for group in grouped.values())


def test_process_query_results_handles_exceptions(unified_interface: UnifiedQueryInterface) -> None:
    requests = [
        _make_request(QueryType.REALTIME, symbols=["AAA"]),
        _make_request(QueryType.HISTORICAL, symbols=["BBB"]),
    ]
    grouped = unified_interface._group_requests_by_type(requests)
    raw_results = [
        _make_result(requests[0].query_id),
        RuntimeError("boom"),
    ]
    processed = unified_interface._process_query_results(raw_results, grouped)
    assert processed[0].success is True
    assert processed[1].success is False
    assert processed[1].error_message == "boom"


def test_handle_query_exception_returns_error_results(unified_interface: UnifiedQueryInterface) -> None:
    requests = [_make_request(QueryType.REALTIME), _make_request(QueryType.HISTORICAL)]
    results = unified_interface._handle_query_exception(RuntimeError("fail"), requests)
    assert len(results) == len(requests)
    assert all(not res.success and res.error_message == "fail" for res in results)


@pytest.mark.asyncio
async def test_execute_concurrent_queries_waits_all(unified_interface: UnifiedQueryInterface) -> None:
    tasks = [asyncio.sleep(0.01, result=i) for i in range(3)]
    results = await unified_interface._execute_concurrent_queries(tasks)
    assert results == [0, 1, 2]


def test_register_and_unregister_adapter(unified_interface: UnifiedQueryInterface) -> None:
    adapter = object()
    unified_interface.register_adapter(StorageType.REDIS, adapter)
    assert StorageType.REDIS in unified_interface.get_registered_adapters()
    unified_interface.unregister_adapter(StorageType.REDIS)
    assert StorageType.REDIS not in unified_interface.get_registered_adapters()


def test_get_query_stats_computes_averages(unified_interface: UnifiedQueryInterface) -> None:
    unified_interface._query_stats.update(
        {
            "total_queries": 4,
            "successful_queries": 3,
            "failed_queries": 1,
            "total_execution_time": 8.0,
            "cache_hits": 6,
            "cache_misses": 2,
        }
    )
    stats = unified_interface.get_query_stats()
    assert stats["avg_execution_time"] == 2.0
    assert stats["cache_hit_rate"] == pytest.approx(0.75)


def test_get_query_statistics_cache_fallback(unified_interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    result = _make_result(request.query_id)
    cache_key = unified_interface._generate_cache_key(request)
    unified_interface.query_cache[cache_key] = (result, time.time())

    stats = unified_interface.get_query_statistics()
    assert stats["cache_size"] == 1
    assert stats["cache_hit_rate"] == 0.0

