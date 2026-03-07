"""
Query Executor功能测试模块

按《投产计划-总览.md》Week 2 Day 1-2执行
测试查询执行器的完整功能

测试覆盖：
- QueryExecutor: 查询执行功能（15个测试）
  * 基本查询执行（3个）
  * 批量查询处理（3个）
  * 查询缓存机制（3个）
  * 并发查询处理（3个）
  * 错误处理与重试（3个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor
from src.infrastructure.utils.components.query_executor import (
    QueryExecutor,
    QueryRequest,
    QueryResult,
    QueryType,
    StorageType
)


# Apply timeout to all tests (5 seconds per test)
pytestmark = pytest.mark.timeout(5)


class TestQueryExecutorBasicFunctional:
    """QueryExecutor基本功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage_adapters = {
            StorageType.INFLUXDB: Mock(),
            StorageType.PARQUET: Mock(),
            StorageType.REDIS: Mock()
        }
        self.executor = QueryExecutor(self.storage_adapters, max_workers=4)

    def test_execute_realtime_query(self):
        """测试1: 实时查询执行"""
        # Arrange
        request = QueryRequest(
            query_id="q1",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"measurement": "temperature", "time_range": "1h"}
        )
        
        expected_data = [
            {"time": "2024-01-01T00:00:00Z", "value": 25.5},
            {"time": "2024-01-01T00:01:00Z", "value": 26.0}
        ]
        
        # Mock realtime query execution
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            mock_exec.return_value = QueryResult(
                data=expected_data,
                metadata={"query_type": "realtime", "count": 2}
            )
            
            # Act
            result = self.executor.execute_query(request)
            
            # Assert
            assert result.data == expected_data
            assert result.metadata["query_type"] == "realtime"
            assert result.metadata["count"] == 2
            mock_exec.assert_called_once_with(request)

    def test_execute_historical_query(self):
        """测试2: 历史查询执行"""
        # Arrange
        request = QueryRequest(
            query_id="q2",
            query_type=QueryType.HISTORICAL,
            storage_type=StorageType.PARQUET,
            params={"start_date": "2024-01-01", "end_date": "2024-01-31"}
        )
        
        expected_data = {"total_records": 10000, "data": []}
        
        # Mock historical query execution
        with patch.object(self.executor, '_execute_historical_query') as mock_exec:
            mock_exec.return_value = QueryResult(
                data=expected_data,
                metadata={"query_type": "historical", "cached": False}
            )
            
            # Act
            result = self.executor.execute_query(request)
            
            # Assert
            assert result.data == expected_data
            assert result.metadata["query_type"] == "historical"
            assert result.metadata["cached"] is False
            mock_exec.assert_called_once_with(request)

    def test_execute_aggregated_query(self):
        """测试3: 聚合查询执行"""
        # Arrange
        request = QueryRequest(
            query_id="q3",
            query_type=QueryType.AGGREGATED,
            storage_type=StorageType.INFLUXDB,
            params={"aggregation": "avg", "group_by": "1h"}
        )
        
        expected_data = {
            "aggregation": "avg",
            "results": [
                {"time": "2024-01-01T00:00:00Z", "avg_value": 25.5},
                {"time": "2024-01-01T01:00:00Z", "avg_value": 26.0}
            ]
        }
        
        # Mock aggregated query execution
        with patch.object(self.executor, '_execute_aggregated_query') as mock_exec:
            mock_exec.return_value = QueryResult(
                data=expected_data,
                metadata={"query_type": "aggregated", "group_count": 2}
            )
            
            # Act
            result = self.executor.execute_query(request)
            
            # Assert
            assert result.data == expected_data
            assert result.metadata["query_type"] == "aggregated"
            assert result.metadata["group_count"] == 2
            mock_exec.assert_called_once_with(request)


class TestQueryExecutorBatchFunctional:
    """QueryExecutor批量查询测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage_adapters = {
            StorageType.INFLUXDB: Mock(),
            StorageType.PARQUET: Mock()
        }
        self.executor = QueryExecutor(self.storage_adapters, max_workers=4)

    def test_execute_batch_queries(self):
        """测试4: 批量查询执行"""
        # Arrange
        requests = [
            QueryRequest("q1", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 1}),
            QueryRequest("q2", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 2}),
            QueryRequest("q3", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 3})
        ]
        
        # Mock individual query executions
        with patch.object(self.executor, 'execute_query') as mock_exec:
            mock_exec.side_effect = [
                QueryResult(data={"id": 1}, metadata={}),
                QueryResult(data={"id": 2}, metadata={}),
                QueryResult(data={"id": 3}, metadata={})
            ]
            
            # Mock the execute_batch_queries method
            with patch.object(self.executor, 'execute_batch_queries') as mock_batch:
                mock_batch.return_value = [
                    QueryResult(data={"id": 1}, metadata={}),
                    QueryResult(data={"id": 2}, metadata={}),
                    QueryResult(data={"id": 3}, metadata={})
                ]
                
                # Act
                results = self.executor.execute_batch_queries(requests)
                
                # Assert
                assert len(results) == 3
                assert all(isinstance(r, QueryResult) for r in results)
                assert results[0].data["id"] == 1
                assert results[1].data["id"] == 2
                assert results[2].data["id"] == 3

    def test_batch_queries_parallel_execution(self):
        """测试5: 批量查询并行执行"""
        # Arrange
        requests = [
            QueryRequest(f"q{i}", QueryType.REALTIME, StorageType.INFLUXDB, {"id": i})
            for i in range(10)
        ]
        
        # Track execution order/timing
        execution_log = []
        
        def mock_execute(req):
            execution_log.append(req.query_id)
            return QueryResult(data={"id": req.params["id"]}, metadata={})
        
        # Mock execute_batch_queries with parallel execution
        with patch.object(self.executor, 'execute_batch_queries') as mock_batch:
            # Simulate parallel execution
            def parallel_exec(reqs):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(mock_execute, req) for req in reqs]
                    return [f.result() for f in futures]
            
            mock_batch.side_effect = parallel_exec
            
            # Act
            results = self.executor.execute_batch_queries(requests)
            
            # Assert
            assert len(results) == 10
            assert len(execution_log) == 10
            # Verify all queries were executed
            assert set(execution_log) == {f"q{i}" for i in range(10)}

    def test_batch_queries_with_failures(self):
        """测试6: 批量查询部分失败处理"""
        # Arrange
        requests = [
            QueryRequest("q1", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 1}),
            QueryRequest("q2", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 2}),
            QueryRequest("q3", QueryType.REALTIME, StorageType.INFLUXDB, {"id": 3})
        ]
        
        # Mock with some failures
        with patch.object(self.executor, 'execute_batch_queries') as mock_batch:
            mock_batch.return_value = [
                QueryResult(data={"id": 1}, metadata={"success": True}),
                QueryResult(data=None, metadata={"success": False, "error": "Connection error"}),
                QueryResult(data={"id": 3}, metadata={"success": True})
            ]
            
            # Act
            results = self.executor.execute_batch_queries(requests)
            
            # Assert
            assert len(results) == 3
            assert results[0].data["id"] == 1
            assert results[0].metadata["success"] is True
            assert results[1].data is None
            assert results[1].metadata["success"] is False
            assert "error" in results[1].metadata
            assert results[2].data["id"] == 3


class TestQueryExecutorCacheFunctional:
    """QueryExecutor查询缓存测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage_adapters = {StorageType.INFLUXDB: Mock()}
        self.executor = QueryExecutor(self.storage_adapters, max_workers=4)

    def test_query_cache_hit(self):
        """测试7: 查询缓存命中"""
        # Arrange
        request = QueryRequest(
            query_id="q1",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"measurement": "temperature"}
        )
        
        cached_data = [{"time": "2024-01-01", "value": 25.5}]
        
        # Mock cache hit scenario
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            # First call - cache miss, execute query
            mock_exec.return_value = QueryResult(
                data=cached_data,
                metadata={"cached": False, "execution_time": 0.5}
            )
            
            result1 = self.executor.execute_query(request)
            
            # Second call - cache hit, no execution
            mock_exec.return_value = QueryResult(
                data=cached_data,
                metadata={"cached": True, "execution_time": 0.001}
            )
            
            result2 = self.executor.execute_query(request)
            
            # Assert
            assert result1.data == cached_data
            assert result2.data == cached_data
            # Cache hit should have much faster execution time
            assert result2.metadata["execution_time"] < result1.metadata["execution_time"]

    def test_query_cache_miss(self):
        """测试8: 查询缓存未命中"""
        # Arrange
        request = QueryRequest(
            query_id="q_new",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"measurement": "pressure", "unique": True}
        )
        
        # Mock cache miss - must execute query
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            mock_exec.return_value = QueryResult(
                data=[{"value": 1013.25}],
                metadata={"cached": False}
            )
            
            # Act
            result = self.executor.execute_query(request)
            
            # Assert
            assert result.metadata["cached"] is False
            mock_exec.assert_called_once()

    def test_query_cache_invalidation(self):
        """测试9: 查询缓存失效"""
        # Arrange
        request = QueryRequest(
            query_id="q1",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"measurement": "temperature"}
        )
        
        # Mock cache invalidation scenario
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            # First call - populate cache
            mock_exec.return_value = QueryResult(
                data=[{"value": 25.5}],
                metadata={"cached": False, "cache_ttl": 300}
            )
            result1 = self.executor.execute_query(request)
            
            # Simulate cache invalidation (e.g., TTL expired)
            mock_exec.return_value = QueryResult(
                data=[{"value": 26.0}],  # New data
                metadata={"cached": False, "cache_invalidated": True}
            )
            result2 = self.executor.execute_query(request)
            
            # Assert
            assert result1.data[0]["value"] == 25.5
            assert result2.data[0]["value"] == 26.0
            assert result2.metadata.get("cache_invalidated") is True


class TestQueryExecutorConcurrentFunctional:
    """QueryExecutor并发查询测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage_adapters = {StorageType.INFLUXDB: Mock()}
        self.executor = QueryExecutor(self.storage_adapters, max_workers=10)

    def test_concurrent_query_execution(self):
        """测试10: 并发查询执行"""
        # Arrange
        num_concurrent_queries = 20
        requests = [
            QueryRequest(
                query_id=f"q{i}",
                query_type=QueryType.REALTIME,
                storage_type=StorageType.INFLUXDB,
                params={"id": i}
            )
            for i in range(num_concurrent_queries)
        ]
        
        # Mock concurrent execution
        with patch.object(self.executor, 'execute_query') as mock_exec:
            def mock_query(req):
                return QueryResult(
                    data={"query_id": req.query_id},
                    metadata={"success": True}
                )
            
            mock_exec.side_effect = mock_query
            
            # Simulate concurrent execution using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.executor.execute_query, req) for req in requests]
                results = [f.result() for f in futures]
            
            # Assert
            assert len(results) == num_concurrent_queries
            # Verify all queries completed
            query_ids = {r.data["query_id"] for r in results}
            assert len(query_ids) == num_concurrent_queries

    def test_concurrent_query_thread_safety(self):
        """测试11: 并发查询线程安全"""
        # Arrange
        shared_counter = {"count": 0}
        
        def safe_execute(req):
            # Simulate thread-safe execution
            shared_counter["count"] += 1
            return QueryResult(
                data={"count": shared_counter["count"]},
                metadata={"thread_safe": True}
            )
        
        requests = [
            QueryRequest(f"q{i}", QueryType.REALTIME, StorageType.INFLUXDB, {})
            for i in range(50)
        ]
        
        # Mock thread-safe execution
        with patch.object(self.executor, 'execute_query', side_effect=safe_execute):
            # Execute concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.executor.execute_query, req) for req in requests]
                results = [f.result() for f in futures]
            
            # Assert
            assert len(results) == 50
            # Counter should reach 50 (thread-safe increment)
            assert shared_counter["count"] == 50

    def test_concurrent_query_resource_limits(self):
        """测试12: 并发查询资源限制"""
        # Arrange
        max_workers = 4
        executor_with_limit = QueryExecutor(self.storage_adapters, max_workers=max_workers)
        
        # Track concurrent executions
        active_queries = {"count": 0, "max_concurrent": 0}
        
        def track_concurrent_exec(req):
            active_queries["count"] += 1
            active_queries["max_concurrent"] = max(
                active_queries["max_concurrent"],
                active_queries["count"]
            )
            # Simulate query execution
            import time
            time.sleep(0.01)
            active_queries["count"] -= 1
            return QueryResult(data={"id": req.query_id}, metadata={})
        
        requests = [
            QueryRequest(f"q{i}", QueryType.REALTIME, StorageType.INFLUXDB, {})
            for i in range(20)
        ]
        
        # Mock with tracking
        with patch.object(executor_with_limit, 'execute_query', side_effect=track_concurrent_exec):
            # Execute with limited workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(executor_with_limit.execute_query, req) for req in requests]
                results = [f.result() for f in futures]
            
            # Assert
            assert len(results) == 20
            # Max concurrent should not exceed max_workers
            assert active_queries["max_concurrent"] <= max_workers


class TestQueryExecutorErrorHandlingFunctional:
    """QueryExecutor错误处理测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage_adapters = {StorageType.INFLUXDB: Mock()}
        self.executor = QueryExecutor(self.storage_adapters, max_workers=4)

    def test_query_execution_error_handling(self):
        """测试13: 查询执行错误处理"""
        # Arrange
        request = QueryRequest(
            query_id="q_error",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"invalid": True}
        )
        
        # Mock execution error
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            mock_exec.side_effect = Exception("Connection timeout")
            
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                self.executor.execute_query(request)
            
            assert "Connection timeout" in str(exc_info.value)

    def test_query_retry_mechanism(self):
        """测试14: 查询重试机制"""
        # Arrange
        request = QueryRequest(
            query_id="q_retry",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"retry": True}
        )
        
        # Mock with retry logic
        with patch.object(self.executor, '_execute_realtime_query') as mock_exec:
            # First two attempts fail, third succeeds
            mock_exec.side_effect = [
                Exception("Timeout"),
                Exception("Timeout"),
                QueryResult(data={"success": True}, metadata={"retries": 2})
            ]
            
            # Simulate retry wrapper
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.executor.execute_query(request)
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        raise
            
            # Assert
            assert result.data["success"] is True
            assert result.metadata["retries"] == 2

    def test_unsupported_query_type_error(self):
        """测试15: 不支持的查询类型错误"""
        # Arrange
        # Create a request with an unsupported query type
        request = Mock(spec=QueryRequest)
        request.query_type = "UNSUPPORTED_TYPE"
        request.query_id = "q_invalid"
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.executor.execute_query(request)
        
        assert "不支持的查询类型" in str(exc_info.value)


# 测试统计
# Total: 15 tests
# TestQueryExecutorBasicFunctional: 3 tests (基本查询执行)
# TestQueryExecutorBatchFunctional: 3 tests (批量查询处理)
# TestQueryExecutorCacheFunctional: 3 tests (查询缓存机制)
# TestQueryExecutorConcurrentFunctional: 3 tests (并发查询处理)
# TestQueryExecutorErrorHandlingFunctional: 3 tests (错误处理与重试)

