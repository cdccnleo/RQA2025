#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层查询执行器组件测试

测试目标：提升utils/components/query_executor.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.query_executor模块
"""

import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor


class TestQueryResult:
    """测试查询结果类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.query_executor import QueryResult
        
        result = QueryResult(query_id="test_id", data={"key": "value"})
        assert result.query_id == "test_id"
        assert result.data == {"key": "value"}
        assert result.success is True
        assert result.error is None
        assert isinstance(result.metadata, dict)
    
    def test_init_with_error(self):
        """测试使用错误初始化"""
        from src.infrastructure.utils.components.query_executor import QueryResult
        
        result = QueryResult(query_id="test_id", data=None, success=False, error="Test error")
        assert result.query_id == "test_id"
        assert result.data is None
        assert result.success is False
        assert result.error == "Test error"
    
    def test_add_metadata(self):
        """测试添加元数据"""
        from src.infrastructure.utils.components.query_executor import QueryResult
        
        result = QueryResult(query_id="test_id", data={})
        result.add_metadata(key1="value1", key2="value2")
        
        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == "value2"
    
    def test_add_metadata_multiple(self):
        """测试多次添加元数据"""
        from src.infrastructure.utils.components.query_executor import QueryResult
        
        result = QueryResult(query_id="test_id", data={})
        result.add_metadata(key1="value1")
        result.add_metadata(key2="value2")
        
        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == "value2"


class TestQueryExecutor:
    """测试查询执行器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor
        
        executor = QueryExecutor()
        assert executor is not None
        assert isinstance(executor.storage_adapters, dict)
        assert isinstance(executor.query_executor, ThreadPoolExecutor)
    
    def test_init_with_adapters(self):
        """测试使用适配器初始化"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        mock_adapter = MagicMock()
        adapters = {StorageType.INFLUXDB: mock_adapter}
        
        executor = QueryExecutor(storage_adapters=adapters)
        assert StorageType.INFLUXDB in executor.storage_adapters
        assert executor.storage_adapters[StorageType.INFLUXDB] == mock_adapter
    
    def test_init_with_max_workers(self):
        """测试使用最大工作线程数初始化"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor
        
        executor = QueryExecutor(max_workers=5)
        assert executor.query_executor._max_workers == 5
    
    def test_normalize_storage_type(self):
        """测试规范化存储类型"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        storage_type = QueryExecutor._normalize_storage_type(StorageType.INFLUXDB)
        assert storage_type == StorageType.INFLUXDB
        
        storage_type = QueryExecutor._normalize_storage_type("influxdb")
        assert storage_type == StorageType.INFLUXDB
    
    def test_normalize_storage_type_invalid(self):
        """测试规范化无效存储类型"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor
        
        with pytest.raises(ValueError):
            QueryExecutor._normalize_storage_type("invalid_type")
    
    def test_resolve_default_storage(self):
        """测试解析默认存储"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        executor = QueryExecutor()
        default = executor._resolve_default_storage()
        assert default == StorageType.INFLUXDB
        
        mock_adapter = MagicMock()
        adapters = {StorageType.REDIS: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        default = executor._resolve_default_storage()
        assert default == StorageType.REDIS
    
    def test_get_adapter(self):
        """测试获取适配器"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        mock_adapter = MagicMock()
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        adapter = executor._get_adapter(StorageType.INFLUXDB)
        assert adapter == mock_adapter
    
    def test_get_adapter_not_found(self):
        """测试获取不存在的适配器"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        executor = QueryExecutor()
        
        with pytest.raises(ValueError):
            executor._get_adapter(StorageType.INFLUXDB)
    
    def test_execute_realtime_query(self):
        """测试执行实时查询"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, QueryRequest, QueryType, StorageType
        
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {"data": "test"}
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        request = QueryRequest(
            query_id="test_id",
            query_type=QueryType.REALTIME,
            storage_type=StorageType.INFLUXDB,
            params={"key": "value"}
        )
        
        result = executor._execute_realtime_query(request)
        assert result.query_id == "test_id"
        assert result.data == {"data": "test"}
        assert result.metadata["query_type"] == "realtime"
        mock_adapter.query.assert_called_once_with({"key": "value"})
    
    def test_execute_historical_query(self):
        """测试执行历史查询"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, QueryRequest, QueryType, StorageType
        
        mock_adapter = MagicMock()
        mock_adapter.query_historical.return_value = {"data": "historical"}
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        request = QueryRequest(
            query_id="test_id",
            query_type=QueryType.HISTORICAL,
            storage_type=StorageType.INFLUXDB,
            params={"key": "value"}
        )
        
        result = executor._execute_historical_query(request)
        assert result.query_id == "test_id"
        assert result.data == {"data": "historical"}
        assert result.metadata["query_type"] == "historical"
        mock_adapter.query_historical.assert_called_once_with({"key": "value"})
    
    def test_execute_historical_query_no_method(self):
        """测试执行历史查询但适配器不支持"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, QueryRequest, QueryType, StorageType
        
        mock_adapter = MagicMock()
        del mock_adapter.query_historical
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        request = QueryRequest(
            query_id="test_id",
            query_type=QueryType.HISTORICAL,
            storage_type=StorageType.INFLUXDB,
            params={"key": "value"}
        )
        
        with pytest.raises(ValueError):
            executor._execute_historical_query(request)
    
    def test_execute_aggregated_query(self):
        """测试执行聚合查询"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, QueryRequest, QueryType, StorageType
        
        mock_adapter = MagicMock()
        mock_adapter.aggregate.return_value = {"data": "aggregated"}
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        request = QueryRequest(
            query_id="test_id",
            query_type=QueryType.AGGREGATED,
            storage_type=StorageType.INFLUXDB,
            params={"key": "value"}
        )
        
        result = executor._execute_aggregated_query(request)
        assert result.query_id == "test_id"
        assert result.data == {"data": "aggregated"}
        assert result.metadata["query_type"] == "aggregated"
        mock_adapter.aggregate.assert_called_once_with({"key": "value"})
    
    def test_execute_aggregated_query_no_method(self):
        """测试执行聚合查询但适配器不支持"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, QueryRequest, QueryType, StorageType
        
        mock_adapter = MagicMock()
        del mock_adapter.aggregate
        adapters = {StorageType.INFLUXDB: mock_adapter}
        executor = QueryExecutor(storage_adapters=adapters)
        
        request = QueryRequest(
            query_id="test_id",
            query_type=QueryType.AGGREGATED,
            storage_type=StorageType.INFLUXDB,
            params={"key": "value"}
        )
        
        with pytest.raises(ValueError):
            executor._execute_aggregated_query(request)
    
    def test_merge_results_empty(self):
        """测试合并空结果"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor
        
        executor = QueryExecutor()
        merged = executor._merge_results([])
        assert merged is None
    
    def test_merge_results_single(self):
        """测试合并单个结果"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        executor = QueryExecutor()
        results = [(StorageType.INFLUXDB, {"data": "test"})]
        merged = executor._merge_results(results)
        assert merged == {"data": "test"}
    
    def test_merge_results_multiple(self):
        """测试合并多个结果"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor, StorageType
        
        executor = QueryExecutor()
        results = [
            (StorageType.INFLUXDB, {"data": "test1"}),
            (StorageType.REDIS, {"data": "test2"})
        ]
        merged = executor._merge_results(results)
        assert isinstance(merged, list)
        assert len(merged) == 2
    
    def test_shutdown(self):
        """测试关闭执行器"""
        from src.infrastructure.utils.components.query_executor import QueryExecutor
        
        executor = QueryExecutor()
        executor.shutdown()
        
        # 验证执行器已关闭
        assert executor.query_executor._shutdown is True

