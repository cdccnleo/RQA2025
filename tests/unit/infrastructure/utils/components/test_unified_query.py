#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层统一查询接口组件测试

测试目标：提升utils/components/unified_query.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.unified_query模块
"""

import pytest
from datetime import datetime, timedelta


class TestUnifiedQueryConstants:
    """测试统一查询常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.unified_query import UnifiedQueryConstants
        
        assert UnifiedQueryConstants.DEFAULT_QUERY_TIMEOUT == 30
        assert UnifiedQueryConstants.MAX_QUERY_TIMEOUT == 300
        assert UnifiedQueryConstants.MIN_QUERY_TIMEOUT == 5
        assert UnifiedQueryConstants.DEFAULT_EXECUTION_TIME == 0.0
        assert UnifiedQueryConstants.DEFAULT_RECORD_COUNT == 0
        assert UnifiedQueryConstants.DEFAULT_MAX_CONCURRENT_QUERIES == 10
        assert UnifiedQueryConstants.MAX_CONCURRENT_QUERIES == 50
        assert UnifiedQueryConstants.DEFAULT_CACHE_TTL == 300
        assert UnifiedQueryConstants.CACHE_TTL_SECONDS == 300
        assert UnifiedQueryConstants.QUERY_CACHE_SIZE == 1000
        assert UnifiedQueryConstants.DEFAULT_TIME_WINDOW_MINUTES == 5
        assert UnifiedQueryConstants.QUERY_ID_LENGTH == 8
        assert UnifiedQueryConstants.DEFAULT_BATCH_SIZE == 1000
        assert UnifiedQueryConstants.MAX_BATCH_SIZE == 10000
        assert UnifiedQueryConstants.MIN_BATCH_SIZE == 100
        assert UnifiedQueryConstants.DEFAULT_MAX_RETRIES == 3
        assert UnifiedQueryConstants.DEFAULT_RETRY_DELAY == 1.0
        assert UnifiedQueryConstants.PERFORMANCE_WARNING_THRESHOLD == 10.0
        assert UnifiedQueryConstants.PERFORMANCE_CRITICAL_THRESHOLD == 30.0
        assert UnifiedQueryConstants.DEFAULT_MAX_ASYNC_WORKERS == 20
        assert UnifiedQueryConstants.ASYNC_TIMEOUT_BUFFER == 5.0
        assert UnifiedQueryConstants.ASYNC_BATCH_SIZE == 1000


class TestQueryType:
    """测试查询类型枚举"""
    
    def test_query_type_enum(self):
        """测试查询类型枚举值"""
        from src.infrastructure.utils.components.unified_query import QueryType
        
        assert QueryType.REALTIME.value == "realtime"
        assert QueryType.HISTORICAL.value == "historical"
        assert QueryType.AGGREGATED.value == "aggregated"
        assert QueryType.CROSS_STORAGE.value == "cross_storage"
        assert QueryType.SELECT.value == "select"
        assert QueryType.INSERT.value == "insert"
        assert QueryType.UPDATE.value == "update"
        assert QueryType.DELETE.value == "delete"


class TestStorageType:
    """测试存储类型枚举"""
    
    def test_storage_type_enum(self):
        """测试存储类型枚举值"""
        from src.infrastructure.utils.components.unified_query import StorageType
        
        assert StorageType.INFLUXDB.value == "influxdb"
        assert StorageType.PARQUET.value == "parquet"
        assert StorageType.REDIS.value == "redis"
        assert StorageType.HYBRID.value == "hybrid"


class TestQueryRequest:
    """测试查询请求"""
    
    def test_query_request_init(self):
        """测试查询请求初始化"""
        from src.infrastructure.utils.components.unified_query import QueryRequest, QueryType, StorageType
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        request = QueryRequest(
            query_id="test_query",
            query_type=QueryType.REALTIME,
            data_type="tick",
            symbols=["600519"],
            start_time=start_time,
            end_time=end_time
        )
        
        assert request.query_id == "test_query"
        assert request.query_type == QueryType.REALTIME
        assert request.data_type == "tick"
        assert request.symbols == ["600519"]
        assert request.start_time == start_time
        assert request.end_time == end_time
        assert request.storage_preference is None
        assert request.timeout == 30
    
    def test_query_request_with_preferences(self):
        """测试带偏好的查询请求"""
        from src.infrastructure.utils.components.unified_query import QueryRequest, QueryType, StorageType
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        request = QueryRequest(
            query_id="test_query",
            query_type=QueryType.HISTORICAL,
            data_type="kline",
            symbols=["600519", "000001"],
            start_time=start_time,
            end_time=end_time,
            storage_preference=StorageType.PARQUET,
            limit=1000,
            timeout=60
        )
        
        assert request.storage_preference == StorageType.PARQUET
        assert request.limit == 1000
        assert request.timeout == 60

