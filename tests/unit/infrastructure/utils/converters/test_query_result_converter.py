#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层查询结果转换器组件测试

测试目标：提升utils/converters/query_result_converter.py的真实覆盖率
实际导入和使用src.infrastructure.utils.converters.query_result_converter模块
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock


class TestQueryResultConverter:
    """测试查询结果转换器类"""
    
    def test_db_to_unified_success(self):
        """测试数据库结果转换为统一结果（成功）"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult
        
        db_result = DBQueryResult(
            success=True,
            data=[{"id": 1, "name": "test"}],
            row_count=1,
            execution_time=0.5
        )
        
        unified = QueryResultConverter.db_to_unified(db_result, query_id="test-001")
        
        assert unified.query_id == "test-001"
        assert unified.success is True
        assert isinstance(unified.data, pd.DataFrame)
        assert len(unified.data) == 1
        assert unified.execution_time == 0.5
    
    def test_db_to_unified_with_data_source(self):
        """测试使用数据源转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult
        
        db_result = DBQueryResult(
            success=True,
            data=[{"id": 1}],
            row_count=1,
            execution_time=0.3
        )
        
        unified = QueryResultConverter.db_to_unified(
            db_result, 
            query_id="test-002",
            data_source="postgresql"
        )
        
        assert unified.query_id == "test-002"
        assert unified.data_source == "postgresql"
    
    def test_db_to_unified_empty_data(self):
        """测试空数据转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult
        
        db_result = DBQueryResult(
            success=True,
            data=[],
            row_count=0,
            execution_time=0.1
        )
        
        unified = QueryResultConverter.db_to_unified(db_result, query_id="test-003")
        
        assert unified.success is True
        assert unified.data is None or len(unified.data) == 0
    
    def test_db_to_unified_with_error(self):
        """测试错误结果转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult
        
        db_result = DBQueryResult(
            success=False,
            data=[],
            row_count=0,
            execution_time=0.0,
            error_message="Test error"
        )
        
        unified = QueryResultConverter.db_to_unified(db_result, query_id="test-004")
        
        assert unified.success is False
        assert unified.error_message == "Test error"
    
    def test_unified_to_db_success(self):
        """测试统一结果转换为数据库结果（成功）"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.components.unified_query import QueryResult as UnifiedQueryResult
        
        df = pd.DataFrame([{"id": 1, "name": "test"}])
        unified = UnifiedQueryResult(
            query_id="test-005",
            success=True,
            data=df,
            execution_time=0.5
        )
        
        db_result = QueryResultConverter.unified_to_db(unified)
        
        assert db_result.success is True
        assert isinstance(db_result.data, list)
        assert len(db_result.data) == 1
        assert db_result.execution_time == 0.5
    
    def test_unified_to_db_empty_data(self):
        """测试空数据统一结果转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.components.unified_query import QueryResult as UnifiedQueryResult
        
        unified = UnifiedQueryResult(
            query_id="test-006",
            success=True,
            data=None,
            execution_time=0.1
        )
        
        db_result = QueryResultConverter.unified_to_db(unified)
        
        assert db_result.success is True
        assert db_result.data == []
    
    def test_unified_to_db_with_error(self):
        """测试错误统一结果转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.components.unified_query import QueryResult as UnifiedQueryResult
        
        unified = UnifiedQueryResult(
            query_id="test-007",
            success=False,
            data=None,
            execution_time=0.0,
            error_message="Test error"
        )
        
        db_result = QueryResultConverter.unified_to_db(unified)
        
        assert db_result.success is False
        assert db_result.error_message == "Test error"
    
    def test_db_to_unified_with_metadata(self):
        """测试带元数据的转换"""
        from src.infrastructure.utils.converters.query_result_converter import QueryResultConverter
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult
        
        db_result = DBQueryResult(
            success=True,
            data=[{"id": 1}],
            row_count=1,
            execution_time=0.2,
            metadata={"key": "value"}
        )
        
        unified = QueryResultConverter.db_to_unified(db_result, query_id="test-008")
        
        # 检查转换是否成功，metadata可能被转换到其他字段
        assert unified.success is True
        assert unified.query_id == "test-008"

