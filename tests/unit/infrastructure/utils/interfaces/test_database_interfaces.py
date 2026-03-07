#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层数据库接口组件测试

测试目标：提升utils/interfaces/database_interfaces.py的真实覆盖率
实际导入和使用src.infrastructure.utils.interfaces.database_interfaces模块
"""

import pytest
from unittest.mock import MagicMock


class TestConnectionStatus:
    """测试连接状态枚举"""
    
    def test_connection_status_enum(self):
        """测试连接状态枚举值"""
        from src.infrastructure.utils.interfaces.database_interfaces import ConnectionStatus
        
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTING.value == "connecting"
        assert ConnectionStatus.ERROR.value == "error"


class TestQueryResult:
    """测试查询结果数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
        
        result = QueryResult(
            success=True,
            data=[{"id": 1, "name": "test"}],
            row_count=1,
            execution_time=0.5
        )
        
        assert result.success is True
        assert len(result.data) == 1
        assert result.row_count == 1
        assert result.execution_time == 0.5
        assert result.error_message is None
    
    def test_init_with_error(self):
        """测试使用错误初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
        
        result = QueryResult(
            success=False,
            data=[],
            row_count=0,
            execution_time=0.0,
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
    
    def test_post_init_auto_row_count(self):
        """测试自动计算行数"""
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
        
        result = QueryResult(
            success=True,
            data=[{"id": 1}, {"id": 2}, {"id": 3}],
            row_count=0,
            execution_time=0.1
        )
        
        assert result.row_count == 3
    
    def test_init_with_metadata(self):
        """测试使用元数据初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
        
        result = QueryResult(
            success=True,
            data=[],
            row_count=0,
            execution_time=0.1,
            metadata={"key": "value"}
        )
        
        assert result.metadata is not None
        assert result.metadata["key"] == "value"
    
    def test_init_with_query_id(self):
        """测试使用查询ID初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
        
        result = QueryResult(
            success=True,
            data=[],
            row_count=0,
            execution_time=0.1,
            query_id="test-001"
        )
        
        assert result.query_id == "test-001"


class TestWriteResult:
    """测试写入结果数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
        
        result = WriteResult(
            success=True,
            affected_rows=5,
            execution_time=0.3
        )
        
        assert result.success is True
        assert result.affected_rows == 5
        assert result.execution_time == 0.3
        assert result.error_message is None
    
    def test_init_with_error(self):
        """测试使用错误初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
        
        result = WriteResult(
            success=False,
            affected_rows=0,
            execution_time=0.0,
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
    
    def test_post_init_rows_affected_compatibility(self):
        """测试rows_affected兼容性"""
        from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
        
        result = WriteResult(
            success=True,
            affected_rows=0,
            execution_time=0.1,
            rows_affected=10
        )
        
        assert result.affected_rows == 10
    
    def test_init_with_insert_id(self):
        """测试使用插入ID初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
        
        result = WriteResult(
            success=True,
            affected_rows=1,
            execution_time=0.2,
            insert_id=123
        )
        
        assert result.insert_id == 123
    
    def test_init_with_metadata(self):
        """测试使用元数据初始化"""
        from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
        
        result = WriteResult(
            success=True,
            affected_rows=1,
            execution_time=0.1,
            metadata={"key": "value"}
        )
        
        assert result.metadata is not None
        assert result.metadata["key"] == "value"


class TestIDatabaseAdapter:
    """测试数据库适配器接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.interfaces.database_interfaces import IDatabaseAdapter
        
        with pytest.raises(TypeError):
            IDatabaseAdapter()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.interfaces.database_interfaces import IDatabaseAdapter
        
        # 检查接口是否有抽象方法
        assert hasattr(IDatabaseAdapter, 'connect')
        assert hasattr(IDatabaseAdapter, 'disconnect')
        assert hasattr(IDatabaseAdapter, 'execute_query')
        # 注意：接口可能使用不同的方法名，如execute_write或batch_write

