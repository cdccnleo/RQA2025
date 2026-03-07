#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层接口定义测试

测试目标：提升utils/core/interfaces.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.interfaces模块
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestConnectionStatus:
    """测试连接状态枚举"""
    
    def test_enum_values(self):
        """测试枚举值"""
        from src.infrastructure.utils.core.interfaces import ConnectionStatus
        
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTING.value == "connecting"
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.ERROR.value == "error"
    
    def test_enum_list(self):
        """测试枚举列表"""
        from src.infrastructure.utils.core.interfaces import ConnectionStatus
        
        all_statuses = list(ConnectionStatus)
        assert len(all_statuses) == 4
        assert ConnectionStatus.DISCONNECTED in all_statuses
        assert ConnectionStatus.CONNECTING in all_statuses
        assert ConnectionStatus.CONNECTED in all_statuses
        assert ConnectionStatus.ERROR in all_statuses


class TestQueryResult:
    """测试查询结果数据类"""
    
    def test_query_result_creation(self):
        """测试查询结果创建"""
        from src.infrastructure.utils.core.interfaces import QueryResult
        
        result = QueryResult(
            success=True,
            data=[{"id": 1, "name": "test"}],
            row_count=1,
            execution_time=0.1
        )
        
        assert result.success is True
        assert len(result.data) == 1
        assert result.row_count == 1
        assert result.execution_time == 0.1
        assert result.error_message is None
    
    def test_query_result_with_error(self):
        """测试带错误的查询结果"""
        from src.infrastructure.utils.core.interfaces import QueryResult
        
        result = QueryResult(
            success=False,
            data=[],
            row_count=0,
            execution_time=0.0,
            error_message="Query failed"
        )
        
        assert result.success is False
        assert result.error_message == "Query failed"


class TestWriteResult:
    """测试写入结果数据类"""
    
    def test_write_result_creation(self):
        """测试写入结果创建"""
        from src.infrastructure.utils.core.interfaces import WriteResult
        
        result = WriteResult(
            success=True,
            affected_rows=5,
            execution_time=0.2
        )
        
        assert result.success is True
        assert result.affected_rows == 5
        assert result.execution_time == 0.2
        assert result.error_message is None
    
    def test_write_result_with_error(self):
        """测试带错误的写入结果"""
        from src.infrastructure.utils.core.interfaces import WriteResult
        
        result = WriteResult(
            success=False,
            affected_rows=0,
            execution_time=0.0,
            error_message="Write failed"
        )
        
        assert result.success is False
        assert result.error_message == "Write failed"


class TestHealthCheckResult:
    """测试健康检查结果数据类"""
    
    def test_health_check_result_creation(self):
        """测试健康检查结果创建"""
        from src.infrastructure.utils.core.interfaces import HealthCheckResult
        
        result = HealthCheckResult(
            is_healthy=True,
            response_time=0.05,
            connection_count=10
        )
        
        assert result.is_healthy is True
        assert result.response_time == 0.05
        assert result.connection_count == 10
        assert result.error_message is None
        assert result.details is None
    
    def test_health_check_result_with_details(self):
        """测试带详细信息的健康检查结果"""
        from src.infrastructure.utils.core.interfaces import HealthCheckResult
        
        details = {"cpu": 50.0, "memory": 60.0}
        result = HealthCheckResult(
            is_healthy=True,
            response_time=0.05,
            connection_count=10,
            details=details
        )
        
        assert result.details == details


class TestIDatabaseAdapter:
    """测试数据库适配器接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        with pytest.raises(TypeError):
            IDatabaseAdapter()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        assert hasattr(IDatabaseAdapter, 'connect')
        assert hasattr(IDatabaseAdapter, 'disconnect')
        assert hasattr(IDatabaseAdapter, 'execute_query')
        assert hasattr(IDatabaseAdapter, 'execute_write')
        assert hasattr(IDatabaseAdapter, 'batch_write')
        assert hasattr(IDatabaseAdapter, 'health_check')
        assert hasattr(IDatabaseAdapter, 'connection_status')
        assert hasattr(IDatabaseAdapter, 'get_connection_info')
    
    def test_validate_config(self):
        """测试配置验证"""
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class ConcreteAdapter(IDatabaseAdapter):
            def connect(self, config):
                return self._validate_config(config)
            
            def disconnect(self):
                return True
            
            def execute_query(self, query, params=None):
                from src.infrastructure.utils.core.interfaces import QueryResult
                return QueryResult(True, [], 0, 0.0)
            
            def execute_write(self, query, params=None):
                from src.infrastructure.utils.core.interfaces import WriteResult
                return WriteResult(True, 0, 0.0)
            
            def batch_write(self, data_list):
                from src.infrastructure.utils.core.interfaces import WriteResult
                return WriteResult(True, 0, 0.0)
            
            def health_check(self):
                from src.infrastructure.utils.core.interfaces import HealthCheckResult
                return HealthCheckResult(True, 0.0, 0)
            
            @property
            def connection_status(self):
                from src.infrastructure.utils.core.interfaces import ConnectionStatus
                return ConnectionStatus.CONNECTED
            
            def get_connection_info(self):
                return {}
        
        adapter = ConcreteAdapter()
        
        # 测试有效配置
        assert adapter.connect({"host": "localhost", "port": 5432}) is True
        
        # 测试无效配置（非字典）
        assert adapter.connect("invalid") is False
        
        # 测试空配置
        assert adapter.connect({}) is False
    
    def test_validate_query(self):
        """测试查询验证"""
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class ConcreteAdapter(IDatabaseAdapter):
            def connect(self, config):
                return True
            
            def disconnect(self):
                return True
            
            def execute_query(self, query, params=None):
                return self._validate_query(query)
            
            def execute_write(self, query, params=None):
                from src.infrastructure.utils.core.interfaces import WriteResult
                return WriteResult(True, 0, 0.0)
            
            def batch_write(self, data_list):
                from src.infrastructure.utils.core.interfaces import WriteResult
                return WriteResult(True, 0, 0.0)
            
            def health_check(self):
                from src.infrastructure.utils.core.interfaces import HealthCheckResult
                return HealthCheckResult(True, 0.0, 0)
            
            @property
            def connection_status(self):
                from src.infrastructure.utils.core.interfaces import ConnectionStatus
                return ConnectionStatus.CONNECTED
            
            def get_connection_info(self):
                return {}
        
        adapter = ConcreteAdapter()
        
        # 测试有效查询
        assert adapter.execute_query("SELECT * FROM users") is True
        
        # 测试无效查询（非字符串）
        assert adapter.execute_query(123) is False
        
        # 测试空查询
        assert adapter.execute_query("") is False
        assert adapter.execute_query("   ") is False
    
    def test_validate_data_list(self):
        """测试数据列表验证"""
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class ConcreteAdapter(IDatabaseAdapter):
            def connect(self, config):
                return True
            
            def disconnect(self):
                return True
            
            def execute_query(self, query, params=None):
                from src.infrastructure.utils.core.interfaces import QueryResult
                return QueryResult(True, [], 0, 0.0)
            
            def execute_write(self, query, params=None):
                from src.infrastructure.utils.core.interfaces import WriteResult
                return WriteResult(True, 0, 0.0)
            
            def batch_write(self, data_list):
                return self._validate_data_list(data_list)
            
            def health_check(self):
                from src.infrastructure.utils.core.interfaces import HealthCheckResult
                return HealthCheckResult(True, 0.0, 0)
            
            @property
            def connection_status(self):
                from src.infrastructure.utils.core.interfaces import ConnectionStatus
                return ConnectionStatus.CONNECTED
            
            def get_connection_info(self):
                return {}
        
        adapter = ConcreteAdapter()
        
        # 测试有效数据列表
        assert adapter.batch_write([{"id": 1}, {"id": 2}]) is True
        
        # 测试空列表（有效）
        assert adapter.batch_write([]) is True
        
        # 测试无效数据列表（非列表）
        assert adapter.batch_write("invalid") is False
        
        # 测试无效数据列表（包含非字典项）
        assert adapter.batch_write([{"id": 1}, "invalid"]) is False


class TestITransaction:
    """测试事务接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.core.interfaces import ITransaction
        
        with pytest.raises(TypeError):
            ITransaction()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.core.interfaces import ITransaction
        
        assert hasattr(ITransaction, 'begin')
        assert hasattr(ITransaction, 'commit')
        assert hasattr(ITransaction, 'rollback')
        assert hasattr(ITransaction, 'is_active')


class TestIConcurrencyController:
    """测试并发控制器接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.core.interfaces import IConcurrencyController
        
        with pytest.raises(TypeError):
            IConcurrencyController()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.core.interfaces import IConcurrencyController
        
        assert hasattr(IConcurrencyController, 'acquire')
        assert hasattr(IConcurrencyController, 'release')
        assert hasattr(IConcurrencyController, 'get_active_count')
        assert hasattr(IConcurrencyController, 'max_concurrent')
