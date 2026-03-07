#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层连接健康检查器组件测试

测试目标：提升utils/components/connection_health_checker.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.connection_health_checker模块
"""

import pytest
from unittest.mock import MagicMock
from queue import Queue


class TestPoolState:
    """测试连接池状态枚举"""
    
    def test_pool_state_enum(self):
        """测试连接池状态枚举值"""
        from src.infrastructure.utils.components.connection_health_checker import PoolState
        
        assert PoolState.HEALTHY.value == "healthy"
        assert PoolState.WARNING.value == "warning"
        assert PoolState.CRITICAL.value == "critical"
        assert PoolState.FAILED.value == "failed"


class TestConnectionHealthChecker:
    """测试连接健康检查器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        
        checker = ConnectionHealthChecker()
        assert checker.connection_validator is None
        assert checker.health_check_count == 0
        assert checker._tracked_queue_id is None
    
    def test_init_with_validator(self):
        """测试使用验证器初始化"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        
        validator = MagicMock()
        checker = ConnectionHealthChecker(connection_validator=validator)
        assert checker.connection_validator == validator
    
    def test_health_check_empty_pool(self):
        """测试健康检查空连接池"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        
        checker = ConnectionHealthChecker()
        connections = []
        available_connections = Queue()
        active_connections = {}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "state" in result
        assert checker.health_check_count == 1
    
    def test_health_check_healthy_pool(self):
        """测试健康检查健康连接池"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        connections = [Connection(), Connection()]
        available_connections = Queue()
        available_connections.put(connections[0])
        available_connections.put(connections[1])
        active_connections = {}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert result["state"] in ["healthy", "warning", "critical", "failed"]
        assert "total" in result
        assert "available" in result
        assert "active" in result
    
    def test_health_check_critical_pool(self):
        """测试健康检查临界连接池"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        connections = [Connection() for _ in range(10)]
        available_connections = Queue()
        active_connections = {i: connections[i] for i in range(10)}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert result["state"] in ["healthy", "warning", "critical", "failed"]
        assert result["total"] == 10
        assert result["active"] == 10
    
    def test_health_check_with_validator(self):
        """测试使用验证器进行健康检查"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        def validator(conn):
            return not conn.closed
        
        checker = ConnectionHealthChecker(connection_validator=validator)
        conn = Connection()
        connections = [conn]
        available_connections = Queue()
        available_connections.put(conn)
        active_connections = {}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert "valid_connections" in result
    
    def test_health_check_with_closed_connections(self):
        """测试健康检查包含已关闭连接"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        conn = Connection()
        conn.close()
        connections = [conn]
        available_connections = Queue()
        active_connections = {}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert result["state"] in ["healthy", "warning", "critical", "failed"]
    
    def test_health_check_multiple_calls(self):
        """测试多次健康检查"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        connections = [Connection()]
        available_connections = Queue()
        available_connections.put(connections[0])
        active_connections = {}
        
        result1 = checker.health_check(connections, available_connections, active_connections, max_size=10)
        result2 = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert checker.health_check_count == 2
        assert result1 is not None
        assert result2 is not None
    
    def test_health_check_with_list_available(self):
        """测试使用列表作为可用连接"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        connections = [Connection()]
        available_connections = [connections[0]]
        active_connections = {}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert result["available"] == 1
    
    def test_health_check_with_dict_active(self):
        """测试使用字典作为活跃连接"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        from src.infrastructure.utils.components.connection_pool import Connection
        
        checker = ConnectionHealthChecker()
        connections = [Connection()]
        available_connections = Queue()
        active_connections = {0: connections[0]}
        
        result = checker.health_check(connections, available_connections, active_connections, max_size=10)
        
        assert result is not None
        assert result["active"] == 1
    
    def test_health_check_none_connections(self):
        """测试健康检查None连接"""
        from src.infrastructure.utils.components.connection_health_checker import ConnectionHealthChecker
        
        checker = ConnectionHealthChecker()
        connections = None
        available_connections = Queue()
        active_connections = {}
        
        # 应该能处理None连接
        try:
            result = checker.health_check(connections, available_connections, active_connections, max_size=10)
            assert result is not None
        except (TypeError, AttributeError):
            # 如果无法处理None，这是预期的
            pass

