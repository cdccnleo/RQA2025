#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层连接生命周期管理器组件测试

测试目标：提升utils/components/connection_lifecycle_manager.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.connection_lifecycle_manager模块
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock


class TestConnectionInfo:
    """测试连接信息数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionInfo
        
        conn_info = ConnectionInfo(
            connection_id="test_id",
            created_at=datetime.now(),
            last_used=datetime.now(),
            use_count=0,
            is_active=False
        )
        
        assert conn_info.connection_id == "test_id"
        assert conn_info.use_count == 0
        assert conn_info.is_active is False
        assert conn_info.error_count == 0
        assert conn_info.last_error is None


class TestConnectionLifecycleManager:
    """测试连接生命周期管理器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        manager = ConnectionLifecycleManager()
        assert manager.connection_factory is None
        assert manager.idle_timeout == 300.0
        assert manager.max_lifetime == 3600.0
        assert manager.max_usage is None
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        def factory():
            return MagicMock()
        
        manager = ConnectionLifecycleManager(
            connection_factory=factory,
            idle_timeout=600.0,
            max_lifetime=7200.0,
            max_usage=1000
        )
        
        assert manager.connection_factory == factory
        assert manager.idle_timeout == 600.0
        assert manager.max_lifetime == 7200.0
        assert manager.max_usage == 1000
    
    def test_create_connection(self):
        """测试创建连接"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        def factory():
            return MagicMock()
        
        manager = ConnectionLifecycleManager(connection_factory=factory)
        
        conn_info = manager.create_connection()
        assert conn_info is not None
        assert conn_info.connection_id is not None
        assert conn_info.connection is not None
        assert conn_info.use_count == 0
        assert conn_info.is_active is False
    
    def test_create_connection_no_factory(self):
        """测试无工厂函数创建连接"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        manager = ConnectionLifecycleManager()
        
        conn_info = manager.create_connection()
        assert conn_info is None
    
    def test_destroy_connection(self):
        """测试销毁连接"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager, ConnectionInfo
        
        mock_conn = MagicMock()
        conn_info = ConnectionInfo(
            connection_id="test_id",
            created_at=datetime.now(),
            last_used=datetime.now(),
            use_count=0,
            is_active=False,
            connection=mock_conn
        )
        
        manager = ConnectionLifecycleManager()
        result = manager.destroy_connection(conn_info)
        
        assert result is True
        mock_conn.close.assert_called_once()
    
    def test_destroy_connection_no_close_method(self):
        """测试销毁无close方法的连接"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager, ConnectionInfo
        
        mock_conn = MagicMock()
        del mock_conn.close
        
        conn_info = ConnectionInfo(
            connection_id="test_id",
            created_at=datetime.now(),
            last_used=datetime.now(),
            use_count=0,
            is_active=False,
            connection=mock_conn
        )
        
        manager = ConnectionLifecycleManager()
        result = manager.destroy_connection(conn_info)
        
        assert result is True
    
    def test_cleanup_expired_connections(self):
        """测试清理过期连接"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager, ConnectionInfo
        
        manager = ConnectionLifecycleManager(idle_timeout=1.0)
        
        old_time = datetime.now() - timedelta(seconds=2)
        new_time = datetime.now()
        
        expired_conn = ConnectionInfo(
            connection_id="expired",
            created_at=old_time,
            last_used=old_time,
            use_count=0,
            is_active=False
        )
        
        active_conn = ConnectionInfo(
            connection_id="active",
            created_at=new_time,
            last_used=new_time,
            use_count=0,
            is_active=True
        )
        
        connections = [expired_conn, active_conn]
        expired = manager.cleanup_expired_connections(connections)
        
        assert len(expired) == 1
        assert expired[0].connection_id == "expired"
    
    def test_ensure_min_connections(self):
        """测试确保最小连接数"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        manager = ConnectionLifecycleManager()
        
        connections = [MagicMock() for _ in range(3)]
        needed = manager.ensure_min_connections(connections, min_size=5)
        
        assert needed == 2
    
    def test_ensure_min_connections_sufficient(self):
        """测试连接数已足够"""
        from src.infrastructure.utils.components.connection_lifecycle_manager import ConnectionLifecycleManager
        
        manager = ConnectionLifecycleManager()
        
        connections = [MagicMock() for _ in range(5)]
        needed = manager.ensure_min_connections(connections, min_size=3)
        
        assert needed == 0

