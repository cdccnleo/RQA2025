#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层连接池组件测试

测试目标：提升utils/components/connection_pool.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.connection_pool模块
"""

import pytest
import time
import threading
from unittest.mock import MagicMock


class TestConnection:
    """测试连接类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_pool import Connection
        
        conn = Connection()
        assert conn.connection_id > 0
        assert conn.last_used > 0
        assert conn.usage_count == 0
        assert conn.closed is False
    
    def test_close(self):
        """测试关闭连接"""
        from src.infrastructure.utils.components.connection_pool import Connection
        
        conn = Connection()
        conn.close()
        
        assert conn.closed is True
    
    def test_connection_id_unique(self):
        """测试连接ID唯一性"""
        from src.infrastructure.utils.components.connection_pool import Connection
        
        conn1 = Connection()
        conn2 = Connection()
        
        assert conn1.connection_id != conn2.connection_id


class TestConnectionPool:
    """测试连接池类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10, leak_detection=False)
        assert pool._max_size == 10
        assert pool._idle_timeout == 300.0
        assert pool._max_usage == 1000
        assert pool._leak_detection is False
        assert pool._created_count == 0
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(
            max_size=20,
            idle_timeout=600.0,
            max_usage=500,
            leak_detection=True
        )
        
        assert pool._max_size == 20
        assert pool._idle_timeout == 600.0
        assert pool._max_usage == 500
        assert pool._leak_detection is True
    
    def test_get_connection(self):
        """测试获取连接"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=5)
        conn = pool.get_connection()
        
        assert conn is not None
        assert conn.usage_count == 1
        assert pool._active_connections == 1
    
    def test_get_connection_timeout(self):
        """测试获取连接超时"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        from queue import Empty
        
        pool = ConnectionPool(max_size=1)
        conn1 = pool.get_connection()
        
        # 尝试获取第二个连接，应该超时
        with pytest.raises(Empty):
            pool.get_connection(timeout=0.1)
    
    def test_release(self):
        """测试释放连接"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=5)
        conn = pool.get_connection()
        
        assert pool._active_connections == 1
        
        pool.release(conn)
        
        assert pool._active_connections == 0
    
    def test_release_none(self):
        """测试释放None连接"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool()
        pool.release(None)
        
        # 应该不会抛出异常
        assert True
    
    def test_release_closed_connection(self):
        """测试释放已关闭的连接"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=5)
        conn = pool.get_connection()
        conn.close()
        
        pool.release(conn)
        
        # 已关闭的连接不应该被放回池中
        assert pool._active_connections == 0
    
    def test_put_connection(self):
        """测试put_connection方法（release的别名）"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=5)
        conn = pool.get_connection()
        
        pool.put_connection(conn)
        
        assert pool._active_connections == 0
    
    def test_health_check(self):
        """测试健康检查"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10, leak_detection=True)
        conn = pool.get_connection()
        
        health = pool.health_check()
        
        assert health["total"] == 1
        assert health["active"] == 1
        assert health["idle"] == 0
        assert health["created"] == 1
        assert "config" in health
    
    def test_get_status(self):
        """测试获取状态"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10, leak_detection=True)
        conn = pool.get_connection()
        
        status = pool.get_status()
        
        assert status["pool_size"] == 0
        assert status["active_connections"] == 1
        assert status["total"] == 1
        assert status["max_size"] == 10
        assert status["created_count"] == 1
        assert status["leak_count"] == 1
    
    def test_get_size(self):
        """测试获取连接池大小"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10)
        conn = pool.get_connection()
        
        size = pool.get_size()
        assert size == 1
    
    def test_get_available_count(self):
        """测试获取可用连接数"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10)
        conn = pool.get_connection()
        
        available = pool.get_available_count()
        assert available == 0
        
        pool.release(conn)
        available = pool.get_available_count()
        assert available == 1
    
    def test_get_stats(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10, leak_detection=True)
        conn = pool.get_connection()
        
        stats = pool.get_stats()
        
        assert stats["current_size"] == 1
        assert stats["available"] == 0
        assert stats["active"] == 1
        assert stats["max_size"] == 10
        assert stats["created"] == 1
        assert stats["leak_count"] == 1
    
    def test_update_config(self):
        """测试更新配置"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10)
        pool.update_config(max_size=20, idle_timeout=600.0, max_usage=500, leak_detection=True)
        
        assert pool._max_size == 20
        assert pool._idle_timeout == 600.0
        assert pool._max_usage == 500
        assert pool._leak_detection is True
    
    def test_update_config_invalid_max_size(self):
        """测试更新配置时max_size小于当前活跃连接数"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(max_size=10)
        conn = pool.get_connection()
        
        with pytest.raises(ValueError):
            pool.update_config(max_size=0)
    
    def test_update_config_invalid_idle_timeout(self):
        """测试更新配置时idle_timeout为负数"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool()
        
        with pytest.raises(ValueError):
            pool.update_config(idle_timeout=-1)
    
    def test_update_config_invalid_max_usage(self):
        """测试更新配置时max_usage小于等于0"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool()
        
        with pytest.raises(ValueError):
            pool.update_config(max_usage=0)
        
        with pytest.raises(ValueError):
            pool.update_config(max_usage=-1)
    
    def test_is_connection_valid(self):
        """测试验证连接有效性"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(idle_timeout=1.0)
        conn = pool.get_connection()
        
        assert pool._is_connection_valid(conn) is True
        
        conn.close()
        assert pool._is_connection_valid(conn) is False
    
    def test_is_connection_valid_idle_timeout(self):
        """测试验证连接有效性（空闲超时）"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(idle_timeout=0.1)
        conn = pool.get_connection()
        
        time.sleep(0.2)
        
        assert pool._is_connection_valid(conn) is False
    
    def test_is_connection_valid_no_timeout(self):
        """测试验证连接有效性（无超时）"""
        from src.infrastructure.utils.components.connection_pool import ConnectionPool
        
        pool = ConnectionPool(idle_timeout=0)
        conn = pool.get_connection()
        
        assert pool._is_connection_valid(conn) is True

