#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层高级连接池组件测试

测试目标：提升utils/components/advanced_connection_pool.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.advanced_connection_pool模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestConnectionPoolMetrics:
    """测试连接池性能指标"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        assert metrics.created_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.destroyed_connections == 0
        assert metrics.connection_requests == 0
        assert metrics.connection_hits == 0
        assert metrics.connection_misses == 0
        assert metrics.connection_timeouts == 0
        assert metrics.average_wait_time == 0.0
        assert metrics.peak_active_connections == 0
    
    def test_record_connection_created(self):
        """测试记录连接创建"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.record_connection_created()
        
        assert metrics.created_connections == 1
    
    def test_record_connection_destroyed(self):
        """测试记录连接销毁"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.record_connection_destroyed()
        
        assert metrics.destroyed_connections == 1
    
    def test_record_connection_request(self):
        """测试记录连接请求"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.record_connection_request()
        
        assert metrics.connection_requests == 1
    
    def test_update_active_connections(self):
        """测试更新活跃连接数"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.update_active_connections(5)
        
        assert metrics.active_connections == 5
        assert metrics.peak_active_connections == 5
    
    def test_update_idle_connections(self):
        """测试更新空闲连接数"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.update_idle_connections(3)
        
        assert metrics.idle_connections == 3
    
    def test_reset(self):
        """测试重置指标"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.record_connection_created()
        metrics.update_active_connections(5)
        
        metrics.reset()
        
        assert metrics.created_connections == 0
        assert metrics.active_connections == 0
        assert metrics.peak_active_connections == 0
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        metrics.record_connection_created()
        metrics.record_connection_request()
        metrics.connection_hits = 1
        
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["created_connections"] == 1
        assert result["connection_requests"] == 1
        assert result["connection_hits"] == 1
        assert "hit_rate" in result
    
    def test_get_stats(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionPoolMetrics
        
        metrics = ConnectionPoolMetrics()
        stats = metrics.get_stats()
        
        assert isinstance(stats, dict)
        assert "created_connections" in stats


class TestConnectionWrapper:
    """测试连接包装器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool)
        
        assert wrapper.connection == mock_conn
        assert wrapper._pool == mock_pool
        assert wrapper._returned is False
        assert wrapper._closed is False
        assert wrapper.is_closed is False
    
    def test_is_expired(self):
        """测试检查连接是否过期"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool, max_age=1.0)
        time.sleep(1.1)
        
        assert wrapper.is_expired() is True
    
    def test_is_idle_timeout(self):
        """测试检查连接是否空闲超时"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool, max_idle_time=1.0)
        time.sleep(1.1)
        
        assert wrapper.is_idle_timeout() is True
    
    def test_get_age(self):
        """测试获取连接年龄"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool)
        age = wrapper.get_age()
        
        assert age >= 0
    
    def test_get_idle_time(self):
        """测试获取空闲时间"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool)
        idle_time = wrapper.get_idle_time()
        
        assert idle_time >= 0
    
    def test_update_last_used(self):
        """测试更新最后使用时间"""
        from src.infrastructure.utils.components.advanced_connection_pool import ConnectionWrapper
        
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        
        wrapper = ConnectionWrapper(mock_conn, mock_pool)
        old_time = wrapper.last_used_time
        
        time.sleep(0.1)
        wrapper.update_last_used()
        
        assert wrapper.last_used_time > old_time

