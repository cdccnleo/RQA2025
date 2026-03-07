#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层优化连接池组件测试

测试目标：提升utils/components/optimized_connection_pool.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.optimized_connection_pool模块
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock


class TestConnectionPoolConstants:
    """测试连接池常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.optimized_connection_pool import ConnectionPoolConstants
        
        assert ConnectionPoolConstants.DEFAULT_MIN_SIZE == 5
        assert ConnectionPoolConstants.DEFAULT_MAX_SIZE == 20
        assert ConnectionPoolConstants.DEFAULT_INITIAL_SIZE == 10
        assert ConnectionPoolConstants.DEFAULT_CONNECTION_TIMEOUT == 30.0
        assert ConnectionPoolConstants.DEFAULT_IDLE_TIMEOUT == 300.0
        assert ConnectionPoolConstants.DEFAULT_MAX_LIFETIME == 3600.0
        assert ConnectionPoolConstants.DEFAULT_LEAK_DETECTION_THRESHOLD == 60.0
        assert ConnectionPoolConstants.DEFAULT_HEALTH_CHECK_INTERVAL == 30.0
        assert ConnectionPoolConstants.DEFAULT_MAX_RETRIES == 3
        assert ConnectionPoolConstants.DEFAULT_RETRY_DELAY == 1.0
        assert ConnectionPoolConstants.CLEANUP_BATCH_SIZE == 10
        assert ConnectionPoolConstants.MAX_CLEANUP_ITERATIONS == 50


class TestPoolState:
    """测试连接池状态枚举"""
    
    def test_pool_state_enum(self):
        """测试连接池状态枚举值"""
        from src.infrastructure.utils.components.optimized_connection_pool import PoolState
        
        assert PoolState.HEALTHY.value == "healthy"
        assert PoolState.WARNING.value == "warning"
        assert PoolState.CRITICAL.value == "critical"
        assert PoolState.FAILED.value == "failed"


class TestConnectionInfo:
    """测试连接信息数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.optimized_connection_pool import ConnectionInfo
        
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


class TestOptimizedConnectionPool:
    """测试优化连接池管理器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.optimized_connection_pool import OptimizedConnectionPool
        
        pool = OptimizedConnectionPool()
        assert pool is not None
        assert hasattr(pool, '_min_size')
        assert hasattr(pool, '_max_size')
        assert hasattr(pool, '_initial_size')
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.components.optimized_connection_pool import OptimizedConnectionPool
        
        pool = OptimizedConnectionPool(
            min_size=2,
            max_size=10,
            initial_size=5,
            connection_timeout=60.0,
            idle_timeout=600.0
        )
        
        assert pool._min_size == 2
        assert pool._max_size == 10
        assert pool._initial_size == 5
        assert pool._connection_timeout == 60.0
        assert pool._idle_timeout == 600.0
    
    def test_shutdown(self):
        """测试关闭连接池"""
        from src.infrastructure.utils.components.optimized_connection_pool import OptimizedConnectionPool
        
        pool = OptimizedConnectionPool()
        result = pool.shutdown()
        
        # shutdown返回bool值
        assert isinstance(result, bool)

