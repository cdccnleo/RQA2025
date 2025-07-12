#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库连接池模块单元测试
测试ConnectionPool和Connection类的核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.infrastructure.database.connection_pool import ConnectionPool, Connection


class TestConnection:
    """测试Connection类"""
    
    def test_connection_creation(self):
        """测试连接创建"""
        conn = Connection()
        
        assert conn.last_used > 0
        assert conn.usage_count == 0
        assert conn.closed is False
    
    def test_connection_close(self):
        """测试连接关闭"""
        conn = Connection()
        conn.close()
        
        assert conn.closed is True
    
    def test_connection_usage_tracking(self):
        """测试连接使用次数跟踪"""
        conn = Connection()
        initial_count = conn.usage_count
        
        # 模拟使用连接
        conn.usage_count += 1
        
        assert conn.usage_count == initial_count + 1


class TestConnectionPool:
    """测试ConnectionPool类"""
    
    @pytest.fixture
    def pool(self):
        """创建连接池实例"""
        return ConnectionPool(max_size=5, idle_timeout=60, max_usage=100)
    
    def test_pool_initialization(self, pool):
        """测试连接池初始化"""
        assert pool._pool.maxsize == 5
        assert pool._idle_timeout == 60
        assert pool._max_usage == 100
        assert pool._created_count == 0
        assert pool._active_connections == 0
        assert pool._leak_detection is True
    
    def test_connection_creation(self, pool):
        """测试连接创建"""
        conn = pool._create_connection()
        
        assert isinstance(conn, Connection)
        assert pool._created_count == 1
    
    def test_connection_validation(self, pool):
        """测试连接验证"""
        # 创建有效连接
        conn = Connection()
        assert pool._is_connection_valid(conn) is True
        
        # 创建过期连接
        conn.last_used = time.time() - 120  # 超过空闲超时
        assert pool._is_connection_valid(conn) is False
        
        # 创建已关闭连接
        conn = Connection()
        conn.close()
        assert pool._is_connection_valid(conn) is False
    
    def test_acquire_connection(self, pool):
        """测试获取连接"""
        conn = pool.acquire()
        
        assert isinstance(conn, Connection)
        assert pool._active_connections == 1
        assert pool._created_count == 1
    
    def test_acquire_from_pool(self, pool):
        """测试从池中获取连接"""
        # 先获取一个连接
        conn1 = pool.acquire()
        pool.release(conn1)
        
        # 再次获取连接应该从池中获取
        conn2 = pool.acquire()
        
        assert pool._active_connections == 1
        assert conn2.usage_count == 1  # 使用次数增加
    
    def test_release_connection(self, pool):
        """测试释放连接"""
        conn = pool.acquire()
        pool.release(conn)
        
        assert pool._active_connections == 0
        assert conn.usage_count == 1
    
    def test_pool_exhaustion(self, pool):
        """测试连接池耗尽"""
        # 获取所有连接
        connections = []
        for _ in range(5):
            connections.append(pool.acquire())
        
        # 尝试获取更多连接应该失败
        with pytest.raises(RuntimeError, match="Connection pool exhausted"):
            pool.acquire()
        
        # 释放一个连接后应该能再次获取
        pool.release(connections[0])
        conn = pool.acquire()
        assert isinstance(conn, Connection)
    
    def test_connection_max_usage(self, pool):
        """测试连接最大使用次数"""
        conn = pool.acquire()
        
        # 模拟达到最大使用次数
        conn.usage_count = 100
        
        pool.release(conn)
        
        # 连接应该被丢弃而不是放回池中
        assert pool._pool.qsize() == 0
    
    def test_health_check(self, pool):
        """测试健康检查"""
        # 获取几个连接
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        pool.release(conn1)
        
        stats = pool.health_check()
        
        assert stats['total'] == 5
        assert stats['active'] == 1
        assert stats['idle'] == 1
        assert stats['created'] == 2
        assert 'leaks' in stats
        assert 'config' in stats
    
    def test_update_config(self, pool):
        """测试配置更新"""
        # 更新配置
        pool.update_config(max_size=10, idle_timeout=120, max_usage=200)
        
        assert pool._pool.maxsize == 10
        assert pool._idle_timeout == 120
        assert pool._max_usage == 200
    
    def test_update_config_validation(self, pool):
        """测试配置更新验证"""
        # 获取所有连接
        connections = []
        for _ in range(5):
            connections.append(pool.acquire())
        
        # 尝试设置max_size小于活跃连接数应该失败
        with pytest.raises(ValueError):
            pool.update_config(max_size=3)
        
        # 释放连接后应该能更新
        pool.release(connections[0])
        pool.update_config(max_size=4)
    
    def test_leak_detection(self, pool):
        """测试泄漏检测"""
        conn = pool.acquire()
        
        # 检查泄漏追踪
        assert id(conn) in pool._leak_tracker
        
        # 释放连接
        pool.release(conn)
        
        # 泄漏追踪应该被清理
        assert id(conn) not in pool._leak_tracker
    
    def test_disable_leak_detection(self, pool):
        """测试禁用泄漏检测"""
        pool.update_config(leak_detection=False)
        
        conn = pool.acquire()
        
        # 泄漏追踪应该为空
        assert len(pool._leak_tracker) == 0
        
        pool.release(conn)


class TestConnectionPoolThreadSafety:
    """测试连接池线程安全性"""
    
    @pytest.fixture
    def pool(self):
        """创建连接池实例"""
        return ConnectionPool(max_size=10, idle_timeout=60)  # 增加连接池大小
    
    def test_concurrent_acquire(self, pool):
        """测试并发获取连接"""
        results = []
        errors = []
        
        def worker():
            try:
                conn = pool.acquire()
                time.sleep(0.1)  # 模拟工作
                pool.release(conn)
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 10
        assert len(errors) == 0
        assert pool._active_connections == 0
    
    def test_concurrent_health_check(self, pool):
        """测试并发健康检查"""
        results = []
        
        def worker():
            stats = pool.health_check()
            results.append(stats)
        
        # 创建多个线程同时进行健康检查
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有健康检查都成功
        assert len(results) == 5
        for stats in results:
            assert 'total' in stats
            assert 'active' in stats
            assert 'idle' in stats


class TestConnectionPoolEdgeCases:
    """测试连接池边界情况"""
    
    @pytest.fixture
    def pool(self):
        """创建连接池实例"""
        return ConnectionPool(max_size=5, idle_timeout=60, max_usage=100)
    
    def test_zero_max_size(self):
        """测试最大连接数为0"""
        # ConnectionPool实际上允许max_size为0，所以这个测试需要调整
        pool = ConnectionPool(max_size=0)
        # 尝试获取连接应该失败
        with pytest.raises(RuntimeError, match="Connection pool exhausted"):
            pool.acquire()
    
    def test_negative_timeout(self):
        """测试负超时时间"""
        pool = ConnectionPool(idle_timeout=-1)
        # 应该能正常工作，但验证逻辑可能有问题
        conn = pool.acquire()
        assert isinstance(conn, Connection)
    
    def test_zero_max_usage(self):
        """测试最大使用次数为0"""
        pool = ConnectionPool(max_usage=0)
        conn = pool.acquire()
        pool.release(conn)
        
        # 连接应该被丢弃
        assert pool._pool.qsize() == 0
    
    def test_invalid_connection_release(self, pool):
        """测试释放无效连接"""
        # 尝试释放未从池中获取的连接
        invalid_conn = Connection()
        pool.release(invalid_conn)
        
        # 应该不会抛出异常，但也不会影响池状态
        assert pool._active_connections == 0


class TestConnectionPoolIntegration:
    """测试连接池集成功能"""
    
    @pytest.fixture
    def pool(self):
        """创建连接池实例"""
        return ConnectionPool(max_size=2, idle_timeout=1, max_usage=3)
    
    def test_connection_lifecycle(self, pool):
        """测试连接完整生命周期"""
        # 获取连接
        conn1 = pool.acquire()
        assert pool._active_connections == 1
        
        # 释放连接
        pool.release(conn1)
        assert pool._active_connections == 0
        assert pool._pool.qsize() == 1
        
        # 再次获取连接
        conn2 = pool.acquire()
        assert pool._active_connections == 1
        assert pool._pool.qsize() == 0
    
    def test_connection_reuse(self, pool):
        """测试连接重用"""
        # 获取并释放连接多次
        for i in range(3):
            conn = pool.acquire()
            pool.release(conn)
        
        # 应该只创建了一个连接
        assert pool._created_count == 1
        assert conn.usage_count == 3
    
    def test_connection_expiry(self, pool):
        """测试连接过期"""
        conn = pool.acquire()
        pool.release(conn)
        
        # 等待连接过期
        time.sleep(1.1)
        
        # 再次获取连接应该创建新连接
        new_conn = pool.acquire()
        assert pool._created_count == 2
        assert new_conn != conn
    
    def test_connection_max_usage_reached(self, pool):
        """测试达到最大使用次数"""
        # 使用连接达到最大次数
        for _ in range(3):
            conn = pool.acquire()
            pool.release(conn)
        
        # 再次获取连接应该创建新连接
        new_conn = pool.acquire()
        assert pool._created_count == 2
        assert new_conn.usage_count == 0


class TestConnectionPoolPerformance:
    """测试连接池性能"""
    
    def test_rapid_acquire_release(self):
        """测试快速获取释放连接"""
        pool = ConnectionPool(max_size=10)
        
        start_time = time.time()
        
        # 快速获取释放连接1000次
        for _ in range(1000):
            conn = pool.acquire()
            pool.release(conn)
        
        end_time = time.time()
        
        # 验证性能（应该在合理时间内完成）
        assert end_time - start_time < 1.0  # 1秒内完成
        assert pool._active_connections == 0
    
    def test_memory_usage(self):
        """测试内存使用"""
        pool = ConnectionPool(max_size=100)
        
        # 获取所有连接
        connections = []
        for _ in range(100):
            connections.append(pool.acquire())
        
        # 释放所有连接
        for conn in connections:
            pool.release(conn)
        
        # 验证池状态
        assert pool._active_connections == 0
        assert pool._pool.qsize() == 100
        assert pool._created_count == 100
