"""
Core Services Integration 组件测试覆盖率补充

补充service_registry、service_executor、integration_monitor、cache_manager、connection_pool的测试覆盖
"""

import time
import threading
from unittest.mock import Mock, MagicMock, patch
import pytest

from src.core.core_services.integration.service_registry import ServiceRegistry
from src.core.core_services.integration.service_executor import ServiceExecutor
from src.core.core_services.integration.integration_monitor import PerformanceMonitor
from src.core.core_services.integration.cache_manager import CacheManager
from src.core.core_services.integration.connection_pool import ConnectionPool, ConnectionPoolManager
from src.core.core_services.integration.integration_manager_core import ServiceIntegrationManagerRefactored
from src.core.core_services.integration.integration_models import ServiceEndpoint, ServiceCall


class TestServiceRegistry:
    """测试ServiceRegistry组件"""

    def test_register_service(self):
        """测试注册服务"""
        registry = ServiceRegistry()
        endpoint = ServiceEndpoint(
            service_name="test_service",
            endpoint_url="http://test.service/api",
            connection_pool_size=10
        )
        
        registry.register_service("test_service", endpoint)
        
        assert registry.get_service_count() == 1
        assert registry.get_service("test_service") == endpoint

    def test_unregister_service(self):
        """测试注销服务"""
        registry = ServiceRegistry()
        endpoint = ServiceEndpoint(
            service_name="test_service",
            endpoint_url="http://test.service/api",
            connection_pool_size=10
        )
        
        registry.register_service("test_service", endpoint)
        result = registry.unregister_service("test_service")
        
        assert result is True
        assert registry.get_service_count() == 0
        assert registry.get_service("test_service") is None

    def test_unregister_nonexistent_service(self):
        """测试注销不存在的服务"""
        registry = ServiceRegistry()
        result = registry.unregister_service("nonexistent")
        
        assert result is False

    def test_get_service(self):
        """测试获取服务"""
        registry = ServiceRegistry()
        endpoint = ServiceEndpoint(
            service_name="test_service",
            endpoint_url="http://test.service/api",
            connection_pool_size=10
        )
        
        registry.register_service("test_service", endpoint)
        retrieved = registry.get_service("test_service")
        
        assert retrieved == endpoint

    def test_get_nonexistent_service(self):
        """测试获取不存在的服务"""
        registry = ServiceRegistry()
        result = registry.get_service("nonexistent")
        
        assert result is None

    def test_list_services(self):
        """测试列出所有服务"""
        registry = ServiceRegistry()
        endpoint1 = ServiceEndpoint(service_name="service1", endpoint_url="http://service1/api", connection_pool_size=10)
        endpoint2 = ServiceEndpoint(service_name="service2", endpoint_url="http://service2/api", connection_pool_size=20)
        
        registry.register_service("service1", endpoint1)
        registry.register_service("service2", endpoint2)
        
        services = registry.list_services()
        
        assert len(services) == 2
        assert "service1" in services
        assert "service2" in services

    def test_get_service_count(self):
        """测试获取服务数量"""
        registry = ServiceRegistry()
        assert registry.get_service_count() == 0
        
        registry.register_service("service1", ServiceEndpoint(service_name="service1", endpoint_url="http://service1/api", connection_pool_size=10))
        assert registry.get_service_count() == 1
        
        registry.register_service("service2", ServiceEndpoint(service_name="service2", endpoint_url="http://service2/api", connection_pool_size=20))
        assert registry.get_service_count() == 2


class TestPerformanceMonitor:
    """测试PerformanceMonitor组件"""

    def test_record_successful_call(self):
        """测试记录成功调用"""
        monitor = PerformanceMonitor()
        
        monitor.record_call(0.1, True)
        stats = monitor.get_stats()
        
        assert stats['total_calls'] == 1
        assert stats['successful_calls'] == 1
        assert stats['failed_calls'] == 0
        assert stats['avg_response_time'] == 0.1

    def test_record_failed_call(self):
        """测试记录失败调用"""
        monitor = PerformanceMonitor()
        
        monitor.record_call(0.2, False)
        stats = monitor.get_stats()
        
        assert stats['total_calls'] == 1
        assert stats['successful_calls'] == 0
        assert stats['failed_calls'] == 1

    def test_calculate_avg_response_time(self):
        """测试计算平均响应时间"""
        monitor = PerformanceMonitor()
        
        monitor.record_call(0.1, True)
        monitor.record_call(0.2, True)
        monitor.record_call(0.3, True)
        
        stats = monitor.get_stats()
        assert stats['avg_response_time'] == pytest.approx(0.2, rel=1e-9)

    def test_track_min_max_response_time(self):
        """测试跟踪最小最大响应时间"""
        monitor = PerformanceMonitor()
        
        monitor.record_call(0.3, True)
        monitor.record_call(0.1, True)
        monitor.record_call(0.2, True)
        
        stats = monitor.get_stats()
        assert stats['min_response_time'] == 0.1
        assert stats['max_response_time'] == 0.3

    def test_thread_safety(self):
        """测试线程安全"""
        monitor = PerformanceMonitor()
        
        def record_calls():
            for _ in range(10):
                monitor.record_call(0.1, True)
        
        threads = [threading.Thread(target=record_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = monitor.get_stats()
        assert stats['total_calls'] == 50


class TestCacheManager:
    """测试CacheManager组件"""

    def test_set_and_get_cache(self):
        """测试设置和获取缓存"""
        cache = CacheManager(max_size=100, ttl=60)
        
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        
        assert result == {"data": "value1"}

    def test_cache_expiration(self):
        """测试缓存过期"""
        cache = CacheManager(max_size=100, ttl=1)  # 1秒TTL
        
        cache.set("key1", {"data": "value1"})
        assert cache.get("key1") == {"data": "value1"}
        
        time.sleep(1.1)  # 等待过期
        result = cache.get("key1")
        
        assert result is None

    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        cache = CacheManager(max_size=2, ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # 应该删除最旧的
        
        assert cache.get("key1") is None  # 最旧的被删除
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_clear_cache(self):
        """测试清空缓存"""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get_stats()['cache_size'] == 0

    def test_get_stats(self):
        """测试获取统计信息"""
        cache = CacheManager(max_size=100, ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['cache_size'] == 2
        assert stats['max_cache_size'] == 100
        assert stats['cache_ttl'] == 60


class TestConnectionPool:
    """测试ConnectionPool组件"""

    def test_create_connection_pool(self):
        """测试创建连接池"""
        pool = ConnectionPool(pool_size=5, service_name="test_service")
        
        assert pool.pool_size == 5
        assert pool.service_name == "test_service"
        assert pool.get_stats()['pool_size'] == 5

    def test_get_and_return_connection(self):
        """测试获取和归还连接"""
        pool = ConnectionPool(pool_size=2, service_name="test")
        
        conn1 = pool.get_connection()
        assert conn1 is not None
        
        conn2 = pool.get_connection()
        assert conn2 is not None
        
        pool.return_connection(conn1)
        pool.return_connection(conn2)
        
        stats = pool.get_stats()
        assert stats['idle_connections'] == 2

    def test_connection_timeout(self):
        """测试连接超时"""
        pool = ConnectionPool(pool_size=1, service_name="test")
        
        # 获取所有连接
        conn1 = pool.get_connection()
        
        # 尝试获取第二个连接（应该超时）
        conn2 = pool.get_connection(timeout=0.1)
        assert conn2 is None

    def test_get_stats(self):
        """测试获取统计信息"""
        pool = ConnectionPool(pool_size=3, service_name="test")
        
        stats = pool.get_stats()
        assert stats['pool_size'] == 3
        assert stats['idle_connections'] == 3
        assert stats['active_connections'] == 0


class TestConnectionPoolManager:
    """测试ConnectionPoolManager组件"""

    def test_get_connection_pool(self):
        """测试获取连接池"""
        manager = ConnectionPoolManager()
        
        pool1 = manager.get_connection_pool("service1", pool_size=10)
        pool2 = manager.get_connection_pool("service1", pool_size=10)
        
        assert pool1 == pool2  # 应该返回同一个池
        assert pool1.pool_size == 10

    def test_get_pool_stats(self):
        """测试获取所有连接池统计"""
        manager = ConnectionPoolManager()
        
        manager.get_connection_pool("service1", pool_size=5)
        manager.get_connection_pool("service2", pool_size=10)
        
        stats = manager.get_pool_stats()
        
        assert len(stats) == 2
        assert "service1" in stats
        assert "service2" in stats

    def test_close_all_pools(self):
        """测试关闭所有连接池"""
        manager = ConnectionPoolManager()
        
        manager.get_connection_pool("service1", pool_size=5)
        manager.get_connection_pool("service2", pool_size=10)
        
        manager.close_all_pools()
        
        stats = manager.get_pool_stats()
        assert len(stats) == 0


class TestServiceExecutor:
    """测试ServiceExecutor组件"""

    def test_call_service_success(self):
        """测试成功调用服务"""
        registry = ServiceRegistry()
        pool_manager = ConnectionPoolManager()
        cache_manager = CacheManager()
        monitor = PerformanceMonitor()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        registry.register_service("test_service", endpoint)
        
        executor = ServiceExecutor(registry, pool_manager, cache_manager, monitor)
        
        call = ServiceCall(
            service_name="test_service",
            method_name="test_method",
            parameters={"param": "value"}
        )
        
        result = executor.call_service(call)
        
        assert result['success'] is True
        assert result['service'] == "test_service"
        assert monitor.get_stats()['successful_calls'] == 1

    def test_call_service_not_registered(self):
        """测试调用未注册的服务"""
        registry = ServiceRegistry()
        pool_manager = ConnectionPoolManager()
        
        executor = ServiceExecutor(registry, pool_manager)
        
        call = ServiceCall(
            service_name="nonexistent_service",
            method_name="test_method",
            parameters={}
        )
        
        result = executor.call_service(call)
        
        assert result['success'] is False
        assert 'error' in result

    def test_call_service_with_cache(self):
        """测试带缓存的服务调用"""
        registry = ServiceRegistry()
        pool_manager = ConnectionPoolManager()
        cache_manager = CacheManager()
        monitor = PerformanceMonitor()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        registry.register_service("test_service", endpoint)
        
        executor = ServiceExecutor(registry, pool_manager, cache_manager, monitor)
        
        call = ServiceCall(
            service_name="test_service",
            method_name="test_method",
            parameters={"param": "value"}
        )
        
        # 第一次调用
        result1 = executor.call_service(call)
        
        # 第二次调用（应该从缓存获取）
        result2 = executor.call_service(call)
        
        assert result1 == result2
        # 缓存命中，不应该增加调用次数
        assert monitor.get_stats()['total_calls'] == 1

    def test_call_service_error_handling(self):
        """测试服务调用错误处理"""
        registry = ServiceRegistry()
        pool_manager = ConnectionPoolManager()
        monitor = PerformanceMonitor()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        registry.register_service("test_service", endpoint)
        
        executor = ServiceExecutor(registry, pool_manager, None, monitor)
        
        # Mock _execute_service_call 抛出异常
        with patch.object(executor, '_execute_service_call', side_effect=Exception("Service error")):
            call = ServiceCall(
                service_name="test_service",
                method_name="test_method",
                parameters={}
            )
            
            result = executor.call_service(call)
            
            assert result['success'] is False
            assert 'error' in result
            assert monitor.get_stats()['failed_calls'] == 1


class TestServiceIntegrationManagerRefactored:
    """测试ServiceIntegrationManagerRefactored组件"""

    def test_register_service(self):
        """测试注册服务"""
        manager = ServiceIntegrationManagerRefactored()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        assert manager.registry.get_service_count() == 1

    def test_unregister_service(self):
        """测试注销服务"""
        manager = ServiceIntegrationManagerRefactored()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        result = manager.unregister_service("test_service")
        
        assert result is True
        assert manager.registry.get_service_count() == 0

    def test_call_service(self):
        """测试调用服务"""
        manager = ServiceIntegrationManagerRefactored()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        call = ServiceCall(
            service_name="test_service",
            method_name="test_method",
            parameters={}
        )
        
        result = manager.call_service(call)
        
        assert result['success'] is True

    def test_get_performance_stats(self):
        """测试获取性能统计"""
        manager = ServiceIntegrationManagerRefactored()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        call = ServiceCall(
            service_name="test_service",
            method_name="test_method",
            parameters={}
        )
        manager.call_service(call)
        
        stats = manager.get_performance_stats()
        
        assert 'call_stats' in stats
        assert 'service_count' in stats
        assert 'connection_pools' in stats
        assert 'cache_size' in stats
        assert stats['service_count'] == 1

    def test_optimize_for_high_load(self):
        """测试高负载优化"""
        manager = ServiceIntegrationManagerRefactored()
        
        manager.optimize_for_high_load()
        
        assert manager.max_workers == 50
        assert manager.cache_manager._max_cache_size == 50000
        assert manager.cache_manager._cache_ttl == 180

    def test_shutdown(self):
        """测试关闭管理器"""
        manager = ServiceIntegrationManagerRefactored()
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        manager.shutdown()
        
        # 验证连接池已关闭
        assert len(manager.pool_manager._connection_pools) == 0
        # 验证缓存已清空
        assert manager.cache_manager.get_stats()['cache_size'] == 0

    def test_manager_without_caching(self):
        """测试禁用缓存的管理器"""
        manager = ServiceIntegrationManagerRefactored(enable_caching=False)
        
        assert manager.cache_manager is None
        assert manager.enable_caching is False
        
        endpoint = ServiceEndpoint(service_name="test_service", endpoint_url="http://test.service/api", connection_pool_size=10)
        manager.register_service("test_service", endpoint)
        
        call = ServiceCall(
            service_name="test_service",
            method_name="test_method",
            parameters={}
        )
        
        result = manager.call_service(call)
        assert result['success'] is True

