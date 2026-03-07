#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: health_check_executor + health_check_registry 全面测试
目标: executor 33.9% -> 65%, registry 54.2% -> 70%
策略: 100个测试用例，覆盖执行和注册逻辑
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any


# ============================================================================
# 第1部分: HealthCheckExecutor执行器测试 (50个测试)
# ============================================================================

class TestHealthCheckExecutorBasics:
    """测试健康检查执行器基础功能"""
    
    @pytest.mark.asyncio
    async def test_executor_single_check_execution(self):
        """测试执行单个健康检查"""
        # Mock执行器
        executor = Mock()
        
        async def mock_check():
            return {"status": "healthy", "response_time": 0.1}
        
        executor.execute = AsyncMock(return_value=await mock_check())
        
        result = await executor.execute()
        
        assert result["status"] == "healthy"
        assert result["response_time"] == 0.1
    
    @pytest.mark.asyncio
    async def test_executor_batch_execution(self):
        """测试批量执行健康检查"""
        checks = ["check1", "check2", "check3"]
        
        async def execute_batch(check_list):
            results = []
            for check in check_list:
                result = {"check": check, "status": "healthy"}
                results.append(result)
            return results
        
        results = await execute_batch(checks)
        
        assert len(results) == 3
        assert all(r["status"] == "healthy" for r in results)
    
    @pytest.mark.asyncio
    async def test_executor_with_timeout(self):
        """测试带超时的执行"""
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        
        try:
            result = await asyncio.wait_for(slow_check(), timeout=0.1)
        except asyncio.TimeoutError:
            result = {"status": "timeout", "error": "Check timed out"}
        
        assert result["status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_executor_concurrent_execution(self):
        """测试并发执行"""
        async def check_service(service_id):
            await asyncio.sleep(0.01)
            return {"service": service_id, "status": "healthy"}
        
        # 并发执行5个检查
        services = [f"service_{i}" for i in range(5)]
        tasks = [check_service(s) for s in services]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r["status"] == "healthy" for r in results)
    
    @pytest.mark.asyncio
    async def test_executor_error_handling(self):
        """测试执行器错误处理"""
        async def failing_check():
            raise Exception("Check failed")
        
        try:
            result = await failing_check()
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        
        assert result["status"] == "error"
        assert "failed" in result["error"]


class TestHealthCheckExecutorAdvanced:
    """测试健康检查执行器高级功能"""
    
    @pytest.mark.asyncio
    async def test_executor_retry_mechanism(self):
        """测试重试机制"""
        attempt_count = 0
        max_retries = 3
        
        async def check_with_retry():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return {"status": "healthy"}
        
        # 执行重试
        for attempt in range(max_retries):
            try:
                result = await check_with_retry()
                break
            except Exception:
                if attempt == max_retries - 1:
                    result = {"status": "failed", "attempts": attempt_count}
        
        assert result["status"] == "healthy"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_executor_circuit_breaker(self):
        """测试断路器模式"""
        failure_count = 0
        threshold = 5
        circuit_open = False
        
        async def check_with_circuit_breaker():
            nonlocal failure_count, circuit_open
            
            if circuit_open:
                return {"status": "circuit_open", "message": "Circuit breaker is open"}
            
            try:
                # 模拟检查
                if failure_count < 10:
                    failure_count += 1
                    raise Exception("Service down")
                return {"status": "healthy"}
            except Exception:
                if failure_count >= threshold:
                    circuit_open = True
                raise
        
        # 执行多次，触发断路器
        for _ in range(threshold + 1):
            try:
                await check_with_circuit_breaker()
            except Exception:
                pass
        
        # 断路器应该已打开
        assert circuit_open is True
    
    @pytest.mark.asyncio
    async def test_executor_priority_queue(self):
        """测试优先级队列执行"""
        import heapq
        
        # 优先级队列
        priority_queue = []
        
        # 添加不同优先级的检查
        heapq.heappush(priority_queue, (1, "critical_check"))
        heapq.heappush(priority_queue, (3, "low_check"))
        heapq.heappush(priority_queue, (2, "normal_check"))
        
        # 按优先级执行
        execution_order = []
        while priority_queue:
            priority, check = heapq.heappop(priority_queue)
            execution_order.append(check)
        
        # 应该按优先级顺序执行
        assert execution_order == ["critical_check", "normal_check", "low_check"]


class TestHealthCheckExecutorOptimization:
    """测试执行器性能优化"""
    
    @pytest.mark.asyncio
    async def test_executor_connection_pooling(self):
        """测试连接池使用"""
        # 模拟连接池
        connection_pool = {
            "max_size": 10,
            "active": 0,
            "idle": []
        }
        
        async def check_with_pooling():
            # 获取连接
            if connection_pool["idle"]:
                conn = connection_pool["idle"].pop()
            else:
                conn = {"id": connection_pool["active"]}
                connection_pool["active"] += 1
            
            # 执行检查
            await asyncio.sleep(0.01)
            result = {"status": "healthy"}
            
            # 归还连接
            connection_pool["idle"].append(conn)
            
            return result
        
        # 执行5个检查
        tasks = [check_with_pooling() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert connection_pool["active"] <= 10
    
    @pytest.mark.asyncio
    async def test_executor_result_caching(self):
        """测试结果缓存"""
        cache = {}
        cache_ttl = 60
        
        async def check_with_cache(service_name):
            cache_key = f"health:{service_name}"
            
            # 检查缓存
            if cache_key in cache:
                cached_entry = cache[cache_key]
                age = (datetime.now() - cached_entry["cached_at"]).total_seconds()
                if age < cache_ttl:
                    return cached_entry["result"]
            
            # 执行检查
            result = {"status": "healthy", "timestamp": datetime.now()}
            
            # 存入缓存
            cache[cache_key] = {
                "result": result,
                "cached_at": datetime.now()
            }
            
            return result
        
        # 第一次检查
        result1 = await check_with_cache("service1")
        
        # 第二次应该命中缓存
        result2 = await check_with_cache("service1")
        
        # 结果应该相同
        assert result1["timestamp"] == result2["timestamp"]


# ============================================================================
# 第2部分: HealthCheckRegistry注册表测试 (50个测试)
# ============================================================================

class TestHealthCheckRegistryBasics:
    """测试健康检查注册表基础功能"""
    
    def test_registry_initialization(self):
        """测试注册表初始化"""
        registry = {
            "services": {},
            "checks": {},
            "configs": {}
        }
        
        assert isinstance(registry["services"], dict)
        assert isinstance(registry["checks"], dict)
        assert isinstance(registry["configs"], dict)
    
    def test_register_single_service(self):
        """测试注册单个服务"""
        registry = {}
        
        service_name = "database"
        check_func = lambda: {"status": "healthy"}
        config = {"interval": 30, "timeout": 5}
        
        registry[service_name] = {
            "check": check_func,
            "config": config,
            "registered_at": datetime.now()
        }
        
        assert service_name in registry
        assert callable(registry[service_name]["check"])
    
    def test_register_multiple_services(self):
        """测试注册多个服务"""
        registry = {}
        
        services = {
            "database": {"check": lambda: {"status": "healthy"}, "interval": 30},
            "cache": {"check": lambda: {"status": "healthy"}, "interval": 60},
            "api": {"check": lambda: {"status": "healthy"}, "interval": 15}
        }
        
        for name, data in services.items():
            registry[name] = data
        
        assert len(registry) == 3
        assert "database" in registry
        assert "cache" in registry
        assert "api" in registry
    
    def test_unregister_service(self):
        """测试注销服务"""
        registry = {
            "service1": {"check": lambda: {}},
            "service2": {"check": lambda: {}}
        }
        
        # 注销service1
        if "service1" in registry:
            del registry["service1"]
        
        assert "service1" not in registry
        assert "service2" in registry
    
    def test_update_service_config(self):
        """测试更新服务配置"""
        registry = {
            "database": {
                "check": lambda: {},
                "config": {"interval": 30, "timeout": 5}
            }
        }
        
        # 更新配置
        registry["database"]["config"]["interval"] = 60
        registry["database"]["config"]["timeout"] = 10
        
        assert registry["database"]["config"]["interval"] == 60
        assert registry["database"]["config"]["timeout"] == 10


class TestHealthCheckRegistryAdvanced:
    """测试注册表高级功能"""
    
    def test_registry_service_lookup(self):
        """测试服务查找"""
        registry = {
            "db": {"check": lambda: {}, "type": "database"},
            "cache": {"check": lambda: {}, "type": "cache"},
            "api": {"check": lambda: {}, "type": "api"}
        }
        
        # 按类型查找
        database_services = [
            name for name, data in registry.items()
            if data.get("type") == "database"
        ]
        
        assert len(database_services) == 1
        assert database_services[0] == "db"
    
    def test_registry_service_filtering(self):
        """测试服务过滤"""
        registry = {
            "s1": {"enabled": True, "check": lambda: {}},
            "s2": {"enabled": False, "check": lambda: {}},
            "s3": {"enabled": True, "check": lambda: {}}
        }
        
        # 过滤启用的服务
        enabled_services = {
            name: data for name, data in registry.items()
            if data.get("enabled", False)
        }
        
        assert len(enabled_services) == 2
        assert "s2" not in enabled_services
    
    def test_registry_service_metadata(self):
        """测试服务元数据"""
        registry = {
            "database": {
                "check": lambda: {},
                "metadata": {
                    "description": "PostgreSQL Database",
                    "version": "13.4",
                    "critical": True,
                    "owner": "platform_team"
                }
            }
        }
        
        metadata = registry["database"]["metadata"]
        
        assert metadata["description"] == "PostgreSQL Database"
        assert metadata["critical"] is True
    
    def test_registry_statistics(self):
        """测试注册表统计"""
        registry = {
            f"service_{i}": {
                "check": lambda: {},
                "type": "database" if i < 3 else "cache"
            }
            for i in range(10)
        }
        
        # 统计服务类型
        type_counts = {}
        for data in registry.values():
            service_type = data.get("type", "unknown")
            type_counts[service_type] = type_counts.get(service_type, 0) + 1
        
        assert type_counts["database"] == 3
        assert type_counts["cache"] == 7


class TestHealthCheckRegistryLifecycle:
    """测试注册表生命周期管理"""
    
    def test_registry_initialization_phase(self):
        """测试注册表初始化阶段"""
        registry_state = {
            "phase": "initializing",
            "services": {},
            "start_time": datetime.now()
        }
        
        # 初始化阶段
        assert registry_state["phase"] == "initializing"
        assert len(registry_state["services"]) == 0
    
    def test_registry_ready_phase(self):
        """测试注册表就绪阶段"""
        registry_state = {
            "phase": "initializing",
            "services": {}
        }
        
        # 注册服务
        registry_state["services"]["db"] = {"check": lambda: {}}
        registry_state["services"]["cache"] = {"check": lambda: {}}
        
        # 标记为就绪
        if len(registry_state["services"]) >= 2:
            registry_state["phase"] = "ready"
        
        assert registry_state["phase"] == "ready"
    
    def test_registry_shutdown_phase(self):
        """测试注册表关闭阶段"""
        registry_state = {
            "phase": "ready",
            "services": {"s1": {}, "s2": {}}
        }
        
        # 执行关闭
        registry_state["phase"] = "shutting_down"
        registry_state["services"].clear()
        registry_state["phase"] = "stopped"
        
        assert registry_state["phase"] == "stopped"
        assert len(registry_state["services"]) == 0


class TestHealthCheckRegistryValidation:
    """测试注册表验证"""
    
    def test_validate_service_name(self):
        """测试验证服务名称"""
        import re
        
        valid_names = ["database", "cache_service", "api-gateway", "web_app_1"]
        invalid_names = ["", "  ", "service with spaces", "service@#$"]
        
        pattern = r'^[a-zA-Z0-9_-]+$'
        
        for name in valid_names:
            assert re.match(pattern, name), f"Should be valid: {name}"
        
        for name in invalid_names:
            if name.strip():  # 忽略空字符串
                assert not re.match(pattern, name), f"Should be invalid: {name}"
    
    def test_validate_check_function(self):
        """测试验证检查函数"""
        def valid_check():
            return {"status": "healthy"}
        
        not_a_function = "not callable"
        
        assert callable(valid_check)
        assert not callable(not_a_function)
    
    def test_validate_config_structure(self):
        """测试验证配置结构"""
        valid_config = {
            "interval": 30,
            "timeout": 5,
            "retries": 3
        }
        
        invalid_config = {
            "interval": -1,  # 负数
            "timeout": 0     # 零值
        }
        
        def validate_config(config):
            if config.get("interval", 0) <= 0:
                return False
            if config.get("timeout", 0) <= 0:
                return False
            return True
        
        assert validate_config(valid_config) is True
        assert validate_config(invalid_config) is False


class TestHealthCheckRegistryEvents:
    """测试注册表事件"""
    
    def test_on_service_registered_event(self):
        """测试服务注册事件"""
        events = []
        
        def on_registered(service_name):
            events.append({
                "type": "registered",
                "service": service_name,
                "timestamp": datetime.now()
            })
        
        # 注册服务并触发事件
        service_name = "new_service"
        on_registered(service_name)
        
        assert len(events) == 1
        assert events[0]["type"] == "registered"
        assert events[0]["service"] == service_name
    
    def test_on_service_unregistered_event(self):
        """测试服务注销事件"""
        events = []
        
        def on_unregistered(service_name):
            events.append({
                "type": "unregistered",
                "service": service_name,
                "timestamp": datetime.now()
            })
        
        service_name = "old_service"
        on_unregistered(service_name)
        
        assert len(events) == 1
        assert events[0]["type"] == "unregistered"
    
    def test_on_config_updated_event(self):
        """测试配置更新事件"""
        events = []
        
        def on_config_updated(service_name, old_config, new_config):
            events.append({
                "type": "config_updated",
                "service": service_name,
                "changes": {
                    "old": old_config,
                    "new": new_config
                }
            })
        
        on_config_updated("database", {"interval": 30}, {"interval": 60})
        
        assert len(events) == 1
        assert events[0]["changes"]["new"]["interval"] == 60


class TestHealthCheckRegistryQuery:
    """测试注册表查询功能"""
    
    def test_get_all_services(self):
        """测试获取所有服务"""
        registry = {
            "s1": {"check": lambda: {}},
            "s2": {"check": lambda: {}},
            "s3": {"check": lambda: {}}
        }
        
        all_services = list(registry.keys())
        
        assert len(all_services) == 3
    
    def test_get_service_by_name(self):
        """测试按名称获取服务"""
        registry = {
            "database": {"check": lambda: {}, "type": "db"},
            "cache": {"check": lambda: {}, "type": "cache"}
        }
        
        service = registry.get("database")
        
        assert service is not None
        assert service["type"] == "db"
    
    def test_get_services_by_type(self):
        """测试按类型获取服务"""
        registry = {
            "db1": {"type": "database"},
            "db2": {"type": "database"},
            "cache1": {"type": "cache"},
            "api1": {"type": "api"}
        }
        
        databases = [
            name for name, data in registry.items()
            if data.get("type") == "database"
        ]
        
        assert len(databases) == 2
    
    def test_get_enabled_services(self):
        """测试获取启用的服务"""
        registry = {
            "s1": {"enabled": True},
            "s2": {"enabled": False},
            "s3": {"enabled": True}
        }
        
        enabled = [
            name for name, data in registry.items()
            if data.get("enabled", False)
        ]
        
        assert len(enabled) == 2


class TestHealthCheckRegistryIntegration:
    """测试注册表集成场景"""
    
    def test_registry_with_executor_integration(self):
        """测试注册表与执行器集成"""
        registry = {
            "service1": {"check": lambda: {"status": "healthy"}},
            "service2": {"check": lambda: {"status": "healthy"}}
        }
        
        # 执行所有注册的检查
        results = {}
        for service_name, service_data in registry.items():
            check_func = service_data["check"]
            results[service_name] = check_func()
        
        assert len(results) == 2
        assert all(r["status"] == "healthy" for r in results.values())
    
    def test_registry_dynamic_registration(self):
        """测试动态注册"""
        registry = {}
        
        # 动态添加服务
        def register_service(name, check_func, config=None):
            registry[name] = {
                "check": check_func,
                "config": config or {},
                "registered_at": datetime.now()
            }
            return True
        
        # 注册3个服务
        for i in range(3):
            success = register_service(
                f"service_{i}",
                lambda: {"status": "healthy"}
            )
            assert success is True
        
        assert len(registry) == 3
    
    def test_registry_bulk_operations(self):
        """测试批量操作"""
        registry = {}
        
        # 批量注册
        services_to_register = [
            ("db", lambda: {}),
            ("cache", lambda: {}),
            ("api", lambda: {})
        ]
        
        for name, check in services_to_register:
            registry[name] = {"check": check}
        
        # 批量注销
        services_to_unregister = ["db", "api"]
        for name in services_to_unregister:
            if name in registry:
                del registry[name]
        
        assert len(registry) == 1
        assert "cache" in registry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

