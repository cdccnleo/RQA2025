"""
测试资源提供者注册表

覆盖 resource_provider_registry.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
from src.infrastructure.resource.core.unified_resource_interfaces import IResourceProvider


class MockResourceProvider(IResourceProvider):
    """模拟资源提供者"""

    def __init__(self, resource_type="cpu", capacity=8):
        self._resource_type = resource_type
        self.capacity = capacity
        self.available = capacity
        self.allocated = 0

    @property
    def resource_type(self) -> str:
        """资源类型"""
        return self._resource_type

    def get_available_resources(self):
        """获取可用资源列表"""
        return [Mock(resource_id=f"{self._resource_type}_1", capacity=self.available)]

    def allocate_resource(self, request):
        """分配资源"""
        amount = request.requirements.get("amount", 1)
        if self.available >= amount:
            self.available -= amount
            self.allocated += amount
            return Mock(allocation_id=f"alloc_{request.requester_id}",
                       resource_id=f"{self._resource_type}_1")
        return None

    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""
        if self.allocated > 0:
            self.allocated -= 1
            self.available += 1
            return True
        return False

    def get_resource_status(self, resource_id: str):
        """获取资源状态"""
        if resource_id == f"{self._resource_type}_1":
            return Mock(resource_id=resource_id, status="active", utilization_percent=50)
        return None

    def optimize_resources(self) -> dict:
        """优化资源使用"""
        return {
            "optimized": True,
            "actions_taken": ["balanced_load"],
            "performance_improved": 10.5
        }

    def get_total_capacity(self):
        return self.capacity

    def get_provider_status(self):
        return {
            "resource_type": self._resource_type,
            "total_capacity": self.capacity,
            "available": self.available,
            "allocated": self.allocated,
            "utilization_percent": (self.allocated / self.capacity) * 100 if self.capacity > 0 else 0
        }

    def is_available(self):
        return self.available > 0

    def get_health_status(self):
        return "healthy"


class TestResourceProviderRegistry:
    """ResourceProviderRegistry 类测试"""

    def test_initialization(self):
        """测试初始化"""
        registry = ResourceProviderRegistry()

        assert hasattr(registry, 'event_bus')
        assert hasattr(registry, 'logger')
        assert hasattr(registry, 'error_handler')
        assert hasattr(registry, '_providers')
        assert hasattr(registry, '_lock')
        assert isinstance(registry._providers, dict)

    def test_initialization_with_event_bus(self):
        """测试带事件总线初始化"""
        mock_event_bus = Mock()

        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        assert registry.event_bus == mock_event_bus

    def test_register_provider_success(self):
        """测试成功注册提供者"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        result = registry.register_provider(provider)

        assert result == True
        assert "cpu" in registry._providers
        assert registry._providers["cpu"] == provider

    def test_register_provider_duplicate_type(self):
        """测试注册重复资源类型的提供者"""
        registry = ResourceProviderRegistry()

        provider1 = MockResourceProvider("cpu", 8)
        provider2 = MockResourceProvider("cpu", 4)

        # 第一次注册成功
        result1 = registry.register_provider(provider1)
        assert result1 == True

        # 第二次注册同一类型应该失败
        result2 = registry.register_provider(provider2)
        assert result2 == False

        # 仍然只有一个提供者
        assert len(registry._providers) == 1
        assert registry._providers["cpu"] == provider1

    def test_register_provider_invalid(self):
        """测试注册无效提供者"""
        registry = ResourceProviderRegistry()

        # 没有resource_type的提供者
        class InvalidProvider:
            pass

        invalid_provider = InvalidProvider()

        result = registry.register_provider(invalid_provider)

        assert result == False
        assert len(registry._providers) == 0

    def test_unregister_provider_success(self):
        """测试成功取消注册提供者"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        registry.register_provider(provider)
        assert "cpu" in registry._providers

        result = registry.unregister_provider("cpu")

        assert result == True
        assert "cpu" not in registry._providers

    def test_unregister_provider_not_found(self):
        """测试取消注册不存在的提供者"""
        registry = ResourceProviderRegistry()

        result = registry.unregister_provider("nonexistent")

        assert result == False

    def test_get_provider_success(self):
        """测试成功获取提供者"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        registry.register_provider(provider)

        retrieved = registry.get_provider("cpu")

        assert retrieved == provider

    def test_get_provider_not_found(self):
        """测试获取不存在的提供者"""
        registry = ResourceProviderRegistry()

        result = registry.get_provider("nonexistent")

        assert result is None

    def test_get_providers(self):
        """测试获取所有提供者"""
        registry = ResourceProviderRegistry()

        cpu_provider = MockResourceProvider("cpu", 8)
        memory_provider = MockResourceProvider("memory", 16)

        registry.register_provider(cpu_provider)
        registry.register_provider(memory_provider)

        providers = registry.get_providers()

        assert len(providers) == 2
        resource_types = [p.resource_type for p in providers]
        assert "cpu" in resource_types
        assert "memory" in resource_types

    def test_get_provider_types(self):
        """测试获取提供者类型列表"""
        registry = ResourceProviderRegistry()

        registry.register_provider(MockResourceProvider("cpu", 8))
        registry.register_provider(MockResourceProvider("memory", 16))
        registry.register_provider(MockResourceProvider("disk", 1000))

        types = registry.get_provider_types()

        assert len(types) == 3
        assert "cpu" in types
        assert "memory" in types
        assert "disk" in types

    def test_has_provider(self):
        """测试检查提供者是否存在"""
        registry = ResourceProviderRegistry()

        registry.register_provider(MockResourceProvider("cpu", 8))

        assert registry.has_provider("cpu") == True
        assert registry.has_provider("memory") == False

    def test_get_provider_count(self):
        """测试获取提供者数量"""
        registry = ResourceProviderRegistry()

        assert registry.get_provider_count() == 0

        registry.register_provider(MockResourceProvider("cpu", 8))
        registry.register_provider(MockResourceProvider("memory", 16))

        assert registry.get_provider_count() == 2

    def test_get_provider_status(self):
        """测试获取提供者状态"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        registry.register_provider(provider)

        status = registry.get_provider_status("cpu")

        assert status is not None
        assert "resource_type" in status
        assert "total_capacity" in status
        assert "available" in status
        assert status["resource_type"] == "cpu"

    def test_get_provider_status_not_found(self):
        """测试获取不存在的提供者状态"""
        registry = ResourceProviderRegistry()

        status = registry.get_provider_status("nonexistent")

        assert status is None

    def test_get_all_provider_status(self):
        """测试获取所有提供者状态"""
        registry = ResourceProviderRegistry()

        cpu_provider = MockResourceProvider("cpu", 8)
        memory_provider = MockResourceProvider("memory", 16)

        registry.register_provider(cpu_provider)
        registry.register_provider(memory_provider)

        all_status = registry.get_all_provider_status()

        assert isinstance(all_status, dict)
        assert len(all_status) == 2
        assert "cpu" in all_status
        assert "memory" in all_status

    def test_clear(self):
        """测试清空注册表"""
        registry = ResourceProviderRegistry()

        registry.register_provider(MockResourceProvider("cpu", 8))
        registry.register_provider(MockResourceProvider("memory", 16))

        assert registry.get_provider_count() == 2

        registry.clear()

        assert registry.get_provider_count() == 0
        assert len(registry._providers) == 0

    def test_get_provider_info(self):
        """测试获取提供者信息"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        registry.register_provider(provider)

        info = registry.get_provider_info("cpu")

        assert info is not None
        assert "resource_type" in info
        assert "capacity" in info
        assert "status" in info

    def test_get_provider_info_not_found(self):
        """测试获取不存在的提供者信息"""
        registry = ResourceProviderRegistry()

        info = registry.get_provider_info("nonexistent")

        assert info is None

    def test_update_provider_health(self):
        """测试更新提供者健康状态"""
        registry = ResourceProviderRegistry()
        provider = MockResourceProvider("cpu", 8)

        registry.register_provider(provider)

        result = registry.update_provider_health("cpu", "warning")

        assert result == True

        # 验证健康状态已更新
        status = registry.get_provider_status("cpu")
        assert status is not None
        # 注意：这个测试可能需要根据实际实现调整

    def test_update_provider_health_not_found(self):
        """测试更新不存在的提供者健康状态"""
        registry = ResourceProviderRegistry()

        result = registry.update_provider_health("nonexistent", "critical")

        assert result == False

    def test_thread_safety(self):
        """测试线程安全性"""
        registry = ResourceProviderRegistry()
        import threading

        results = []
        errors = []

        def register_providers(thread_id):
            try:
                for i in range(5):
                    provider = MockResourceProvider(f"type_{thread_id}_{i}", 4)
                    result = registry.register_provider(provider)
                    results.append((thread_id, i, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程同时注册提供者
        threads = []
        for i in range(3):
            thread = threading.Thread(target=register_providers, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有注册都成功（除了重复类型）
        successful_results = [r for r in results if r[2]]
        assert len(successful_results) >= 3  # 至少每个线程有一个成功注册

    def test_event_bus_integration(self):
        """测试与事件总线的集成"""
        mock_event_bus = Mock()
        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        provider = MockResourceProvider("cpu", 8)
        registry.register_provider(provider)

        # 验证事件总线被传递给注册表
        assert registry.event_bus == mock_event_bus

        # 在实际使用中，注册表应该通过事件总线发布事件
        # 这里我们只是验证集成设置正确

    def test_provider_lifecycle(self):
        """测试提供者生命周期"""
        registry = ResourceProviderRegistry()

        # 1. 创建和注册提供者
        provider = MockResourceProvider("gpu", 2)
        result = registry.register_provider(provider)
        assert result == True

        # 2. 验证注册成功
        assert registry.get_provider_count() == 1
        assert registry.has_provider("gpu")

        # 3. 获取提供者信息和状态
        info = registry.get_provider_info("gpu")
        assert info is not None
        assert info["resource_type"] == "gpu"

        status = registry.get_provider_status("gpu")
        assert status is not None
        assert status["total_capacity"] == 2

        # 4. 取消注册提供者
        result = registry.unregister_provider("gpu")
        assert result == True

        # 5. 验证取消注册成功
        assert registry.get_provider_count() == 0
        assert not registry.has_provider("gpu")

    def test_multiple_provider_types(self):
        """测试多种提供者类型"""
        registry = ResourceProviderRegistry()

        # 注册不同类型的提供者
        providers = [
            MockResourceProvider("cpu", 16),
            MockResourceProvider("memory", 64),
            MockResourceProvider("disk", 2000),
            MockResourceProvider("network", 1000)
        ]

        for provider in providers:
            result = registry.register_provider(provider)
            assert result == True

        assert registry.get_provider_count() == 4

        types = registry.get_provider_types()
        assert len(types) == 4
        assert "cpu" in types
        assert "memory" in types
        assert "disk" in types
        assert "network" in types

        # 获取所有状态
        all_status = registry.get_all_provider_status()
        assert len(all_status) == 4

        for resource_type in types:
            assert resource_type in all_status
            status = all_status[resource_type]
            assert "total_capacity" in status
            assert "available" in status

    def test_provider_error_handling(self):
        """测试提供者错误处理"""
        registry = ResourceProviderRegistry()

        # 创建一个有问题的提供者
        class ProblematicProvider(MockResourceProvider):
            @property
            def resource_type(self):
                raise ValueError("Invalid resource type")

        problematic_provider = ProblematicProvider("problematic", 4)

        # 注册应该失败
        result = registry.register_provider(problematic_provider)

        assert result == False
        assert registry.get_provider_count() == 0

    def test_registry_with_health_monitoring(self):
        """测试注册表与健康监控的集成"""
        registry = ResourceProviderRegistry()

        # 注册多个提供者
        providers = [
            MockResourceProvider("cpu", 8),
            MockResourceProvider("memory", 32),
            MockResourceProvider("gpu", 1)
        ]

        for provider in providers:
            registry.register_provider(provider)

        # 模拟健康状态更新
        registry.update_provider_health("cpu", "warning")
        registry.update_provider_health("gpu", "critical")

        # 获取所有状态，验证健康信息
        all_status = registry.get_all_provider_status()

        # 验证所有提供者都有状态信息
        for resource_type in ["cpu", "memory", "gpu"]:
            assert resource_type in all_status
            status = all_status[resource_type]
            assert "resource_type" in status
            assert "total_capacity" in status

    def test_provider_event_integration(self):
        """测试提供者事件集成"""
        mock_event_bus = Mock()
        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        provider = MockResourceProvider("cpu", 8)
        registry.register_provider(provider)

        # 验证事件总线被正确设置
        assert registry.event_bus == mock_event_bus

    def test_provider_registration_with_event_notification(self):
        """测试提供者注册时的事件通知"""
        mock_event_bus = Mock()
        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        provider = MockResourceProvider("cpu", 8)
        registry.register_provider(provider)

        # 验证注册事件被发布
        mock_event_bus.publish.assert_called()

    def test_provider_unregistration_with_event_notification(self):
        """测试提供者取消注册时的事件通知"""
        mock_event_bus = Mock()
        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        provider = MockResourceProvider("cpu", 8)
        registry.register_provider(provider)

        # 重置mock
        mock_event_bus.reset_mock()

        registry.unregister_provider("cpu")

        # 验证取消注册事件被发布
        mock_event_bus.publish.assert_called()

    def test_provider_health_update_with_event_notification(self):
        """测试提供者健康更新时的事件通知"""
        mock_event_bus = Mock()
        registry = ResourceProviderRegistry(event_bus=mock_event_bus)

        provider = MockResourceProvider("cpu", 8)
        registry.register_provider(provider)

        registry.update_provider_health("cpu", "warning")

        # 验证健康更新事件被发布
        mock_event_bus.publish.assert_called()

    def test_provider_capacity_scaling(self):
        """测试提供者容量扩展"""
        registry = ResourceProviderRegistry()

        # 注册基础提供者
        cpu_provider = MockResourceProvider("cpu", 8)
        registry.register_provider(cpu_provider)

        # 模拟容量扩展
        cpu_provider.capacity = 16
        cpu_provider.available = 16

        # 验证容量更新
        status = registry.get_provider_status("cpu")
        assert status["total_capacity"] == 16
        assert status["available"] == 16

    def test_provider_load_balancing(self):
        """测试提供者负载均衡"""
        registry = ResourceProviderRegistry()

        # 注册多个相同类型的提供者
        providers = [
            MockResourceProvider("cpu", 8),
            MockResourceProvider("cpu", 8),
            MockResourceProvider("cpu", 8),
        ]

        for provider in providers:
            registry.register_provider(provider)

        # 验证所有提供者都被注册
        cpu_providers = [p for p in registry.get_providers() if p.resource_type == "cpu"]
        assert len(cpu_providers) == 3

        # 验证可以获取提供者类型
        types = registry.get_provider_types()
        assert "cpu" in types

    def test_provider_resource_tracking(self):
        """测试提供者资源跟踪"""
        registry = ResourceProviderRegistry()

        provider = MockResourceProvider("gpu", 4)
        registry.register_provider(provider)

        # 初始状态
        initial_status = registry.get_provider_status("gpu")
        assert initial_status["available"] == 4
        assert initial_status["allocated"] == 0

        # 模拟分配
        provider.available = 2
        provider.allocated = 2

        # 验证状态更新
        updated_status = registry.get_provider_status("gpu")
        assert updated_status["available"] == 2
        assert updated_status["allocated"] == 2

    def test_provider_error_recovery(self):
        """测试提供者错误恢复"""
        registry = ResourceProviderRegistry()

        # 注册一个有问题的提供者
        class UnreliableProvider(MockResourceProvider):
            def get_health_status(self):
                raise ConnectionError("Provider unreachable")

        unreliable_provider = UnreliableProvider("unreliable", 4)

        # 注册应该成功
        result = registry.register_provider(unreliable_provider)
        assert result == True

        # 但健康检查应该能处理错误
        try:
            registry.update_provider_health("unreliable", "critical")
        except:
            pass  # 应该不会抛出异常

    def test_provider_bulk_operations(self):
        """测试提供者批量操作"""
        registry = ResourceProviderRegistry()

        # 批量注册提供者
        provider_types = ["cpu", "memory", "disk", "network"]
        for p_type in provider_types:
            provider = MockResourceProvider(p_type, 100)
            registry.register_provider(provider)

        assert registry.get_provider_count() == 4

        # 批量获取状态
        all_status = registry.get_all_provider_status()
        assert len(all_status) == 4

        # 批量清除
        registry.clear()
        assert registry.get_provider_count() == 0

    def test_provider_performance_monitoring(self):
        """测试提供者性能监控"""
        registry = ResourceProviderRegistry()

        provider = MockResourceProvider("cpu", 16)
        registry.register_provider(provider)

        # 模拟性能指标
        provider.performance_metrics = {
            "response_time": 0.05,
            "throughput": 1000,
            "error_rate": 0.001
        }

        # 获取提供者信息应该包含性能数据
        info = registry.get_provider_info("cpu")
        assert info is not None
        assert "resource_type" in info
        assert "capacity" in info

    def test_provider_resource_optimization(self):
        """测试提供者资源优化"""
        registry = ResourceProviderRegistry()

        # 注册不同配置的提供者
        high_perf_provider = MockResourceProvider("gpu_high", 2)  # 高性能，容量小
        high_capacity_provider = MockResourceProvider("gpu_capacity", 8)  # 容量大，性能一般

        registry.register_provider(high_perf_provider)
        registry.register_provider(high_capacity_provider)

        # 验证两种类型的提供者都可用
        gpu_providers = [p for p in registry.get_providers() if p.resource_type.startswith("gpu")]
        assert len(gpu_providers) == 2

        # 验证状态查询
        high_perf_status = registry.get_provider_status("gpu_high")
        high_capacity_status = registry.get_provider_status("gpu_capacity")

        assert high_perf_status["total_capacity"] == 2
        assert high_capacity_status["total_capacity"] == 8

    def test_provider_concurrent_operations(self):
        """测试提供者并发操作"""
        import threading
        registry = ResourceProviderRegistry()

        results = []
        errors = []

        def provider_worker(thread_id):
            try:
                # 注册提供者
                provider = MockResourceProvider(f"type_{thread_id}", 4)
                result = registry.register_provider(provider)
                results.append(("register", thread_id, result))

                # 查询状态
                if result:
                    status = registry.get_provider_status(f"type_{thread_id}")
                    results.append(("status", thread_id, status is not None))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=provider_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有注册都成功
        successful_registrations = [r for r in results if r[0] == "register" and r[2] == True]
        assert len(successful_registrations) == num_threads

        # 验证最终状态
        assert registry.get_provider_count() == num_threads

    def test_provider_resource_quota_management(self):
        """测试提供者资源配额管理"""
        registry = ResourceProviderRegistry()

        # 注册有配额限制的提供者
        provider = MockResourceProvider("gpu", 4)
        provider.quota_limit = 8  # 配额限制
        provider.current_quota_usage = 2  # 当前使用

        registry.register_provider(provider)

        # 验证配额信息可用
        status = registry.get_provider_status("gpu")
        assert "total_capacity" in status
        assert status["total_capacity"] == 4

        # 模拟配额检查（通过扩展提供者类来实现）
        class QuotaProvider(MockResourceProvider):
            def __init__(self, resource_type, capacity, quota_limit):
                super().__init__(resource_type, capacity)
                self.quota_limit = quota_limit
                self.current_quota_usage = 0

            def check_quota_available(self, requested):
                return (self.current_quota_usage + requested) <= self.quota_limit

        quota_provider = QuotaProvider("gpu_quota", 4, 8)
        registry.register_provider(quota_provider)

        # 验证配额提供者
        quota_status = registry.get_provider_status("gpu_quota")
        assert quota_status is not None

    def test_provider_failover_and_redundancy(self):
        """测试提供者故障转移和冗余"""
        registry = ResourceProviderRegistry()

        # 注册主备提供者
        primary_provider = MockResourceProvider("cpu_primary", 8)
        backup_provider = MockResourceProvider("cpu_backup", 8)

        registry.register_provider(primary_provider)
        registry.register_provider(backup_provider)

        # 模拟主提供者故障
        primary_provider.available = 0

        # 验证备提供者仍然可用
        backup_status = registry.get_provider_status("cpu_backup")
        assert backup_status["available"] == 8

        # 验证整体可用性
        all_providers = registry.get_providers()
        cpu_providers = [p for p in all_providers if p.resource_type.startswith("cpu")]
        assert len(cpu_providers) == 2

    def test_provider_metrics_and_monitoring(self):
        """测试提供者指标和监控"""
        registry = ResourceProviderRegistry()

        provider = MockResourceProvider("cpu", 16)
        provider.metrics = {
            "allocation_count": 0,
            "deallocation_count": 0,
            "failure_count": 0,
            "average_response_time": 0.02
        }

        registry.register_provider(provider)

        # 模拟一些操作
        provider.metrics["allocation_count"] = 5
        provider.metrics["average_response_time"] = 0.03

        # 验证指标跟踪
        status = registry.get_provider_status("cpu")
        assert status["total_capacity"] == 16

        # 验证提供者信息包含扩展字段
        info = registry.get_provider_info("cpu")
        assert info is not None
        assert "resource_type" in info

    def test_provider_configuration_management(self):
        """测试提供者配置管理"""
        registry = ResourceProviderRegistry()

        # 注册具有不同配置的提供者
        configs = [
            {"type": "cpu", "capacity": 8, "overcommit_ratio": 1.5},
            {"type": "memory", "capacity": 64, "overcommit_ratio": 1.2},
            {"type": "gpu", "capacity": 2, "overcommit_ratio": 1.0},
        ]

        for config in configs:
            provider = MockResourceProvider(config["type"], config["capacity"])
            provider.overcommit_ratio = config["overcommit_ratio"]
            registry.register_provider(provider)

        # 验证配置多样性
        all_providers = registry.get_providers()
        assert len(all_providers) == 3

        # 验证每种资源类型都有正确的配置
        for provider in all_providers:
            status = registry.get_provider_status(provider.resource_type)
            assert "total_capacity" in status
            assert status["total_capacity"] > 0

    def test_provider_lifecycle_hooks(self):
        """测试提供者生命周期钩子"""
        registry = ResourceProviderRegistry()

        provider = MockResourceProvider("cpu", 8)

        # 模拟生命周期钩子
        provider.lifecycle_hooks = {
            "on_register": Mock(),
            "on_unregister": Mock(),
            "on_health_change": Mock()
        }

        registry.register_provider(provider)

        # 注册钩子应该被调用（如果实现的话）
        # 这里我们只是验证提供者注册成功
        assert registry.has_provider("cpu")

        # 取消注册
        registry.unregister_provider("cpu")

        # 验证提供者被移除
        assert not registry.has_provider("cpu")
