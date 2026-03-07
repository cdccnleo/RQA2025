#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resource_provider_registry 模块测试
测试资源提供者注册表的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional

try:
    from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
    from src.infrastructure.resource.core.unified_resource_interfaces import (
        IResourceProvider, ResourceInfo, ResourceRequest, ResourceAllocation
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_provider_registry模块导入失败")
class TestResourceProvider(unittest.TestCase):
    """测试资源提供者实现"""
    
    def setUp(self):
        """创建测试用的资源提供者"""
        class TestProvider(IResourceProvider):
            def __init__(self, resource_type="test_resource"):
                self._resource_type = resource_type
                self.resources = []
                self.allocations = {}
            
            @property
            def resource_type(self) -> str:
                return self._resource_type
            
            def get_available_resources(self) -> List[ResourceInfo]:
                return self.resources.copy()
            
            def allocate_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
                if not self.resources:
                    return None
                
                resource = self.resources.pop(0)
                allocation = ResourceAllocation(
                    allocation_id=f"alloc_{request.request_id}",
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    allocated_resources=request.requirements
                )
                self.allocations[allocation.allocation_id] = allocation
                return allocation
            
            def release_resource(self, allocation_id: str) -> bool:
                if allocation_id in self.allocations:
                    allocation = self.allocations.pop(allocation_id)
                    # 将资源返回到可用列表
                    resource_info = ResourceInfo(
                        resource_id=allocation.resource_id,
                        resource_type=self.resource_type,
                        name=f"released_resource_{allocation.resource_id}"
                    )
                    self.resources.append(resource_info)
                    return True
                return False
            
            def get_resource_status(self, resource_id: str) -> Optional[ResourceInfo]:
                for resource in self.resources:
                    if resource.resource_id == resource_id:
                        return resource
                return None
            
            def optimize_resources(self) -> Dict[str, Any]:
                return {"optimized": True, "efficiency": 95.0}
        
        self.TestProvider = TestProvider

    def test_provider_interface_implementation(self):
        """测试提供者接口实现"""
        provider = self.TestProvider("cpu")
        
        # 测试资源类型
        self.assertEqual(provider.resource_type, "cpu")
        
        # 测试获取可用资源
        resources = provider.get_available_resources()
        self.assertIsInstance(resources, list)
        
        # 测试优化资源
        optimization = provider.optimize_resources()
        self.assertIn("optimized", optimization)


@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_provider_registry模块导入失败") 
class TestResourceProviderRegistry(unittest.TestCase):
    """测试资源提供者注册表"""

    def setUp(self):
        """测试前准备"""
        # Mock事件总线
        mock_event_bus = Mock()
        
        self.registry = ResourceProviderRegistry(event_bus=mock_event_bus)
        
        # 创建测试提供者
        class TestProvider(IResourceProvider):
            def __init__(self, resource_type="test_resource"):
                self._resource_type = resource_type
                self.resources = []
                self.allocations = {}
            
            @property
            def resource_type(self) -> str:
                return self._resource_type
            
            def get_available_resources(self) -> List[ResourceInfo]:
                return self.resources.copy()
            
            def allocate_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
                return None
            
            def release_resource(self, allocation_id: str) -> bool:
                return True
            
            def get_resource_status(self, resource_id: str) -> Optional[ResourceInfo]:
                return None
            
            def optimize_resources(self) -> Dict[str, Any]:
                return {"efficiency": 100.0}
        
        self.TestProvider = TestProvider

    def test_registry_initialization(self):
        """测试注册表初始化"""
        self.assertIsNotNone(self.registry)
        self.assertIsNotNone(self.registry.logger)
        self.assertIsNotNone(self.registry.error_handler)
        self.assertIsNotNone(self.registry.event_bus)

    def test_register_provider_success(self):
        """测试成功注册提供者"""
        provider = self.TestProvider("cpu")
        
        result = self.registry.register_provider(provider)
        self.assertTrue(result)
        
        # 验证提供者已注册
        retrieved_provider = self.registry.get_provider("cpu")
        self.assertEqual(retrieved_provider, provider)

    def test_register_duplicate_provider_type(self):
        """测试注册重复资源类型的提供者"""
        provider1 = self.TestProvider("cpu")
        provider2 = self.TestProvider("cpu")
        
        # 第一次注册应该成功
        result1 = self.registry.register_provider(provider1)
        self.assertTrue(result1)
        
        # 第二次注册相同资源类型应该失败
        result2 = self.registry.register_provider(provider2)
        self.assertFalse(result2)

    def test_unregister_provider(self):
        """测试注销提供者"""
        provider = self.TestProvider("memory")
        
        # 注册提供者
        self.registry.register_provider(provider)
        self.assertTrue(self.registry.has_provider("memory"))
        
        # 注销提供者
        result = self.registry.unregister_provider("memory")
        self.assertTrue(result)
        self.assertFalse(self.registry.has_provider("memory"))

    def test_unregister_nonexistent_provider(self):
        """测试注销不存在的提供者"""
        result = self.registry.unregister_provider("nonexistent_resource")
        self.assertFalse(result)

    def test_get_provider(self):
        """测试获取提供者"""
        provider = self.TestProvider("disk")
        
        self.registry.register_provider(provider)
        
        retrieved_provider = self.registry.get_provider("disk")
        self.assertIsNotNone(retrieved_provider)
        self.assertEqual(retrieved_provider, provider)

    def test_get_nonexistent_provider(self):
        """测试获取不存在的提供者"""
        provider = self.registry.get_provider("nonexistent_resource")
        self.assertIsNone(provider)

    def test_get_providers(self):
        """测试获取所有提供者"""
        provider1 = self.TestProvider("cpu")
        provider2 = self.TestProvider("memory")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        providers = self.registry.get_providers()
        self.assertEqual(len(providers), 2)
        # get_providers返回的是提供者对象列表，不是字典
        provider_list = list(providers)
        self.assertEqual(len(provider_list), 2)

    def test_has_provider(self):
        """测试检查提供者是否存在"""
        provider = self.TestProvider("gpu")
        
        # 注册前应该不存在
        self.assertFalse(self.registry.has_provider("gpu"))
        
        self.registry.register_provider(provider)
        self.assertTrue(self.registry.has_provider("gpu"))

    def test_get_provider_status(self):
        """测试获取提供者状态"""
        provider = self.TestProvider("network")
        
        self.registry.register_provider(provider)
        
        status = self.registry.get_provider_status("network")
        self.assertIsNotNone(status)
        self.assertIn("resource_type", status)
        self.assertIn("status", status)

    def test_get_nonexistent_provider_status(self):
        """测试获取不存在提供者的状态"""
        status = self.registry.get_provider_status("nonexistent")
        self.assertIsNone(status)


    def test_get_provider_count(self):
        """测试获取提供者数量"""
        self.assertEqual(self.registry.get_provider_count(), 0)
        
        provider1 = self.TestProvider("type1")
        provider2 = self.TestProvider("type2")
        
        self.registry.register_provider(provider1)
        self.assertEqual(self.registry.get_provider_count(), 1)
        
        self.registry.register_provider(provider2)
        self.assertEqual(self.registry.get_provider_count(), 2)

    def test_get_provider_types(self):
        """测试获取提供者类型列表"""
        provider1 = self.TestProvider("cpu")
        provider2 = self.TestProvider("memory")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        types = self.registry.get_provider_types()
        self.assertEqual(len(types), 2)
        self.assertIn("cpu", types)
        self.assertIn("memory", types)

    def test_event_bus_integration(self):
        """测试事件总线集成"""
        provider = self.TestProvider("event_test")
        
        # 注册提供者应该触发事件发布
        result = self.registry.register_provider(provider)
        self.assertTrue(result)
        
        # 验证事件总线被调用
        self.registry.event_bus.publish.assert_called()

    def test_error_handling_in_register(self):
        """测试注册过程中的错误处理"""
        # 测试传递None作为提供者
        result = self.registry.register_provider(None)
        self.assertFalse(result)
        
        # 测试提供者没有resource_type属性
        class InvalidProvider:
            pass
        
        invalid_provider = InvalidProvider()
        result = self.registry.register_provider(invalid_provider)
        self.assertFalse(result)

    def test_registry_thread_safety(self):
        """测试注册表的线程安全性"""
        import threading
        import time
        
        def register_providers():
            for i in range(5):
                # 需要确保每个提供者都有唯一的resource_type，因为注册表不允许重复类型
                provider = self.TestProvider(f"thread_resource_{threading.current_thread().ident}_{i}")
                self.registry.register_provider(provider)
                time.sleep(0.001)  # 模拟并发
        
        # 创建多个线程同时注册提供者
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=register_providers)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证最终状态 - 每个线程注册5个，总共15个
        self.assertEqual(self.registry.get_provider_count(), 15)

    def test_concurrent_access(self):
        """测试并发访问"""
        import threading
        
        results = {}
        
        def register_and_get(index):
            provider = self.TestProvider(f"concurrent_{index}")
            register_result = self.registry.register_provider(provider)
            get_result = self.registry.get_provider(f"concurrent_{index}")
            results[index] = (register_result, get_result is not None)
        
        # 创建多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_and_get, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), 10)
        for index, (register_result, get_result) in results.items():
            self.assertTrue(register_result)
            self.assertTrue(get_result)

    def test_provider_status_with_optimization(self):
        """测试包含优化信息的提供者状态"""
        class OptimizableProvider(self.TestProvider):
            def optimize_resources(self) -> Dict[str, Any]:
                return {
                    "efficiency": 98.5,
                    "optimized_count": 10,
                    "memory_saved": 1024
                }
        
        provider = OptimizableProvider("optimizable")
        self.registry.register_provider(provider)
        
        status = self.registry.get_provider_status("optimizable")
        self.assertIsNotNone(status)

    def test_clear_providers(self):
        """测试清空所有提供者"""
        provider1 = self.TestProvider("clear1")
        provider2 = self.TestProvider("clear2")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        self.assertEqual(self.registry.get_provider_count(), 2)
        
        # 测试clear方法
        self.registry.clear()
        self.assertEqual(self.registry.get_provider_count(), 0)

    def test_provider_with_none_event_bus(self):
        """测试没有事件总线的注册表"""
        registry_no_bus = ResourceProviderRegistry(event_bus=None)
        provider = self.TestProvider("no_bus_test")
        
        # 应该不会因为事件总线为None而失败
        result = registry_no_bus.register_provider(provider)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
