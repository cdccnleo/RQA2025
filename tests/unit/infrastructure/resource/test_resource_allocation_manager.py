#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resource_allocation_manager 模块测试
测试资源分配管理器的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional, List

try:
    from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
    from src.infrastructure.resource.core.unified_resource_interfaces import (
        IResourceProvider, ResourceAllocation, ResourceRequest,
        ResourceError, ResourceNotFoundError
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")




@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_allocation_manager模块导入失败")
class TestResourceAllocationManager(unittest.TestCase):
    """测试资源分配管理器"""

    def setUp(self):
        """测试前准备"""
        # 创建模拟的依赖组件
        self.mock_provider_registry = Mock()
        self.mock_event_bus = Mock()
        
        # 创建测试提供者类
        class TestProvider(IResourceProvider):
            def __init__(self, resource_type="test_resource"):
                self._resource_type = resource_type
                self.allocations = {}
            
            @property
            def resource_type(self) -> str:
                return self._resource_type
            
            def get_available_resources(self) -> List:
                return []
            
            def allocate_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
                if request.requirements.get("should_fail", False):
                    return None
                
                allocation = ResourceAllocation(
                    allocation_id=f"alloc_{request.request_id}",
                    request_id=request.request_id,
                    resource_id=f"res_{request.resource_type}",
                    allocated_resources=request.requirements
                )
                # 设置resource_type属性，因为源代码中需要用到
                allocation.resource_type = request.resource_type
                self.allocations[allocation.allocation_id] = allocation
                return allocation
            
            def release_resource(self, allocation_id: str) -> bool:
                return allocation_id in self.allocations
            
            def get_resource_status(self, resource_id: str):
                return None
            
            def optimize_resources(self) -> Dict[str, Any]:
                return {"optimized": True}
        
        self.test_provider = TestProvider("cpu")
        
        self.manager = ResourceAllocationManager(
            provider_registry=self.mock_provider_registry,
            event_bus=self.mock_event_bus
        )
        
        # 设置默认mock行为
        self.mock_provider_registry.has_provider.return_value = True
        self.mock_provider_registry.get_provider.return_value = self.test_provider

    def test_manager_initialization(self):
        """测试管理器初始化"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.provider_registry, self.mock_provider_registry)
        self.assertEqual(self.manager.event_bus, self.mock_event_bus)
        self.assertIsInstance(self.manager._allocations, dict)
        self.assertIsInstance(self.manager._requests, dict)

    def test_request_resource_success(self):
        """测试成功请求资源"""
        consumer_id = "test_consumer"
        resource_type = "cpu"
        requirements = {"cores": 2, "memory": 1024}
        
        result = self.manager.request_resource(consumer_id, resource_type, requirements)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("alloc_req_"))

    def test_request_resource_provider_not_found(self):
        """测试请求资源时提供者不存在"""
        self.mock_provider_registry.has_provider.return_value = False
        
        result = self.manager.request_resource("consumer", "nonexistent", {})
        
        self.assertIsNone(result)

    def test_request_resource_allocation_failed(self):
        """测试资源分配失败"""
        consumer_id = "test_consumer"
        resource_type = "cpu"
        requirements = {"should_fail": True}  # 使用特殊标记让分配失败
        
        result = self.manager.request_resource(consumer_id, resource_type, requirements)
        
        self.assertIsNone(result)

    def test_validate_and_get_provider_success(self):
        """测试验证和获取提供者成功"""
        provider = self.manager._validate_and_get_provider("cpu")
        self.assertEqual(provider, self.test_provider)

    def test_validate_and_get_provider_not_found(self):
        """测试验证和获取提供者失败"""
        self.mock_provider_registry.has_provider.return_value = False
        
        with self.assertRaises(ResourceNotFoundError):
            self.manager._validate_and_get_provider("nonexistent")

    def test_validate_and_get_provider_no_registry(self):
        """测试没有注册表时验证提供者"""
        manager = ResourceAllocationManager(provider_registry=None)
        
        with self.assertRaises(ResourceNotFoundError):
            manager._validate_and_get_provider("any_type")

    def test_create_resource_request(self):
        """测试创建资源请求"""
        consumer_id = "test_consumer"
        resource_type = "cpu"
        requirements = {"cores": 2}
        priority = 3
        
        request = self.manager._create_resource_request(
            consumer_id, resource_type, requirements, priority
        )
        
        self.assertIsInstance(request, ResourceRequest)
        self.assertEqual(request.resource_type, resource_type)
        self.assertEqual(request.requester_id, consumer_id)
        self.assertEqual(request.requirements, requirements)
        self.assertEqual(request.priority, priority)
        self.assertTrue(request.request_id.startswith("req_"))

    def test_store_request(self):
        """测试存储请求"""
        # 创建请求
        request = ResourceRequest(
            request_id="test_req_1",
            resource_type="cpu",
            requester_id="consumer1",
            requirements={"cores": 1}
        )
        
        self.manager._store_request(request)
        
        self.assertIn("test_req_1", self.manager._requests)
        self.assertEqual(self.manager._requests["test_req_1"], request)

    def test_attempt_allocation(self):
        """测试尝试分配资源"""
        request = ResourceRequest(
            request_id="test_req",
            resource_type="cpu",
            requester_id="consumer",
            requirements={"cores": 1}
        )
        
        result = self.manager._attempt_allocation(self.test_provider, request)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ResourceAllocation)

    def test_handle_successful_allocation(self):
        """测试处理成功分配"""
        allocation = ResourceAllocation(
            allocation_id="test_alloc_1",
            request_id="test_req_1",
            resource_id="cpu_res_1",
            allocated_resources={"cores": 2}
        )
        consumer_id = "test_consumer"
        
        result = self.manager._handle_successful_allocation(allocation, consumer_id)
        
        self.assertEqual(result, "test_alloc_1")
        self.assertIn("test_alloc_1", self.manager._allocations)
        
        # 验证事件发布
        self.mock_event_bus.publish.assert_called()

    def test_handle_successful_allocation_no_event_bus(self):
        """测试没有事件总线时的成功分配"""
        manager = ResourceAllocationManager(provider_registry=None, event_bus=None)
        
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req", 
            resource_id="cpu_res",
            allocated_resources={}
        )
        
        result = manager._handle_successful_allocation(allocation, "consumer")
        self.assertEqual(result, "test_alloc")

    def test_handle_failed_allocation(self):
        """测试处理失败分配"""
        resource_type = "cpu"
        consumer_id = "test_consumer"
        requirements = {"cores": 4}
        
        result = self.manager._handle_failed_allocation(resource_type, consumer_id, requirements)
        
        self.assertIsNone(result)
        # 验证事件发布
        self.mock_event_bus.publish.assert_called()

    def test_release_resource_success(self):
        """测试成功释放资源"""
        # 先分配一个资源
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req",
            resource_id="cpu_res",
            allocated_resources={}
        )
        # 设置resource_type属性，因为_get_resource_type方法需要
        allocation.resource_type = "cpu"
        self.manager._allocations["test_alloc"] = allocation
        
        # 确保provider知道这个分配存在
        self.test_provider.allocations["test_alloc"] = allocation
        
        result = self.manager.release_resource("test_alloc")
        
        self.assertTrue(result)
        self.assertNotIn("test_alloc", self.manager._allocations)

    def test_release_resource_not_found(self):
        """测试释放不存在的资源"""
        result = self.manager.release_resource("nonexistent_alloc")
        
        self.assertFalse(result)

    def test_release_resource_provider_not_found(self):
        """测试释放资源时提供者不存在"""
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req", 
            resource_id="cpu_res",
            allocated_resources={}
        )
        # 为测试添加resource_type属性
        allocation.resource_type = "invalid_type"
        self.manager._allocations["test_alloc"] = allocation
        self.mock_provider_registry.has_provider.return_value = False
        
        result = self.manager.release_resource("test_alloc")
        
        self.assertFalse(result)

    def test_release_resource_no_registry(self):
        """测试没有注册表时释放资源"""
        manager = ResourceAllocationManager(provider_registry=None)
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req",
            resource_id="cpu_res", 
            allocated_resources={}
        )
        manager._allocations["test_alloc"] = allocation
        
        result = manager.release_resource("test_alloc")
        
        self.assertFalse(result)

    def test_get_allocation(self):
        """测试获取分配信息"""
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req",
            resource_id="cpu_res",
            allocated_resources={}
        )
        self.manager._allocations["test_alloc"] = allocation
        
        result = self.manager.get_allocation("test_alloc")
        self.assertEqual(result, allocation)
        
        # 测试获取不存在的分配
        result = self.manager.get_allocation("nonexistent")
        self.assertIsNone(result)

    def test_get_request(self):
        """测试获取请求信息"""
        request = ResourceRequest(
            request_id="test_req",
            resource_type="cpu",
            requester_id="consumer",
            requirements={}
        )
        self.manager._requests["test_req"] = request
        
        result = self.manager.get_request("test_req")
        self.assertEqual(result, request)
        
        # 测试获取不存在的请求
        result = self.manager.get_request("nonexistent")
        self.assertIsNone(result)

    def test_get_allocation_count(self):
        """测试获取分配数量"""
        self.assertEqual(self.manager.get_allocation_count(), 0)
        
        # 添加一些分配
        allocation1 = ResourceAllocation(
            allocation_id="alloc1", request_id="req1", resource_id="res1", allocated_resources={}
        )
        allocation2 = ResourceAllocation(
            allocation_id="alloc2", request_id="req2", resource_id="res2", allocated_resources={}
        )
        self.manager._allocations["alloc1"] = allocation1
        self.manager._allocations["alloc2"] = allocation2
        
        self.assertEqual(self.manager.get_allocation_count(), 2)

    def test_get_request_count(self):
        """测试获取请求数量"""
        self.assertEqual(self.manager.get_request_count(), 0)
        
        # 添加一些请求
        request1 = ResourceRequest("req1", "cpu", "consumer1", {})
        request2 = ResourceRequest("req2", "memory", "consumer2", {})
        self.manager._requests["req1"] = request1
        self.manager._requests["req2"] = request2
        
        self.assertEqual(self.manager.get_request_count(), 2)

    def test_get_allocation_summary(self):
        """测试获取分配摘要"""
        summary = self.manager.get_allocation_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_allocations", summary)
        self.assertIn("by_resource_type", summary)
        self.assertIn("by_consumer", summary)

    def test_get_allocation_summary_with_data(self):
        """测试有数据时获取分配摘要"""
        # 添加分配数据
        allocation1 = ResourceAllocation(
            allocation_id="alloc1",
            request_id="req1", 
            resource_id="cpu_res1",
            allocated_resources={"cores": 2}
        )
        allocation2 = ResourceAllocation(
            allocation_id="alloc2",
            request_id="req2",
            resource_id="memory_res1", 
            allocated_resources={"size": 1024}
        )
        
        # 需要设置resource_type属性
        allocation1.resource_type = "cpu"
        allocation2.resource_type = "memory"
        
        self.manager._allocations["alloc1"] = allocation1
        self.manager._allocations["alloc2"] = allocation2
        
        summary = self.manager.get_allocation_summary()
        
        self.assertEqual(summary["total_allocations"], 2)
        self.assertIn("cpu", summary["by_resource_type"])
        self.assertIn("memory", summary["by_resource_type"])

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        
        def worker():
            for i in range(5):
                # 模拟并发请求和释放
                allocation_id = self.manager.request_resource(
                    f"consumer_{i}", "cpu", {"cores": 1}
                )
                if allocation_id:
                    time.sleep(0.01)
                    self.manager.release_resource(allocation_id)
        
        # 创建多个线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证最终状态
        self.assertEqual(self.manager.get_allocation_count(), 0)

    def test_manager_with_none_dependencies(self):
        """测试管理器使用None依赖"""
        manager = ResourceAllocationManager(
            provider_registry=None,
            event_bus=None
        )
        
        self.assertIsNone(manager.provider_registry)
        self.assertIsNone(manager.event_bus)
        
        # 测试在没有依赖时不会崩溃
        result = manager.request_resource("consumer", "cpu", {})
        self.assertIsNone(result)

    def test_error_handling_in_request_resource(self):
        """测试请求资源时的错误处理"""
        # 模拟提供者抛出异常
        self.mock_provider_registry.has_provider.side_effect = Exception("Registry error")
        
        result = self.manager.request_resource("consumer", "cpu", {})
        
        self.assertIsNone(result)

    def test_error_handling_in_release_resource(self):
        """测试释放资源时的错误处理"""
        allocation = ResourceAllocation(
            allocation_id="test_alloc",
            request_id="test_req",
            resource_id="cpu_res",
            allocated_resources={}
        )
        # 设置resource_type属性
        allocation.resource_type = "cpu"
        self.manager._allocations["test_alloc"] = allocation
        
        # 模拟提供者抛出异常
        self.mock_provider_registry.get_provider.side_effect = Exception("Provider error")
        
        result = self.manager.release_resource("test_alloc")
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()