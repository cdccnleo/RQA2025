#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resource_consumer_registry 模块测试
测试资源消费者注册表的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

try:
    from src.infrastructure.resource.core.resource_consumer_registry import ResourceConsumerRegistry
    from src.infrastructure.resource.core.unified_resource_interfaces import IResourceConsumer, ResourceAllocation
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_consumer_registry模块导入失败")
class TestResourceConsumer(unittest.TestCase):
    """测试资源消费者实现"""
    
    def setUp(self):
        """创建测试用的资源消费者"""
        class TestConsumer(IResourceConsumer):
            def __init__(self, name="test_consumer"):
                self.name = name
                self.allocations = []
                self.resource_usage = {"cpu": 50.0, "memory": 1024}
            
            def request_resource(self, resource_type: str, requirements: Dict[str, Any], 
                               priority: int = 1, timeout: float = None) -> str:
                allocation_id = f"alloc_{len(self.allocations)}_{resource_type}"
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    request_id=f"req_{allocation_id}",
                    resource_id=f"res_{resource_type}",
                    allocated_resources=requirements
                )
                self.allocations.append(allocation)
                return allocation_id
            
            def release_resource(self, allocation_id: str) -> bool:
                self.allocations = [a for a in self.allocations if a.allocation_id != allocation_id]
                return True
            
            def get_consumed_resources(self) -> List[ResourceAllocation]:
                return self.allocations.copy()
            
            def get_resource_usage(self) -> Dict[str, Any]:
                return self.resource_usage.copy()
        
        self.TestConsumer = TestConsumer

    def test_consumer_interface_implementation(self):
        """测试消费者接口实现"""
        consumer = self.TestConsumer("test")
        
        # 测试请求资源
        allocation_id = consumer.request_resource("cpu", {"cores": 2})
        self.assertIsNotNone(allocation_id)
        self.assertTrue(allocation_id.startswith("alloc_"))
        
        # 测试获取已消费资源
        allocations = consumer.get_consumed_resources()
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].allocation_id, allocation_id)
        
        # 测试获取资源使用情况
        usage = consumer.get_resource_usage()
        self.assertIn("cpu", usage)
        self.assertIn("memory", usage)
        
        # 测试释放资源
        result = consumer.release_resource(allocation_id)
        self.assertTrue(result)
        self.assertEqual(len(consumer.get_consumed_resources()), 0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_consumer_registry模块导入失败")
class TestResourceConsumerRegistry(unittest.TestCase):
    """测试资源消费者注册表"""

    def setUp(self):
        """测试前准备"""
        self.registry = ResourceConsumerRegistry()
        
        # 创建测试消费者
        class TestConsumer(IResourceConsumer):
            def __init__(self, name="test_consumer"):
                self.name = name
                self.allocations = []
                self.resource_usage = {"cpu": 50.0, "memory": 1024}
            
            def request_resource(self, resource_type: str, requirements: Dict[str, Any], 
                               priority: int = 1, timeout: float = None) -> str:
                allocation_id = f"alloc_{len(self.allocations)}_{resource_type}"
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    request_id=f"req_{allocation_id}",
                    resource_id=f"res_{resource_type}",
                    allocated_resources=requirements
                )
                self.allocations.append(allocation)
                return allocation_id
            
            def release_resource(self, allocation_id: str) -> bool:
                self.allocations = [a for a in self.allocations if a.allocation_id != allocation_id]
                return True
            
            def get_consumed_resources(self) -> List[ResourceAllocation]:
                return self.allocations.copy()
            
            def get_resource_usage(self) -> Dict[str, Any]:
                return self.resource_usage.copy()
        
        self.TestConsumer = TestConsumer

    def test_registry_initialization(self):
        """测试注册表初始化"""
        self.assertIsNotNone(self.registry)
        self.assertEqual(self.registry.get_consumer_count(), 0)
        self.assertIsNotNone(self.registry.logger)
        self.assertIsNotNone(self.registry.error_handler)

    def test_register_consumer_success(self):
        """测试成功注册消费者"""
        consumer = self.TestConsumer("test_consumer_1")
        
        result = self.registry.register_consumer(consumer)
        self.assertTrue(result)
        self.assertEqual(self.registry.get_consumer_count(), 1)

    def test_register_duplicate_consumer(self):
        """测试注册重复消费者"""
        consumer = self.TestConsumer("test_consumer_2")
        
        # 第一次注册应该成功
        result1 = self.registry.register_consumer(consumer)
        self.assertTrue(result1)
        
        # 第二次注册相同实例应该失败（返回False而不是异常）
        result2 = self.registry.register_consumer(consumer)
        self.assertFalse(result2)

    def test_unregister_consumer(self):
        """测试注销消费者"""
        consumer = self.TestConsumer("test_consumer_3")
        
        # 注册消费者
        self.registry.register_consumer(consumer)
        consumer_ids = self.registry.get_consumer_ids()
        self.assertEqual(len(consumer_ids), 1)
        
        # 注销消费者
        result = self.registry.unregister_consumer(consumer_ids[0])
        self.assertTrue(result)
        self.assertEqual(self.registry.get_consumer_count(), 0)

    def test_unregister_nonexistent_consumer(self):
        """测试注销不存在的消费者"""
        result = self.registry.unregister_consumer("nonexistent_id")
        self.assertFalse(result)

    def test_get_consumer(self):
        """测试获取消费者"""
        consumer = self.TestConsumer("test_consumer_4")
        
        self.registry.register_consumer(consumer)
        consumer_ids = self.registry.get_consumer_ids()
        
        retrieved_consumer = self.registry.get_consumer(consumer_ids[0])
        self.assertIsNotNone(retrieved_consumer)
        self.assertEqual(retrieved_consumer, consumer)

    def test_get_nonexistent_consumer(self):
        """测试获取不存在的消费者"""
        consumer = self.registry.get_consumer("nonexistent_id")
        self.assertIsNone(consumer)

    def test_get_consumers(self):
        """测试获取所有消费者"""
        consumer1 = self.TestConsumer("consumer_1")
        consumer2 = self.TestConsumer("consumer_2")
        
        self.registry.register_consumer(consumer1)
        self.registry.register_consumer(consumer2)
        
        consumers = self.registry.get_consumers()
        self.assertEqual(len(consumers), 2)
        self.assertIn(consumer1, consumers)
        self.assertIn(consumer2, consumers)

    def test_get_consumer_ids(self):
        """测试获取消费者ID列表"""
        consumer = self.TestConsumer("test_consumer_5")
        
        self.registry.register_consumer(consumer)
        consumer_ids = self.registry.get_consumer_ids()
        
        self.assertEqual(len(consumer_ids), 1)
        self.assertTrue(consumer_ids[0].startswith("TestConsumer_"))

    def test_has_consumer(self):
        """测试检查消费者是否存在"""
        consumer = self.TestConsumer("test_consumer_6")
        
        # 注册前应该不存在
        self.registry.register_consumer(consumer)
        consumer_ids = self.registry.get_consumer_ids()
        
        self.assertTrue(self.registry.has_consumer(consumer_ids[0]))
        self.assertFalse(self.registry.has_consumer("nonexistent_id"))

    def test_get_consumer_info_success(self):
        """测试获取消费者信息（成功情况）"""
        consumer = self.TestConsumer("test_consumer_7")
        
        self.registry.register_consumer(consumer)
        consumer_ids = self.registry.get_consumer_ids()
        
        info = self.registry.get_consumer_info(consumer_ids[0])
        self.assertIsNotNone(info)
        self.assertIn("consumer_id", info)
        self.assertIn("consumer_type", info)
        self.assertIn("consumed_resources_count", info)
        self.assertIn("resource_usage", info)
        self.assertEqual(info["status"], "active")

    def test_get_consumer_info_nonexistent(self):
        """测试获取不存在消费者的信息"""
        info = self.registry.get_consumer_info("nonexistent_id")
        self.assertIsNone(info)

    def test_get_consumer_info_with_exception(self):
        """测试获取消费者信息时发生异常"""
        # 创建会抛出异常的消费者
        consumer = self.TestConsumer("failing_consumer")
        
        # 模拟 get_consumed_resources 抛出异常
        with unittest.mock.patch.object(consumer, 'get_consumed_resources', side_effect=Exception("Test error")):
            self.registry.register_consumer(consumer)
            consumer_ids = self.registry.get_consumer_ids()
            
            info = self.registry.get_consumer_info(consumer_ids[0])
            self.assertIsNotNone(info)
            self.assertEqual(info["status"], "error")
            self.assertIn("error", info)

    def test_get_all_consumer_info(self):
        """测试获取所有消费者信息"""
        consumer1 = self.TestConsumer("consumer_info_1")
        consumer2 = self.TestConsumer("consumer_info_2")
        
        self.registry.register_consumer(consumer1)
        self.registry.register_consumer(consumer2)
        
        all_info = self.registry.get_all_consumer_info()
        self.assertEqual(len(all_info), 2)
        
        # 验证每个消费者都有信息
        for consumer_id, info in all_info.items():
            self.assertIn("consumer_id", info)
            self.assertIn("consumer_type", info)

    def test_clear_consumers(self):
        """测试清空所有消费者"""
        consumer1 = self.TestConsumer("clear_consumer_1")
        consumer2 = self.TestConsumer("clear_consumer_2")
        
        self.registry.register_consumer(consumer1)
        self.registry.register_consumer(consumer2)
        self.assertEqual(self.registry.get_consumer_count(), 2)
        
        self.registry.clear()
        self.assertEqual(self.registry.get_consumer_count(), 0)

    def test_registry_thread_safety(self):
        """测试注册表的线程安全性"""
        import threading
        import time
        
        def register_consumers():
            for i in range(10):
                consumer = self.TestConsumer(f"thread_consumer_{i}")
                self.registry.register_consumer(consumer)
                time.sleep(0.001)  # 模拟并发
        
        # 创建多个线程同时注册消费者
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=register_consumers)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证最终状态
        self.assertEqual(self.registry.get_consumer_count(), 30)

    def test_consumer_id_generation(self):
        """测试消费者ID生成规则"""
        consumer1 = self.TestConsumer("id_test_consumer")
        consumer2 = self.TestConsumer("id_test_consumer")
        
        self.registry.register_consumer(consumer1)
        self.registry.register_consumer(consumer2)
        
        consumer_ids = self.registry.get_consumer_ids()
        self.assertEqual(len(consumer_ids), 2)
        
        # 验证ID格式
        for consumer_id in consumer_ids:
            self.assertTrue(consumer_id.startswith("TestConsumer_"))
            # ID应该包含对象的id，所以应该不同
            self.assertTrue("_" in consumer_id)

    def test_error_handling_in_register(self):
        """测试注册过程中的错误处理"""
        # 测试传递None作为消费者
        result = self.registry.register_consumer(None)
        self.assertFalse(result)

    def test_error_handling_in_get_info(self):
        """测试获取信息过程中的错误处理"""
        # 创建一个会抛出异常的消费者
        class FailingConsumer(IResourceConsumer):
            def request_resource(self, resource_type: str, requirements: Dict[str, Any], 
                               priority: int = 1, timeout: float = None) -> str:
                return "test_id"
            
            def release_resource(self, allocation_id: str) -> bool:
                return True
            
            def get_consumed_resources(self) -> List[ResourceAllocation]:
                raise Exception("获取消费资源失败")
            
            def get_resource_usage(self) -> Dict[str, Any]:
                return {"error": "usage"}
        
        failing_consumer = FailingConsumer()
        self.registry.register_consumer(failing_consumer)
        consumer_ids = self.registry.get_consumer_ids()
        
        # 应该处理异常并返回错误信息
        info = self.registry.get_consumer_info(consumer_ids[0])
        self.assertIsNotNone(info)
        self.assertIn("error", info)


if __name__ == '__main__':
    unittest.main()
