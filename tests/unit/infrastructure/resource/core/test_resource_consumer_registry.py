"""
测试资源消费者注册表

覆盖 resource_consumer_registry.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.resource.core.resource_consumer_registry import ResourceConsumerRegistry
from src.infrastructure.resource.core.unified_resource_interfaces import IResourceConsumer


class MockResourceConsumer(IResourceConsumer):
    """模拟资源消费者"""

    def __init__(self, consumer_id="test_consumer"):
        self.consumer_id = consumer_id
        self.resource_requirements = {"cpu": 2, "memory": 4}

    def request_resource(self, resource_type: str, requirements, priority: int = 1, timeout=None):
        """请求资源"""
        return f"alloc_{self.consumer_id}_{resource_type}"

    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""
        return True

    def get_consumed_resources(self):
        """获取已消费的资源"""
        return []

    def get_resource_usage(self):
        """获取资源使用情况"""
        return {"cpu_percent": 50, "memory_percent": 60}

    def get_resource_requirements(self):
        return self.resource_requirements

    def allocate_resources(self, allocation):
        pass

    def release_resources(self, allocation_id):
        pass

    def get_allocated_resources(self):
        return []

    def get_consumer_status(self):
        return {"status": "active", "allocated_resources": []}


class TestResourceConsumerRegistry:
    """ResourceConsumerRegistry 类测试"""

    def test_initialization(self):
        """测试初始化"""
        registry = ResourceConsumerRegistry()

        assert hasattr(registry, 'logger')
        assert hasattr(registry, 'error_handler')
        assert hasattr(registry, '_consumers')
        assert hasattr(registry, '_lock')
        assert isinstance(registry._consumers, dict)

    def test_initialization_with_components(self):
        """测试带组件初始化"""
        mock_logger = Mock()
        mock_error_handler = Mock()

        registry = ResourceConsumerRegistry(
            logger=mock_logger,
            error_handler=mock_error_handler
        )

        assert registry.logger == mock_logger
        assert registry.error_handler == mock_error_handler

    def test_register_consumer_success(self):
        """测试成功注册消费者"""
        registry = ResourceConsumerRegistry()
        consumer = MockResourceConsumer("consumer_1")

        result = registry.register_consumer(consumer)

        assert result == True
        assert len(registry._consumers) == 1

    def test_register_consumer_none(self):
        """测试注册None消费者"""
        registry = ResourceConsumerRegistry()

        result = registry.register_consumer(None)

        assert result == False
        assert len(registry._consumers) == 0

    def test_register_consumer_duplicate(self):
        """测试注册重复消费者"""
        registry = ResourceConsumerRegistry()
        consumer1 = MockResourceConsumer("consumer_1")

        # 第一次注册成功
        result1 = registry.register_consumer(consumer1)
        assert result1 == True

        # 第二次注册同一个消费者应该失败
        result2 = registry.register_consumer(consumer1)
        assert result2 == False

        # 仍然只有一个消费者
        assert len(registry._consumers) == 1

    def test_unregister_consumer_success(self):
        """测试成功取消注册消费者"""
        registry = ResourceConsumerRegistry()
        consumer = MockResourceConsumer("consumer_1")

        # 先注册
        registry.register_consumer(consumer)
        assert len(registry._consumers) == 1

        # 取消注册
        result = registry.unregister_consumer(list(registry._consumers.keys())[0])

        assert result == True
        assert len(registry._consumers) == 0

    def test_unregister_consumer_not_found(self):
        """测试取消注册不存在的消费者"""
        registry = ResourceConsumerRegistry()

        result = registry.unregister_consumer("nonexistent_id")

        assert result == False

    def test_get_consumer_success(self):
        """测试成功获取消费者"""
        registry = ResourceConsumerRegistry()
        consumer = MockResourceConsumer("consumer_1")

        registry.register_consumer(consumer)

        retrieved = registry.get_consumer(list(registry._consumers.keys())[0])

        assert retrieved is not None
        assert retrieved.consumer_id == "consumer_1"

    def test_get_consumer_not_found(self):
        """测试获取不存在的消费者"""
        registry = ResourceConsumerRegistry()

        result = registry.get_consumer("nonexistent_id")

        assert result is None

    def test_get_consumers(self):
        """测试获取所有消费者"""
        registry = ResourceConsumerRegistry()

        consumer1 = MockResourceConsumer("consumer_1")
        consumer2 = MockResourceConsumer("consumer_2")

        registry.register_consumer(consumer1)
        registry.register_consumer(consumer2)

        consumers = registry.get_consumers()

        assert len(consumers) == 2
        consumer_ids = [c.consumer_id for c in consumers]
        assert "consumer_1" in consumer_ids
        assert "consumer_2" in consumer_ids

    def test_get_consumer_ids(self):
        """测试获取消费者ID列表"""
        registry = ResourceConsumerRegistry()

        consumer1 = MockResourceConsumer("consumer_1")
        consumer2 = MockResourceConsumer("consumer_2")

        registry.register_consumer(consumer1)
        registry.register_consumer(consumer2)

        ids = registry.get_consumer_ids()

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

    def test_has_consumer(self):
        """测试检查消费者是否存在"""
        registry = ResourceConsumerRegistry()
        consumer = MockResourceConsumer("consumer_1")

        registry.register_consumer(consumer)

        consumer_id = list(registry._consumers.keys())[0]

        assert registry.has_consumer(consumer_id) == True
        assert registry.has_consumer("nonexistent") == False

    def test_get_consumer_count(self):
        """测试获取消费者数量"""
        registry = ResourceConsumerRegistry()

        assert registry.get_consumer_count() == 0

        consumer1 = MockResourceConsumer("consumer_1")
        consumer2 = MockResourceConsumer("consumer_2")

        registry.register_consumer(consumer1)
        registry.register_consumer(consumer2)

        assert registry.get_consumer_count() == 2

    def test_get_consumer_info(self):
        """测试获取消费者信息"""
        registry = ResourceConsumerRegistry()
        consumer = MockResourceConsumer("consumer_1")

        registry.register_consumer(consumer)

        consumer_id = list(registry._consumers.keys())[0]
        info = registry.get_consumer_info(consumer_id)

        assert info is not None
        assert "consumer_id" in info
        assert "resource_requirements" in info
        assert "status" in info

    def test_get_consumer_info_not_found(self):
        """测试获取不存在的消费者信息"""
        registry = ResourceConsumerRegistry()

        info = registry.get_consumer_info("nonexistent")

        assert info is None

    def test_get_all_consumer_info(self):
        """测试获取所有消费者信息"""
        registry = ResourceConsumerRegistry()

        consumer1 = MockResourceConsumer("consumer_1")
        consumer2 = MockResourceConsumer("consumer_2")

        registry.register_consumer(consumer1)
        registry.register_consumer(consumer2)

        all_info = registry.get_all_consumer_info()

        assert isinstance(all_info, dict)
        assert len(all_info) == 2

        # 检查每个消费者的信息
        for consumer_id, info in all_info.items():
            assert "consumer_id" in info
            assert "resource_requirements" in info
            assert "status" in info

    def test_clear(self):
        """测试清空注册表"""
        registry = ResourceConsumerRegistry()

        consumer1 = MockResourceConsumer("consumer_1")
        consumer2 = MockResourceConsumer("consumer_2")

        registry.register_consumer(consumer1)
        registry.register_consumer(consumer2)

        assert registry.get_consumer_count() == 2

        registry.clear()

        assert registry.get_consumer_count() == 0
        assert len(registry._consumers) == 0

    def test_thread_safety(self):
        """测试线程安全性"""
        registry = ResourceConsumerRegistry()
        import threading
        import time

        results = []
        errors = []

        def register_consumers(thread_id):
            try:
                for i in range(10):
                    consumer = MockResourceConsumer(f"consumer_{thread_id}_{i}")
                    result = registry.register_consumer(consumer)
                    results.append((thread_id, i, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程同时注册消费者
        threads = []
        for i in range(3):
            thread = threading.Thread(target=register_consumers, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有注册都成功
        assert len(results) == 30  # 3个线程 * 10个消费者
        assert all(result[2] for result in results)  # 所有注册都返回True

        # 最终应该有30个消费者
        assert registry.get_consumer_count() == 30

    def test_error_handling_in_registration(self):
        """测试注册过程中的错误处理"""
        registry = ResourceConsumerRegistry()

        # 创建一个有问题的消费者（模拟）
        class ProblematicConsumer(MockResourceConsumer):
            def get_resource_requirements(self):
                raise ValueError("Invalid requirements")

        problematic_consumer = ProblematicConsumer("problematic")

        # 注册应该成功（错误在获取信息时处理）
        result = registry.register_consumer(problematic_consumer)

        assert result == True
        assert registry.get_consumer_count() == 1

        # 但获取信息时应该能处理错误
        consumer_id = list(registry._consumers.keys())[0]
        info = registry.get_consumer_info(consumer_id)

        # 即使有错误，也应该返回基本信息
        assert info is not None
        assert "consumer_id" in info

    def test_consumer_info_with_different_types(self):
        """测试不同类型消费者的信息获取"""
        registry = ResourceConsumerRegistry()

        # 注册不同类型的消费者
        web_consumer = MockResourceConsumer("web_app")
        web_consumer.resource_requirements = {"cpu": 2, "memory": 4, "network": 100}

        batch_consumer = MockResourceConsumer("batch_job")
        batch_consumer.resource_requirements = {"cpu": 8, "memory": 16, "storage": 100}

        registry.register_consumer(web_consumer)
        registry.register_consumer(batch_consumer)

        all_info = registry.get_all_consumer_info()

        assert len(all_info) == 2

        # 验证不同消费者的需求
        web_info = None
        batch_info = None

        for consumer_id, info in all_info.items():
            if "web_app" in consumer_id:
                web_info = info
            elif "batch_job" in consumer_id:
                batch_info = info

        assert web_info is not None
        assert batch_info is not None

        assert web_info["resource_requirements"]["cpu"] == 2
        assert batch_info["resource_requirements"]["cpu"] == 8

    def test_registry_with_event_bus_integration(self):
        """测试注册表与事件总线的集成"""
        from src.infrastructure.resource.core.event_bus import EventBus

        event_bus = EventBus()
        registry = ResourceConsumerRegistry()

        # 在实际实现中，注册表可能需要与事件总线集成
        # 这里我们只是测试基本的集成可能性

        assert registry is not None
        assert event_bus is not None

        # 验证可以同时创建和使用两个组件
        consumer = MockResourceConsumer("integrated_consumer")
        registry.register_consumer(consumer)

        assert registry.get_consumer_count() == 1

    def test_bulk_operations(self):
        """测试批量操作"""
        registry = ResourceConsumerRegistry()

        # 批量注册消费者
        consumers = []
        for i in range(50):
            consumer = MockResourceConsumer(f"bulk_consumer_{i}")
            consumers.append(consumer)
            registry.register_consumer(consumer)

        assert registry.get_consumer_count() == 50

        # 批量获取信息
        all_info = registry.get_all_consumer_info()
        assert len(all_info) == 50

        # 批量清除
        registry.clear()
        assert registry.get_consumer_count() == 0

    def test_consumer_lifecycle(self):
        """测试消费者生命周期"""
        registry = ResourceConsumerRegistry()

        # 1. 创建和注册消费者
        consumer = MockResourceConsumer("lifecycle_consumer")
        result = registry.register_consumer(consumer)
        assert result == True

        # 2. 验证注册成功
        assert registry.get_consumer_count() == 1
        consumer_id = list(registry._consumers.keys())[0]
        assert registry.has_consumer(consumer_id)

        # 3. 获取消费者信息
        info = registry.get_consumer_info(consumer_id)
        assert info is not None
        assert info["consumer_id"] == "lifecycle_consumer"

        # 4. 取消注册消费者
        result = registry.unregister_consumer(consumer_id)
        assert result == True

        # 5. 验证取消注册成功
        assert registry.get_consumer_count() == 0
        assert not registry.has_consumer(consumer_id)

    def test_register_consumer_with_event_bus(self):
        """测试带事件总线的消费者注册"""
        from src.infrastructure.resource.core.event_bus import EventBus
        mock_event_bus = Mock()
        registry = ResourceConsumerRegistry()

        # 注意：实际的ResourceConsumerRegistry没有event_bus参数，这里只是测试概念
        consumer = MockResourceConsumer("event_consumer")
        result = registry.register_consumer(consumer)

        assert result == True
        # 在实际实现中，这里可能需要发布事件

    def test_get_consumer_info_with_error_handling(self):
        """测试消费者信息获取的错误处理"""
        registry = ResourceConsumerRegistry()

        # 创建一个有问题的消费者
        class ProblematicConsumer(MockResourceConsumer):
            def get_resource_requirements(self):
                raise ValueError("Invalid requirements")

        problematic_consumer = ProblematicConsumer("problematic")
        registry.register_consumer(problematic_consumer)

        consumer_id = list(registry._consumers.keys())[0]

        # 获取信息时不应该抛出异常
        info = registry.get_consumer_info(consumer_id)

        assert info is not None
        assert "consumer_id" in info
        # 其他字段可能为空或有默认值

    def test_bulk_consumer_operations(self):
        """测试批量消费者操作"""
        registry = ResourceConsumerRegistry()

        # 批量注册消费者
        consumers = []
        for i in range(50):
            consumer = MockResourceConsumer(f"bulk_consumer_{i}")
            consumers.append(consumer)
            registry.register_consumer(consumer)

        assert registry.get_consumer_count() == 50

        # 批量获取信息
        all_info = registry.get_all_consumer_info()
        assert len(all_info) == 50

        # 批量取消注册
        consumer_ids = list(registry._consumers.keys())
        for consumer_id in consumer_ids[:25]:  # 取消注册前25个
            registry.unregister_consumer(consumer_id)

        assert registry.get_consumer_count() == 25

    def test_consumer_filtering_and_search(self):
        """测试消费者过滤和搜索功能"""
        registry = ResourceConsumerRegistry()

        # 注册不同类型的消费者
        web_consumer = MockResourceConsumer("web_app_1")
        web_consumer.resource_requirements = {"cpu": 2, "memory": 4}

        batch_consumer = MockResourceConsumer("batch_job_1")
        batch_consumer.resource_requirements = {"cpu": 8, "memory": 16}

        api_consumer = MockResourceConsumer("api_service_1")
        api_consumer.resource_requirements = {"cpu": 1, "memory": 2}

        registry.register_consumer(web_consumer)
        registry.register_consumer(batch_consumer)
        registry.register_consumer(api_consumer)

        # 测试按资源需求过滤（这里通过遍历来模拟）
        all_consumers = registry.get_consumers()

        high_cpu_consumers = [
            c for c in all_consumers
            if c.resource_requirements.get("cpu", 0) >= 4
        ]

        assert len(high_cpu_consumers) == 1
        assert high_cpu_consumers[0].consumer_id == "batch_job_1"

    def test_consumer_resource_utilization_analysis(self):
        """测试消费者资源利用率分析"""
        registry = ResourceConsumerRegistry()

        # 注册具有不同资源需求的消费者
        consumers_data = [
            ("web_1", {"cpu": 2, "memory": 4}),
            ("web_2", {"cpu": 2, "memory": 4}),
            ("batch_1", {"cpu": 8, "memory": 16}),
            ("api_1", {"cpu": 1, "memory": 2}),
            ("ml_1", {"cpu": 16, "memory": 32, "gpu": 1}),
        ]

        for consumer_id, requirements in consumers_data:
            consumer = MockResourceConsumer(consumer_id)
            consumer.resource_requirements = requirements
            registry.register_consumer(consumer)

        all_info = registry.get_all_consumer_info()

        # 计算总资源需求
        total_cpu = sum(info["resource_requirements"].get("cpu", 0)
                       for info in all_info.values())
        total_memory = sum(info["resource_requirements"].get("memory", 0)
                          for info in all_info.values())
        total_gpu = sum(info["resource_requirements"].get("gpu", 0)
                       for info in all_info.values())

        assert total_cpu == 2+2+8+1+16 == 29
        assert total_memory == 4+4+16+2+32 == 58
        assert total_gpu == 1

    def test_consumer_health_and_status_monitoring(self):
        """测试消费者健康和状态监控"""
        registry = ResourceConsumerRegistry()

        # 创建不同状态的消费者
        active_consumer = MockResourceConsumer("active_consumer")

        # 模拟不活跃的消费者（通过修改内部状态）
        inactive_consumer = MockResourceConsumer("inactive_consumer")

        registry.register_consumer(active_consumer)
        registry.register_consumer(inactive_consumer)

        # 获取所有消费者信息
        all_info = registry.get_all_consumer_info()

        # 验证所有消费者都有状态信息
        for consumer_id, info in all_info.items():
            assert "status" in info
            assert "resource_requirements" in info
            assert "consumer_id" in info

        # 验证消费者数量
        assert len(all_info) == 2

    def test_consumer_registration_validation(self):
        """测试消费者注册验证"""
        registry = ResourceConsumerRegistry()

        # 测试无效消费者
        invalid_consumers = [
            None,
            "not_an_object",
            123,
            [],
            {},
        ]

        for invalid_consumer in invalid_consumers:
            result = registry.register_consumer(invalid_consumer)
            assert result == False

        # 注册表应该仍然为空
        assert registry.get_consumer_count() == 0

    def test_consumer_unregistration_validation(self):
        """测试消费者取消注册验证"""
        registry = ResourceConsumerRegistry()

        # 尝试取消注册不存在的消费者
        result = registry.unregister_consumer("nonexistent")
        assert result == False

        # 尝试取消注册无效ID
        invalid_ids = [None, "", 123, [], {}]
        for invalid_id in invalid_ids:
            result = registry.unregister_consumer(invalid_id)
            assert result == False

    def test_consumer_information_consistency(self):
        """测试消费者信息一致性"""
        registry = ResourceConsumerRegistry()

        consumer = MockResourceConsumer("consistency_test")
        original_requirements = {"cpu": 4, "memory": 8}
        consumer.resource_requirements = original_requirements

        registry.register_consumer(consumer)

        # 获取消费者ID
        consumer_id = list(registry._consumers.keys())[0]

        # 多次获取信息应该一致
        info1 = registry.get_consumer_info(consumer_id)
        info2 = registry.get_consumer_info(consumer_id)

        assert info1 == info2
        assert info1["resource_requirements"] == original_requirements
        assert info2["resource_requirements"] == original_requirements

    def test_consumer_registry_memory_management(self):
        """测试消费者注册表内存管理"""
        import gc
        registry = ResourceConsumerRegistry()

        # 注册大量消费者
        initial_count = 100
        consumers = []
        for i in range(initial_count):
            consumer = MockResourceConsumer(f"memory_test_{i}")
            consumers.append(consumer)
            registry.register_consumer(consumer)

        assert registry.get_consumer_count() == initial_count

        # 清除所有消费者
        registry.clear()

        assert registry.get_consumer_count() == 0

        # 强制垃圾回收
        consumers.clear()
        gc.collect()

        # 注册表应该保持为空
        assert registry.get_consumer_count() == 0

    def test_consumer_registry_concurrent_access(self):
        """测试消费者注册表的并发访问"""
        import threading
        registry = ResourceConsumerRegistry()

        results = []
        errors = []

        def registry_worker(thread_id):
            try:
                # 每个线程注册自己的消费者
                consumer = MockResourceConsumer(f"concurrent_consumer_{thread_id}")
                result = registry.register_consumer(consumer)
                results.append((thread_id, result))

                # 查询操作
                count = registry.get_consumer_count()
                results.append((thread_id, "count", count))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        num_threads = 10

        for i in range(num_threads):
            thread = threading.Thread(target=registry_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有注册都成功
        successful_registrations = [r for r in results if isinstance(r, tuple) and len(r) == 2 and r[1] == True]
        assert len(successful_registrations) == num_threads

        # 最终应该有正确的消费者数量
        assert registry.get_consumer_count() == num_threads

    def test_consumer_registry_with_complex_consumer_types(self):
        """测试注册表与复杂消费者类型"""
        registry = ResourceConsumerRegistry()

        # 创建不同类型的消费者
        class WebServiceConsumer(MockResourceConsumer):
            def __init__(self, service_name):
                super().__init__(f"web_{service_name}")
                self.service_type = "web"

        class DatabaseConsumer(MockResourceConsumer):
            def __init__(self, db_name):
                super().__init__(f"db_{db_name}")
                self.service_type = "database"

        class QueueConsumer(MockResourceConsumer):
            def __init__(self, queue_name):
                super().__init__(f"queue_{queue_name}")
                self.service_type = "queue"

        # 注册不同类型的消费者
        consumers = [
            WebServiceConsumer("api"),
            WebServiceConsumer("admin"),
            DatabaseConsumer("main"),
            DatabaseConsumer("cache"),
            QueueConsumer("tasks"),
            QueueConsumer("notifications"),
        ]

        for consumer in consumers:
            result = registry.register_consumer(consumer)
            assert result == True

        assert registry.get_consumer_count() == 6

        # 按类型分组统计
        all_info = registry.get_all_consumer_info()

        web_services = [info for info in all_info.values()
                       if info["consumer_id"].startswith("web_")]
        db_services = [info for info in all_info.values()
                      if info["consumer_id"].startswith("db_")]
        queue_services = [info for info in all_info.values()
                         if info["consumer_id"].startswith("queue_")]

        assert len(web_services) == 2
        assert len(db_services) == 2
        assert len(queue_services) == 2

    def test_registry_thread_safety(self):
        """测试注册表的线程安全性"""
        import threading
        import time

        registry = ResourceConsumerRegistry()
        errors = []

        def concurrent_operation(thread_id):
            try:
                consumer = MockResourceConsumer(f"consumer_{thread_id}")
                registry.register_consumer(consumer)
                time.sleep(0.01)  # 模拟一些操作时间
                retrieved = registry.get_consumer(f"consumer_{thread_id}")
                if retrieved != consumer:
                    errors.append(f"Thread {thread_id}: consumer mismatch")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        num_threads = 10

        # 启动并发线程
        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0
        assert registry.get_consumer_count() == num_threads

    def test_registry_bulk_operations(self):
        """测试注册表的批量操作"""
        registry = ResourceConsumerRegistry()

        # 批量注册消费者
        consumers = []
        for i in range(100):
            consumer = MockResourceConsumer(f"bulk_consumer_{i}")
            consumers.append(consumer)
            registry.register_consumer(consumer)

        assert registry.get_consumer_count() == 100

        # 批量获取消费者
        all_consumers = registry.get_consumers()
        assert len(all_consumers) == 100

        # 批量注销消费者
        for consumer in consumers[:50]:  # 注销一半
            registry.unregister_consumer(consumer.consumer_id)

        assert registry.get_consumer_count() == 50

    def test_registry_error_handling(self):
        """测试注册表的错误处理"""
        registry = ResourceConsumerRegistry()

        # 测试注册None消费者
        result = registry.register_consumer(None)
        assert result == False

        # 测试重复注册
        consumer = MockResourceConsumer("duplicate_consumer")
        registry.register_consumer(consumer)
        result = registry.register_consumer(consumer)
        assert result == False

        # 测试注销不存在的消费者
        result = registry.unregister_consumer("nonexistent")
        assert result == False

        # 测试获取不存在的消费者
        result = registry.get_consumer("nonexistent")
        assert result is None

    def test_registry_performance(self):
        """测试注册表的性能"""
        import time
        registry = ResourceConsumerRegistry()

        # 测试大量消费者的注册性能
        start_time = time.time()
        for i in range(1000):
            consumer = MockResourceConsumer(f"perf_consumer_{i}")
            registry.register_consumer(consumer)
        end_time = time.time()

        assert registry.get_consumer_count() == 1000
        assert end_time - start_time < 5.0  # 5秒内完成

        # 测试查询性能
        start_time = time.time()
        for i in range(100):
            consumer = registry.get_consumer(f"perf_consumer_{i}")
            assert consumer is not None
        end_time = time.time()

        assert end_time - start_time < 1.0  # 1秒内完成100次查询
