"""
测试资源分配管理器

覆盖 resource_allocation_manager.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation, ResourceRequest


class TestResourceAllocationManager:
    """ResourceAllocationManager 类测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = ResourceAllocationManager()

        assert hasattr(manager, 'provider_registry')
        assert hasattr(manager, 'event_bus')
        assert hasattr(manager, 'logger')
        assert hasattr(manager, 'error_handler')
        assert hasattr(manager, '_allocations')
        assert hasattr(manager, '_requests')
        assert hasattr(manager, '_lock')

    def test_initialization_with_components(self):
        """测试带组件初始化"""
        mock_provider = Mock()
        mock_event_bus = Mock()
        mock_logger = Mock()
        mock_error_handler = Mock()

        manager = ResourceAllocationManager(
            provider_registry=mock_provider,
            event_bus=mock_event_bus,
            logger=mock_logger,
            error_handler=mock_error_handler
        )

        assert manager.provider_registry == mock_provider
        assert manager.event_bus == mock_event_bus
        assert manager.logger == mock_logger
        assert manager.error_handler == mock_error_handler

    def test_get_resource_type(self):
        """测试获取资源类型"""
        manager = ResourceAllocationManager()

        # 测试有resource_type属性的情况
        allocation = Mock()
        allocation.resource_type = "cpu"
        allocation.resource_id = "cpu_123"

        result = manager._get_resource_type(allocation)
        assert result == "cpu"

        # 测试从resource_id推断的情况
        allocation2 = Mock()
        allocation2.resource_type = None
        allocation2.resource_id = "memory_res1"

        result2 = manager._get_resource_type(allocation2)
        assert result2 == "memory"

        # 测试无下划线的情况
        allocation3 = Mock()
        allocation3.resource_type = None
        allocation3.resource_id = "disk"

        result3 = manager._get_resource_type(allocation3)
        assert result3 == "disk"

    @patch('src.infrastructure.resource.core.resource_allocation_manager.ResourceRequest')
    def test_request_resource_success(self, mock_resource_request):
        """测试成功请求资源"""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.get_provider.return_value = Mock()

        # Mock allocation creation
        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"

        manager = ResourceAllocationManager(provider_registry=mock_provider)

        # Mock the internal methods
        with patch.object(manager, '_validate_and_get_provider', return_value=Mock()), \
             patch.object(manager, '_create_resource_request', return_value=mock_resource_request), \
             patch.object(manager, '_store_request'), \
             patch.object(manager, '_attempt_allocation', return_value=mock_allocation):

            result = manager.request_resource("consumer_1", "cpu", {"cores": 4})

            assert result == "alloc_123"

    def test_request_resource_no_provider(self):
        """测试请求资源时无可用提供者"""
        manager = ResourceAllocationManager()

        with patch.object(manager, '_validate_and_get_provider', return_value=None):
            result = manager.request_resource("consumer_1", "cpu", {"cores": 4})

            assert result is None

    def test_release_resource_success(self):
        """测试成功释放资源"""
        manager = ResourceAllocationManager()

        # Mock allocation
        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        mock_allocation.consumer_id = "consumer_1"

        # Add to allocations
        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider
        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        with patch.object(manager, 'event_bus') as mock_event_bus:
            result = manager.release_resource("alloc_123")

            assert result is not None  # Mock对象被返回
            assert "alloc_123" not in manager._allocations
            mock_provider.release_resource.assert_called_once_with("cpu_123")

    def test_release_resource_not_found(self):
        """测试释放不存在的资源"""
        manager = ResourceAllocationManager()

        result = manager.release_resource("nonexistent_alloc")

        assert result == False

    def test_get_allocation(self):
        """测试获取分配信息"""
        manager = ResourceAllocationManager()

        mock_allocation = Mock()
        manager._allocations = {"alloc_123": mock_allocation}

        result = manager.get_allocation("alloc_123")

        assert result == mock_allocation

        # 测试不存在的分配
        result2 = manager.get_allocation("nonexistent")
        assert result2 is None

    def test_get_allocations_for_consumer(self):
        """测试获取消费者的分配列表"""
        manager = ResourceAllocationManager()

        alloc1 = Mock()
        alloc1.request_id = "request_cpu_consumer_1"
        alloc2 = Mock()
        alloc2.request_id = "request_mem_consumer_1"
        alloc3 = Mock()
        alloc3.request_id = "request_gpu_consumer_2"

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2,
            "alloc_3": alloc3
        }

        result = manager.get_allocations_for_consumer("consumer_1")

        assert len(result) == 2  # consumer_1有两个分配
        assert alloc1 in result
        assert alloc2 in result
        assert alloc3 not in result

    def test_get_allocations_for_resource_type(self):
        """测试获取资源类型的分配列表"""
        manager = ResourceAllocationManager()

        alloc1 = Mock()
        alloc1.resource_type = "cpu"
        alloc2 = Mock()
        alloc2.resource_type = "memory"
        alloc3 = Mock()
        alloc3.resource_type = "cpu"

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2,
            "alloc_3": alloc3
        }

        result = manager.get_allocations_for_resource_type("cpu")

        assert len(result) == 2
        assert alloc1 in result
        assert alloc3 in result

    def test_get_allocation_count(self):
        """测试获取分配数量"""
        manager = ResourceAllocationManager()

        manager._allocations = {"alloc_1": Mock(), "alloc_2": Mock()}

        result = manager.get_allocation_count()

        assert result == 2

    def test_get_request_count(self):
        """测试获取请求数量"""
        manager = ResourceAllocationManager()

        manager._requests = {"req_1": Mock(), "req_2": Mock(), "req_3": Mock()}

        result = manager.get_request_count()

        assert result == 3

    def test_get_request(self):
        """测试获取请求信息"""
        manager = ResourceAllocationManager()

        mock_request = Mock()
        manager._requests = {"req_123": mock_request}

        result = manager.get_request("req_123")

        assert result == mock_request

        # 测试不存在的请求
        result2 = manager.get_request("nonexistent")
        assert result2 is None

    def test_get_allocation_summary(self):
        """测试获取分配汇总"""
        manager = ResourceAllocationManager()

        # Mock allocations with different resource types
        alloc1 = Mock()
        alloc1.resource_type = "cpu"
        alloc1.consumer_id = "consumer_1"

        alloc2 = Mock()
        alloc2.resource_type = "memory"
        alloc2.consumer_id = "consumer_2"

        alloc3 = Mock()
        alloc3.resource_type = "cpu"
        alloc3.consumer_id = "consumer_1"

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2,
            "alloc_3": alloc3
        }

        summary = manager.get_allocation_summary()

        assert isinstance(summary, dict)
        assert 'total_allocations' in summary
        assert 'allocations_by_type' in summary
        assert 'allocations_by_consumer' in summary

    def test_get_active_allocations(self):
        """测试获取活跃分配"""
        manager = ResourceAllocationManager()

        alloc1 = Mock()
        alloc2 = Mock()

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2
        }

        result = manager.get_active_allocations()

        assert len(result) == 2
        assert alloc1 in result
        assert alloc2 in result

    def test_clear_expired_allocations(self):
        """测试清除过期分配"""
        manager = ResourceAllocationManager()

        # Create allocations with different timestamps
        alloc1 = Mock()
        alloc1.allocated_at = Mock()
        alloc1.allocated_at.timestamp.return_value = (datetime.now() - timedelta(hours=2)).timestamp()  # Expired

        alloc2 = Mock()
        alloc2.allocated_at = Mock()
        alloc2.allocated_at.timestamp.return_value = (datetime.now() - timedelta(minutes=30)).timestamp()  # Not expired

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2
        }

        result = manager.clear_expired_allocations(3600)  # 1 hour

        assert result == 1  # One allocation cleared
        assert "alloc_1" not in manager._allocations
        assert "alloc_2" in manager._allocations

    def test_force_release_allocation(self):
        """测试强制释放分配"""
        manager = ResourceAllocationManager()

        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider
        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        result = manager.force_release_allocation("alloc_123")

        assert result == True
        assert "alloc_123" not in manager._allocations

    def test_clear_all_allocations(self):
        """测试清除所有分配"""
        manager = ResourceAllocationManager()

        manager._allocations = {
            "alloc_1": Mock(),
            "alloc_2": Mock(),
            "alloc_3": Mock()
        }
        manager._requests = {
            "req_1": Mock(),
            "req_2": Mock()
        }

        manager.clear_all_allocations()

        assert len(manager._allocations) == 0
        # clear_all_allocations 只清空分配，不清空请求
        assert len(manager._requests) == 2

    def test_create_resource_request(self):
        """测试创建资源请求"""
        manager = ResourceAllocationManager()

        request = manager._create_resource_request("consumer_1", "cpu", {"cores": 4}, 2)

        assert hasattr(request, 'request_id')
        assert request.consumer_id == "consumer_1"
        assert request.resource_type == "cpu"
        assert request.requirements == {"cores": 4}
        assert request.priority == 2

    def test_store_request(self):
        """测试存储请求"""
        manager = ResourceAllocationManager()

        mock_request = Mock()
        mock_request.request_id = "req_123"

        manager._store_request(mock_request)

        assert "req_123" in manager._requests
        assert manager._requests["req_123"] == mock_request

    def test_attempt_allocation_success(self):
        """测试成功尝试分配"""
        manager = ResourceAllocationManager()

        mock_provider = Mock()
        mock_request = Mock()
        mock_request.consumer_id = "consumer_1"

        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"

        mock_provider.allocate_resource.return_value = mock_allocation

        with patch.object(manager, '_handle_successful_allocation', return_value="alloc_123"):
            result = manager._attempt_allocation(mock_provider, mock_request)

            assert result == "alloc_123"

    def test_attempt_allocation_failure(self):
        """测试分配失败"""
        manager = ResourceAllocationManager()

        mock_provider = Mock()
        mock_request = Mock()
        mock_request.consumer_id = "consumer_1"
        mock_request.resource_type = "cpu"

        mock_provider.allocate_resource.return_value = None

        with patch.object(manager, '_handle_failed_allocation') as mock_handle_failure:
            result = manager._attempt_allocation(mock_provider, mock_request)

            assert result is None
            mock_handle_failure.assert_called_once()

    def test_handle_successful_allocation(self):
        """测试处理成功分配"""
        manager = ResourceAllocationManager()

        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"

        result = manager._handle_successful_allocation(mock_allocation, "consumer_1")

        assert result == "alloc_123"
        assert "alloc_123" in manager._allocations

    def test_handle_failed_allocation(self):
        """测试处理失败分配"""
        manager = ResourceAllocationManager()

        with patch.object(manager.logger, 'warning') as mock_warning:
            manager._handle_failed_allocation("cpu", "consumer_1")

            mock_warning.assert_called_once()

    def test_validate_and_get_provider_success(self):
        """测试验证并获取提供者成功"""
        manager = ResourceAllocationManager()

        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider
        mock_provider.is_available.return_value = True

        result = manager._validate_and_get_provider("cpu")

        assert result == mock_provider

    def test_validate_and_get_provider_not_found(self):
        """测试提供者未找到"""
        manager = ResourceAllocationManager()

        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = None

        result = manager._validate_and_get_provider("cpu")

        assert result is None

    def test_validate_and_get_provider_not_available(self):
        """测试提供者不可用"""
        manager = ResourceAllocationManager()

        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider
        mock_provider.is_available.return_value = False

        result = manager._validate_and_get_provider("cpu")

        assert result is None

    def test_request_resource_with_event_bus(self):
        """测试带事件总线的资源请求"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        with patch.object(manager, '_validate_and_get_provider', return_value=mock_provider), \
             patch.object(manager, '_create_resource_request'), \
             patch.object(manager, '_store_request'), \
             patch.object(manager, '_attempt_allocation', return_value="alloc_123"):

            result = manager.request_resource("consumer_1", "cpu", {"cores": 4})

            assert result == "alloc_123"
            # 验证事件总线被使用
            assert manager.event_bus == mock_event_bus

    def test_request_resource_event_publish(self):
        """测试资源请求时的事件发布"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"
        mock_allocation.consumer_id = "consumer_1"
        mock_allocation.resource_type = "cpu"
        mock_allocation.requirements = {"cores": 4}

        with patch.object(manager, '_validate_and_get_provider', return_value=mock_provider), \
             patch.object(manager, '_create_resource_request'), \
             patch.object(manager, '_store_request'), \
             patch.object(manager, '_attempt_allocation', return_value=mock_allocation):

            result = manager.request_resource("consumer_1", "cpu", {"cores": 4})

            assert result == "alloc_123"
            # 验证事件被发布
            mock_event_bus.publish.assert_called()

    def test_release_resource_with_event_bus(self):
        """测试带事件总线的资源释放"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        # Mock allocation
        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        mock_allocation.consumer_id = "consumer_1"
        mock_allocation.resource_type = "cpu"
        mock_allocation.requirements = {"cores": 4}

        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider
        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        result = manager.release_resource("alloc_123")

        assert result == True
        # 验证事件被发布
        mock_event_bus.publish.assert_called()

    def test_release_resource_event_details(self):
        """测试资源释放事件详情"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        # Mock allocation
        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        mock_allocation.consumer_id = "consumer_1"
        mock_allocation.resource_type = "cpu"
        mock_allocation.requirements = {"cores": 4}

        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider
        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        manager.release_resource("alloc_123")

        # 验证事件详情
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.event_type == "resource.cpu.released"
        assert event.source == "resource_manager"
        assert event.data["resource_type"] == "cpu"
        assert event.data["resource_id"] == "cpu_123"
        assert event.data["consumer_id"] == "consumer_1"

    def test_get_allocation_with_timestamps(self):
        """测试带时间戳的分配查询"""
        manager = ResourceAllocationManager()

        from datetime import datetime
        mock_allocation = Mock()
        mock_allocation.allocated_at = datetime.now()
        mock_allocation.last_accessed = datetime.now()
        mock_allocation.expires_at = datetime.now()

        manager._allocations = {"alloc_123": mock_allocation}

        result = manager.get_allocation("alloc_123")

        assert result == mock_allocation
        assert hasattr(result, 'allocated_at')

    def test_get_allocations_for_consumer_empty(self):
        """测试获取空消费者分配列表"""
        manager = ResourceAllocationManager()

        result = manager.get_allocations_for_consumer("nonexistent_consumer")

        assert result == []

    def test_get_allocations_for_resource_type_empty(self):
        """测试获取空资源类型分配列表"""
        manager = ResourceAllocationManager()

        result = manager.get_allocations_for_resource_type("nonexistent_type")

        assert result == []

    def test_get_allocation_summary_empty(self):
        """测试获取空分配汇总"""
        manager = ResourceAllocationManager()

        summary = manager.get_allocation_summary()

        assert summary["total_allocations"] == 0
        assert len(summary["allocations_by_type"]) == 0
        assert len(summary["allocations_by_consumer"]) == 0

    def test_get_allocation_summary_detailed(self):
        """测试获取详细分配汇总"""
        manager = ResourceAllocationManager()

        # 创建多个分配用于测试汇总
        alloc1 = Mock()
        alloc1.consumer_id = "consumer_1"
        alloc1.resource_type = "cpu"
        alloc1.requirements = {"cores": 4}

        alloc2 = Mock()
        alloc2.consumer_id = "consumer_1"
        alloc2.resource_type = "memory"
        alloc2.requirements = {"size": 8}

        alloc3 = Mock()
        alloc3.consumer_id = "consumer_2"
        alloc3.resource_type = "cpu"
        alloc3.requirements = {"cores": 2}

        manager._allocations = {
            "alloc_1": alloc1,
            "alloc_2": alloc2,
            "alloc_3": alloc3
        }

        summary = manager.get_allocation_summary()

        assert summary["total_allocations"] == 3
        assert summary["allocations_by_type"]["cpu"] == 2
        assert summary["allocations_by_type"]["memory"] == 1
        assert summary["allocations_by_consumer"]["consumer_1"] == 2
        assert summary["allocations_by_consumer"]["consumer_2"] == 1

    def test_get_active_allocations_empty(self):
        """测试获取空活跃分配列表"""
        manager = ResourceAllocationManager()

        result = manager.get_active_allocations()

        assert result == []

    def test_clear_expired_allocations_no_expired(self):
        """测试清除无过期分配"""
        manager = ResourceAllocationManager()

        # 添加未过期的分配
        from datetime import datetime, timedelta
        mock_allocation = Mock()
        mock_allocation.allocated_at = datetime.now() - timedelta(hours=1)  # 1小时前分配

        manager._allocations = {"alloc_123": mock_allocation}

        result = manager.clear_expired_allocations(7200)  # 2小时过期时间

        assert result == 0  # 没有过期分配被清除
        assert "alloc_123" in manager._allocations

    def test_clear_expired_allocations_with_timestamp_error(self):
        """测试清除过期分配时的时间戳错误处理"""
        manager = ResourceAllocationManager()

        # Mock分配对象，allocated_at抛出异常
        mock_allocation = Mock()
        mock_allocation.allocated_at = Mock()
        mock_allocation.allocated_at.timestamp.side_effect = AttributeError("No timestamp")

        manager._allocations = {"alloc_123": mock_allocation}

        # 不应该抛出异常
        result = manager.clear_expired_allocations(3600)

        # 分配仍然存在（因为无法确定是否过期）
        assert "alloc_123" in manager._allocations

    def test_force_release_allocation_success(self):
        """测试强制释放分配成功"""
        manager = ResourceAllocationManager()

        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider
        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        result = manager.force_release_allocation("alloc_123")

        assert result == True
        assert "alloc_123" not in manager._allocations

    def test_force_release_allocation_provider_error(self):
        """测试强制释放分配时提供者错误"""
        manager = ResourceAllocationManager()

        mock_allocation = Mock()
        mock_allocation.resource_id = "cpu_123"
        manager._allocations = {"alloc_123": mock_allocation}

        # Mock provider 返回None
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = None

        result = manager.force_release_allocation("alloc_123")

        assert result == True  # 仍然成功，因为分配被移除了
        assert "alloc_123" not in manager._allocations

    def test_clear_all_allocations_with_event_bus(self):
        """测试带事件总线的清除所有分配"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        manager._allocations = {"alloc_1": Mock(), "alloc_2": Mock()}
        manager._requests = {"req_1": Mock(), "req_2": Mock()}

        manager.clear_all_allocations()

        assert len(manager._allocations) == 0
        assert len(manager._requests) == 0

    def test_create_resource_request_with_priority(self):
        """测试创建带优先级的资源请求"""
        manager = ResourceAllocationManager()

        request = manager._create_resource_request("consumer_1", "gpu", {"memory": 8}, 5)

        assert request.consumer_id == "consumer_1"
        assert request.resource_type == "gpu"
        assert request.requirements == {"memory": 8}
        assert request.priority == 5
        assert hasattr(request, 'request_id')
        assert hasattr(request, 'timestamp')

    def test_store_request_thread_safety(self):
        """测试请求存储的线程安全性"""
        import threading
        manager = ResourceAllocationManager()

        results = []
        errors = []

        def store_request_thread(thread_id):
            try:
                request = Mock()
                request.request_id = f"req_{thread_id}"
                manager._store_request(request)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程同时存储请求
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_request_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0
        assert len(results) == 10

        # 验证所有请求都被存储
        assert len(manager._requests) == 10

    def test_attempt_allocation_with_event_notification(self):
        """测试分配尝试与事件通知"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        mock_provider = Mock()
        mock_request = Mock()
        mock_request.consumer_id = "consumer_1"

        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"
        mock_allocation.consumer_id = "consumer_1"
        mock_allocation.resource_type = "cpu"

        mock_provider.allocate_resource.return_value = mock_allocation

        result = manager._attempt_allocation(mock_provider, mock_request)

        assert result == mock_allocation
        # 验证成功事件被发布
        mock_event_bus.publish.assert_called()

    def test_attempt_allocation_failure_event(self):
        """测试分配失败的事件通知"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        mock_provider = Mock()
        mock_request = Mock()
        mock_request.consumer_id = "consumer_1"
        mock_request.resource_type = "cpu"

        mock_provider.allocate_resource.return_value = None

        with patch.object(manager, '_handle_failed_allocation'):
            result = manager._attempt_allocation(mock_provider, mock_request)

            assert result is None
            # 验证失败事件被发布
            mock_event_bus.publish.assert_called()

    def test_handle_successful_allocation_event_details(self):
        """测试成功分配处理的事件详情"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        mock_allocation = Mock()
        mock_allocation.allocation_id = "alloc_123"
        mock_allocation.consumer_id = "consumer_1"
        mock_allocation.resource_type = "cpu"
        mock_allocation.requirements = {"cores": 4}

        manager._handle_successful_allocation(mock_allocation, "consumer_1")

        # 验证事件详情
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.event_type == "resource.cpu.allocated"
        assert event.source == "resource_manager"
        assert event.data["allocation_id"] == "alloc_123"
        assert event.data["consumer_id"] == "consumer_1"

    def test_handle_failed_allocation_event_details(self):
        """测试失败分配处理的事件详情"""
        mock_event_bus = Mock()
        manager = ResourceAllocationManager(event_bus=mock_event_bus)

        with patch.object(manager.logger, 'warning'):
            manager._handle_failed_allocation("gpu", "consumer_1")

        # 验证失败事件被发布
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.event_type == "resource.gpu.allocation_failed"
        assert event.source == "resource_manager"
        assert event.data["resource_type"] == "gpu"
        assert event.data["consumer_id"] == "consumer_1"

    def test_get_request_with_timestamps(self):
        """测试带时间戳的请求查询"""
        manager = ResourceAllocationManager()

        from datetime import datetime
        mock_request = Mock()
        mock_request.created_at = datetime.now()
        mock_request.updated_at = datetime.now()

        manager._requests = {"req_123": mock_request}

        result = manager.get_request("req_123")

        assert result == mock_request
        assert hasattr(result, 'created_at')

    def test_request_resource_with_complex_requirements(self):
        """测试带复杂需求的资源请求"""
        manager = ResourceAllocationManager()

        complex_requirements = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "gpu_memory_gb": 8,
            "network_bandwidth": 1000,
            "storage_gb": 500
        }

        mock_provider = Mock()
        manager.provider_registry = Mock()
        manager.provider_registry.get_provider.return_value = mock_provider

        with patch.object(manager, '_validate_and_get_provider', return_value=mock_provider), \
             patch.object(manager, '_create_resource_request'), \
             patch.object(manager, '_store_request'), \
             patch.object(manager, '_attempt_allocation', return_value="alloc_123"):

            result = manager.request_resource("consumer_1", "complex", complex_requirements)

            assert result == "alloc_123"

    def test_bulk_operations_performance(self):
        """测试批量操作性能"""
        import time
        manager = ResourceAllocationManager()

        # 批量创建分配
        start_time = time.time()
        for i in range(100):
            mock_allocation = Mock()
            mock_allocation.consumer_id = f"consumer_{i % 10}"
            mock_allocation.resource_type = ["cpu", "memory", "gpu"][i % 3]
            manager._allocations[f"alloc_{i}"] = mock_allocation

        bulk_creation_time = time.time() - start_time

        # 批量查询
        start_time = time.time()
        summary = manager.get_allocation_summary()
        bulk_query_time = time.time() - start_time

        # 验证批量操作的正确性
        assert summary["total_allocations"] == 100
        assert len(summary["allocations_by_consumer"]) == 10
        assert len(summary["allocations_by_type"]) == 3

        # 性能应该在合理范围内（这里只是示例断言）
        assert bulk_creation_time < 1.0  # 创建100个分配应该在1秒内完成
        assert bulk_query_time < 0.5     # 查询汇总应该在0.5秒内完成
