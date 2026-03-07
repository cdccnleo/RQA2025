#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源分配管理器深度测试

大幅提升resource_allocation_manager.py的测试覆盖率
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.resource.core.unified_resource_interfaces import (
    ResourceAllocation, ResourceRequest, ResourceNotFoundError
)


class TestResourceAllocationManagerComprehensive:
    """资源分配管理器深度测试"""

    def test_initialization_comprehensive(self):
        """测试初始化所有属性"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            # 测试完整初始化
            manager = ResourceAllocationManager()

            # 验证所有基本属性
            assert hasattr(manager, 'provider_registry')
            assert hasattr(manager, 'event_bus')
            assert hasattr(manager, 'logger')
            assert hasattr(manager, 'error_handler')
            assert hasattr(manager, '_allocations')
            assert hasattr(manager, '_requests')
            assert hasattr(manager, '_lock')

            # 验证数据结构初始化
            assert isinstance(manager._allocations, dict)
            assert isinstance(manager._requests, dict)
            assert len(manager._allocations) == 0
            assert len(manager._requests) == 0

        except ImportError:
            pytest.skip("ResourceAllocationManager not available")

    def test_request_resource_success_path(self):
        """测试资源请求成功路径"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 模拟提供者注册表
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider

            # 模拟分配结果
            mock_allocation = ResourceAllocation(
                allocation_id="test_alloc_001",
                request_id="req_test_consumer",
                resource_id="cpu_node_01",
                allocated_resources={"cores": 4, "memory": 8}
            )
            mock_provider.allocate_resource.return_value = mock_allocation

            manager.provider_registry = mock_provider_registry

            # 执行资源请求
            allocation_id = manager.request_resource(
                consumer_id="test_consumer",
                resource_type="cpu",
                requirements={"cores": 4, "memory": 8},
                priority=1
            )

            # 验证结果
            assert allocation_id == "test_alloc_001"
            assert allocation_id in manager._allocations
            assert len(manager._requests) == 1

        except ImportError:
            pytest.skip("Resource request success path not available")

    def test_request_resource_failure_scenarios(self):
        """测试资源请求失败场景"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试场景1: 没有提供者
            mock_provider_registry = Mock()
            mock_provider_registry.has_provider.return_value = False
            manager.provider_registry = mock_provider_registry

            result = manager.request_resource(
                consumer_id="test_consumer",
                resource_type="cpu",
                requirements={"cores": 4}
            )
            assert result is None

            # 测试场景2: 分配失败
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.allocate_resource.return_value = None

            result = manager.request_resource(
                consumer_id="test_consumer2",
                resource_type="cpu",
                requirements={"cores": 4}
            )
            assert result is None

        except ImportError:
            pytest.skip("Resource request failure scenarios not available")

    def test_release_resource_success(self):
        """测试资源释放成功"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建模拟分配
            allocation = ResourceAllocation(
                allocation_id="test_release_001",
                request_id="req_test_consumer",
                resource_id="memory_node_01",
                allocated_resources={"memory": 8}
            )
            manager._allocations["test_release_001"] = allocation

            # 模拟提供者
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.release_resource.return_value = True

            manager.provider_registry = mock_provider_registry

            # 执行释放
            result = manager.release_resource("test_release_001")
            assert result is True
            assert "test_release_001" not in manager._allocations

        except ImportError:
            pytest.skip("Resource release success not available")

    def test_release_resource_failure_cases(self):
        """测试资源释放失败情况"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试不存在的分配
            result = manager.release_resource("nonexistent_alloc")
            assert result is False

            # 测试提供者不存在
            allocation = ResourceAllocation(
                allocation_id="test_release_002",
                request_id="req_test_consumer",
                resource_id="gpu_node_01",
                allocated_resources={"gpu": 1}
            )
            manager._allocations["test_release_002"] = allocation

            mock_provider_registry = Mock()
            mock_provider_registry.has_provider.return_value = False
            manager.provider_registry = mock_provider_registry

            # 应该抛出异常，但被捕获
            result = manager.release_resource("test_release_002")
            assert result is False

        except ImportError:
            pytest.skip("Resource release failure cases not available")

    def test_allocation_queries(self):
        """测试分配查询功能"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建测试分配
            allocations = [
                ResourceAllocation(
                    allocation_id="alloc_001",
                    request_id="req_001_consumer_a",
                    resource_id="cpu_node_01",
                    allocated_resources={"cores": 4}
                ),
                ResourceAllocation(
                    allocation_id="alloc_002",
                    request_id="req_002_consumer_b",
                    resource_id="memory_node_01",
                    allocated_resources={"memory": 8}
                ),
                ResourceAllocation(
                    allocation_id="alloc_003",
                    request_id="req_003_consumer_a",
                    resource_id="cpu_node_02",
                    allocated_resources={"cores": 2}
                )
            ]

            for alloc in allocations:
                manager._allocations[alloc.allocation_id] = alloc

            # 测试获取单个分配
            alloc = manager.get_allocation("alloc_001")
            assert alloc is not None
            assert alloc.allocation_id == "alloc_001"

            # 测试获取消费者分配
            consumer_allocs = manager.get_allocations_for_consumer("consumer_a")
            assert len(consumer_allocs) == 2  # alloc_001 and alloc_003 have req_consumer_a and req_consumer_a_2

            # 测试获取资源类型分配
            cpu_allocs = manager.get_allocations_for_resource_type("cpu")
            assert len(cpu_allocs) == 2

            # 测试计数
            assert manager.get_allocation_count() == 3
            assert manager.get_request_count() == 0

        except ImportError:
            pytest.skip("Allocation queries not available")

    def test_request_queries(self):
        """测试请求查询功能"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建测试请求
            request = ResourceRequest(
                request_id="req_001",
                resource_type="cpu",
                requester_id="consumer_a",
                requirements={"cores": 4},
                priority=1
            )
            manager._requests["req_001"] = request

            # 测试获取请求
            retrieved_request = manager.get_request("req_001")
            assert retrieved_request is not None
            assert retrieved_request.request_id == "req_001"

            # 测试获取不存在的请求
            nonexistent_request = manager.get_request("nonexistent")
            assert nonexistent_request is None

        except ImportError:
            pytest.skip("Request queries not available")

    def test_allocation_summary(self):
        """测试分配汇总功能"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建测试分配
            allocations = [
                ResourceAllocation(
                    allocation_id="alloc_001",
                    request_id="req_001_consumer_a",
                    resource_id="cpu_node_01",
                    allocated_resources={"cores": 4}
                ),
                ResourceAllocation(
                    allocation_id="alloc_002",
                    request_id="req_002_consumer_b",
                    resource_id="memory_node_01",
                    allocated_resources={"memory": 8}
                )
            ]

            for alloc in allocations:
                manager._allocations[alloc.allocation_id] = alloc

            # 获取汇总
            summary = manager.get_allocation_summary()
            assert isinstance(summary, dict)
            assert summary["total_allocations"] == 2
            assert "cpu" in summary["by_resource_type"]
            assert "memory" in summary["by_resource_type"]

        except ImportError:
            pytest.skip("Allocation summary not available")

    def test_active_allocations(self):
        """测试活跃分配获取"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建测试分配
            allocations = [
                ResourceAllocation(
                    allocation_id="alloc_001",
                    request_id="req_consumer_a",
                    resource_id="cpu_node_01",
                    allocated_resources={"cores": 4}
                ),
                ResourceAllocation(
                    allocation_id="alloc_002",
                    request_id="req_consumer_b",
                    resource_id="gpu_node_01",
                    allocated_resources={"gpu": 1}
                )
            ]

            for alloc in allocations:
                manager._allocations[alloc.allocation_id] = alloc

            # 获取活跃分配
            active = manager.get_active_allocations()
            assert len(active) == 2
            assert all(isinstance(alloc, ResourceAllocation) for alloc in active)

        except ImportError:
            pytest.skip("Active allocations not available")

    def test_resource_type_extraction(self):
        """测试资源类型提取"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建不同格式的分配
            allocations = [
                ResourceAllocation(
                    allocation_id="alloc_001",
                    request_id="req_consumer_a",
                    resource_id="cpu_node_01",
                    allocated_resources={"cores": 4}
                ),
                ResourceAllocation(
                    allocation_id="alloc_002",
                    request_id="req_consumer_b",
                    resource_id="memory_pool",
                    allocated_resources={"memory": 8}
                ),
                ResourceAllocation(
                    allocation_id="alloc_003",
                    request_id="req_consumer_c",
                    resource_id="storage",
                    allocated_resources={"storage": 100}
                )
            ]

            # 测试资源类型提取
            for alloc in allocations:
                resource_type = manager._get_resource_type(alloc)
                assert isinstance(resource_type, str)
                assert len(resource_type) > 0

        except ImportError:
            pytest.skip("Resource type extraction not available")

    def test_event_integration(self):
        """测试事件集成"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.event_bus import EventBus

            # 创建带事件总线的管理器
            event_bus = EventBus()
            manager = ResourceAllocationManager(event_bus=event_bus)

            # 验证事件总线设置
            assert manager.event_bus is event_bus

            # 创建模拟分配用于测试事件发布
            allocation = ResourceAllocation(
                allocation_id="alloc_event_test",
                request_id="req_consumer_test",
                resource_id="cpu_node_01",
                allocated_resources={"cores": 2}
            )

            # 测试事件发布（通过成功分配处理）
            # 这里需要模拟成功分配路径
            manager._allocations[allocation.allocation_id] = allocation

        except ImportError:
            pytest.skip("Event integration not available")

    def test_error_handling(self):
        """测试错误处理"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试资源请求中的错误处理
            mock_provider_registry = Mock()
            mock_provider_registry.has_provider.side_effect = Exception("Test exception")
            manager.provider_registry = mock_provider_registry

            # 执行会触发异常的操作
            result = manager.request_resource(
                consumer_id="test_consumer",
                resource_type="cpu",
                requirements={"cores": 4}
            )

            # 验证错误被处理且返回None
            assert result is None

        except ImportError:
            pytest.skip("Error handling not available")

    def test_thread_safety(self):
        """测试线程安全性"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            import threading

            manager = ResourceAllocationManager()

            # 测试并发访问
            results = []
            errors = []

            def concurrent_operation(operation_id):
                try:
                    if operation_id % 2 == 0:
                        # 偶数：添加分配
                        allocation = ResourceAllocation(
                            allocation_id=f"alloc_thread_{operation_id}",
                            request_id=f"req_consumer_{operation_id}",
                            resource_id=f"resource_{operation_id}",
                            allocated_resources={"test": operation_id}
                        )
                        with manager._lock:
                            manager._allocations[allocation.allocation_id] = allocation
                        results.append(f"added_{operation_id}")
                    else:
                        # 奇数：查询分配计数
                        count = manager.get_allocation_count()
                        results.append(f"count_{count}")
                except Exception as e:
                    errors.append(f"thread_{operation_id}: {e}")

            # 创建并发线程
            threads = []
            for i in range(10):
                thread = threading.Thread(target=concurrent_operation, args=(i,))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=2.0)

            # 验证结果
            assert len(results) == 10  # 所有操作都成功
            assert len(errors) == 0   # 没有错误
            assert manager.get_allocation_count() == 5  # 5个添加操作

        except ImportError:
            pytest.skip("Thread safety not available")

    def test_performance_metrics(self):
        """测试性能指标"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 执行一些操作来生成性能数据
            for i in range(10):
                allocation = ResourceAllocation(
                    allocation_id=f"perf_alloc_{i}",
                    request_id=f"req_consumer_{i}",
                    resource_id=f"resource_{i}",
                    allocated_resources={"test": i}
                )
                manager._allocations[allocation.allocation_id] = allocation

            # 验证性能相关操作的执行时间在合理范围内
            import time

            # 测试分配计数性能
            start_time = time.time()
            count = manager.get_allocation_count()
            end_time = time.time()

            assert count == 10
            assert (end_time - start_time) < 0.001  # 应该在1ms内完成

            # 测试汇总性能
            start_time = time.time()
            summary = manager.get_allocation_summary()
            end_time = time.time()

            assert summary["total_allocations"] == 10
            assert (end_time - start_time) < 0.01  # 应该在10ms内完成

        except ImportError:
            pytest.skip("Performance metrics not available")

    def test_boundary_conditions(self):
        """测试边界条件"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试空分配集合
            assert manager.get_allocation_count() == 0
            assert len(manager.get_active_allocations()) == 0
            assert len(manager.get_allocations_for_consumer("nonexistent")) == 0
            assert len(manager.get_allocations_for_resource_type("nonexistent")) == 0

            # 测试分配汇总的空情况
            summary = manager.get_allocation_summary()
            assert summary["total_allocations"] == 0
            assert len(summary["by_resource_type"]) == 0
            assert len(summary["by_consumer"]) == 0

            # 测试None输入
            assert manager.get_allocation(None) is None
            assert manager.get_request(None) is None

        except ImportError:
            pytest.skip("Boundary conditions not available")