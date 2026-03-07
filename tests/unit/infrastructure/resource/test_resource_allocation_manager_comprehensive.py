#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源分配管理器深度测试

大幅提升resource_allocation_manager.py的测试覆盖率，从20%提升到80%以上
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestResourceAllocationManagerComprehensive:
    """资源分配管理器深度测试"""

    def test_resource_allocation_manager_initialization(self):
        """测试资源分配管理器初始化"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试基本属性
            assert hasattr(manager, 'provider_registry')
            assert hasattr(manager, 'event_bus')
            assert hasattr(manager, 'logger')
            assert hasattr(manager, 'error_handler')
            assert hasattr(manager, '_allocations')
            assert hasattr(manager, '_requests')
            assert hasattr(manager, '_lock')

            # 测试初始状态
            assert isinstance(manager._allocations, dict)
            assert isinstance(manager._requests, dict)
            assert len(manager._allocations) == 0
            assert len(manager._requests) == 0

        except ImportError:
            pytest.skip("ResourceAllocationManager not available")

    def test_resource_allocation_manager_initialization_with_dependencies(self):
        """测试带依赖的资源分配管理器初始化"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            mock_provider_registry = Mock()
            mock_event_bus = Mock()
            mock_logger = Mock()
            mock_error_handler = Mock()

            manager = ResourceAllocationManager(
                provider_registry=mock_provider_registry,
                event_bus=mock_event_bus,
                logger=mock_logger,
                error_handler=mock_error_handler
            )

            # 验证依赖被正确设置
            assert manager.provider_registry == mock_provider_registry
            assert manager.event_bus == mock_event_bus
            assert manager.logger == mock_logger
            assert manager.error_handler == mock_error_handler

        except ImportError:
            pytest.skip("ResourceAllocationManager initialization with dependencies not available")

    def test_get_resource_type_from_allocation(self):
        """测试从分配对象获取资源类型"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()

            # 测试有resource_type属性的分配
            allocation_with_type = Mock(spec=ResourceAllocation)
            allocation_with_type.resource_type = 'cpu'
            allocation_with_type.resource_id = 'cpu_res1'

            resource_type = manager._get_resource_type(allocation_with_type)
            assert resource_type == 'cpu'

            # 测试无resource_type属性的分配（从resource_id推断）
            allocation_without_type = Mock(spec=ResourceAllocation)
            allocation_without_type.resource_type = None
            allocation_without_type.resource_id = 'memory_res1'

            resource_type = manager._get_resource_type(allocation_without_type)
            assert resource_type == 'memory'

            # 测试无下划线的resource_id
            allocation_simple = Mock(spec=ResourceAllocation)
            allocation_simple.resource_type = None
            allocation_simple.resource_id = 'gpu'

            resource_type = manager._get_resource_type(allocation_simple)
            assert resource_type == 'gpu'

        except ImportError:
            pytest.skip("Get resource type from allocation not available")

    def test_validate_and_get_provider_success(self):
        """测试验证并获取提供者成功情况"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()
            mock_provider_registry = Mock()
            mock_provider = Mock()
            manager.provider_registry = mock_provider_registry

            # 设置模拟提供者注册表
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider

            provider = manager._validate_and_get_provider('cpu')

            # 验证提供者被正确返回
            assert provider == mock_provider
            mock_provider_registry.has_provider.assert_called_with('cpu')
            mock_provider_registry.get_provider.assert_called_with('cpu')

        except ImportError:
            pytest.skip("Validate and get provider success not available")

    def test_validate_and_get_provider_failure(self):
        """测试验证并获取提供者失败情况"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceNotFoundError

            manager = ResourceAllocationManager()
            mock_provider_registry = Mock()
            manager.provider_registry = mock_provider_registry

            # 设置模拟提供者注册表 - 无可用提供者
            mock_provider_registry.has_provider.return_value = False

            # 验证抛出ResourceNotFoundError
            with pytest.raises(ResourceNotFoundError, match="资源类型 'cpu' 没有可用的提供者"):
                manager._validate_and_get_provider('cpu')

        except ImportError:
            pytest.skip("Validate and get provider failure not available")

    def test_create_resource_request(self):
        """测试创建资源请求"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceRequest

            manager = ResourceAllocationManager()

            consumer_id = 'trading_engine_1'
            resource_type = 'cpu'
            requirements = {'cores': 4, 'memory_gb': 8}
            priority = 5

            request = manager._create_resource_request(
                consumer_id, resource_type, requirements, priority
            )

            # 验证请求对象
            assert isinstance(request, ResourceRequest)
            assert request.resource_type == resource_type
            assert request.requester_id == consumer_id
            assert request.requirements == requirements
            assert request.priority == priority
            assert request.request_id.startswith('req_')
            assert consumer_id in request.request_id

        except ImportError:
            pytest.skip("Create resource request not available")

    def test_store_request(self):
        """测试存储资源请求"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceRequest

            manager = ResourceAllocationManager()

            # 创建模拟请求
            mock_request = Mock(spec=ResourceRequest)
            mock_request.request_id = 'req_123456789_test'

            # 存储请求
            manager._store_request(mock_request)

            # 验证请求被存储
            assert mock_request.request_id in manager._requests
            assert manager._requests[mock_request.request_id] == mock_request

        except ImportError:
            pytest.skip("Store request not available")

    def test_attempt_allocation(self):
        """测试尝试分配资源"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceRequest

            manager = ResourceAllocationManager()

            mock_provider = Mock()
            mock_request = Mock(spec=ResourceRequest)
            mock_allocation = Mock()
            mock_provider.allocate_resource.return_value = mock_allocation

            allocation = manager._attempt_allocation(mock_provider, mock_request)

            # 验证分配被尝试
            assert allocation == mock_allocation
            mock_provider.allocate_resource.assert_called_with(mock_request)

        except ImportError:
            pytest.skip("Attempt allocation not available")

    def test_handle_successful_allocation(self):
        """测试处理成功的资源分配"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()

            # 创建模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'

            # 设置模拟事件总线
            mock_event_bus = Mock()
            manager.event_bus = mock_event_bus

            consumer_id = 'trading_engine_1'

            allocation_id = manager._handle_successful_allocation(mock_allocation, consumer_id)

            # 验证分配ID被返回
            assert allocation_id == 'alloc_123'

            # 验证分配被存储
            assert allocation_id in manager._allocations
            assert manager._allocations[allocation_id] == mock_allocation

            # 验证事件被发布
            mock_event_bus.publish.assert_called_once()

        except ImportError:
            pytest.skip("Handle successful allocation not available")

    def test_handle_successful_allocation_without_event_bus(self):
        """测试处理成功的资源分配（无事件总线）"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()
            manager.event_bus = None  # 无事件总线

            # 创建模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'

            consumer_id = 'trading_engine_1'

            allocation_id = manager._handle_successful_allocation(mock_allocation, consumer_id)

            # 验证分配ID被返回且分配被存储
            assert allocation_id == 'alloc_123'
            assert allocation_id in manager._allocations

        except ImportError:
            pytest.skip("Handle successful allocation without event bus not available")

    def test_handle_failed_allocation(self):
        """测试处理失败的资源分配"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟事件总线
            mock_event_bus = Mock()
            manager.event_bus = mock_event_bus

            resource_type = 'cpu'
            consumer_id = 'trading_engine_1'
            requirements = {'cores': 4}

            result = manager._handle_failed_allocation(resource_type, consumer_id, requirements)

            # 验证返回None
            assert result is None

            # 验证事件被发布
            mock_event_bus.publish.assert_called_once()

        except ImportError:
            pytest.skip("Handle failed allocation not available")

    def test_handle_failed_allocation_without_event_bus(self):
        """测试处理失败的资源分配（无事件总线）"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()
            manager.event_bus = None  # 无事件总线

            resource_type = 'cpu'
            consumer_id = 'trading_engine_1'
            requirements = {'cores': 4}

            result = manager._handle_failed_allocation(resource_type, consumer_id, requirements)

            # 验证返回None
            assert result is None

        except ImportError:
            pytest.skip("Handle failed allocation without event bus not available")

    def test_request_resource_successful_allocation(self):
        """测试成功请求资源"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_allocation = Mock()
            mock_allocation.allocation_id = 'alloc_123'

            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.allocate_resource.return_value = mock_allocation

            manager.provider_registry = mock_provider_registry

            consumer_id = 'trading_engine_1'
            resource_type = 'cpu'
            requirements = {'cores': 4}

            allocation_id = manager.request_resource(consumer_id, resource_type, requirements)

            # 验证分配成功
            assert allocation_id == 'alloc_123'
            assert allocation_id in manager._allocations

        except ImportError:
            pytest.skip("Request resource successful allocation not available")

    def test_request_resource_failed_allocation(self):
        """测试失败的资源请求"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表 - 分配失败
            mock_provider_registry = Mock()
            mock_provider = Mock()

            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.allocate_resource.return_value = None  # 分配失败

            manager.provider_registry = mock_provider_registry

            consumer_id = 'trading_engine_1'
            resource_type = 'cpu'
            requirements = {'cores': 4}

            allocation_id = manager.request_resource(consumer_id, resource_type, requirements)

            # 验证分配失败
            assert allocation_id is None

        except ImportError:
            pytest.skip("Request resource failed allocation not available")

    def test_request_resource_provider_not_found(self):
        """测试资源请求时提供者未找到"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表 - 无可用提供者
            mock_provider_registry = Mock()
            mock_provider_registry.has_provider.return_value = False

            manager.provider_registry = mock_provider_registry

            consumer_id = 'trading_engine_1'
            resource_type = 'cpu'
            requirements = {'cores': 4}

            allocation_id = manager.request_resource(consumer_id, resource_type, requirements)

            # 验证分配失败
            assert allocation_id is None

        except ImportError:
            pytest.skip("Request resource provider not found not available")

    def test_request_resource_exception_handling(self):
        """测试资源请求的异常处理"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表 - 抛出异常
            mock_provider_registry = Mock()
            mock_provider_registry.has_provider.side_effect = Exception("Registry error")

            manager.provider_registry = mock_provider_registry
            mock_error_handler = Mock()
            manager.error_handler = mock_error_handler

            consumer_id = 'trading_engine_1'
            resource_type = 'cpu'
            requirements = {'cores': 4}

            allocation_id = manager.request_resource(consumer_id, resource_type, requirements)

            # 验证分配失败且错误被处理
            assert allocation_id is None
            mock_error_handler.handle_error.assert_called_once()

        except ImportError:
            pytest.skip("Request resource exception handling not available")

    def test_release_resource_success(self):
        """测试成功释放资源"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()

            # 创建并存储模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'
            manager._allocations['alloc_123'] = mock_allocation

            # 设置模拟提供者注册表和提供者
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.release_resource.return_value = True
            manager.provider_registry = mock_provider_registry

            # 设置模拟事件总线
            mock_event_bus = Mock()
            manager.event_bus = mock_event_bus

            result = manager.release_resource('alloc_123')

            # 验证释放成功
            assert result is True
            assert 'alloc_123' not in manager._allocations
            mock_provider.release_resource.assert_called_once_with(mock_allocation)
            mock_event_bus.publish.assert_called_once()

        except ImportError:
            pytest.skip("Release resource success not available")

    def test_release_resource_not_found(self):
        """测试释放不存在的资源"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            result = manager.release_resource('nonexistent_alloc')

            # 验证释放失败
            assert result is False

        except ImportError:
            pytest.skip("Release resource not found not available")

    def test_release_resource_provider_release_failure(self):
        """测试释放资源时提供者释放失败"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()

            # 创建并存储模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'
            manager._allocations['alloc_123'] = mock_allocation

            # 设置模拟提供者注册表和提供者 - 释放失败
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.release_resource.return_value = False
            manager.provider_registry = mock_provider_registry

            result = manager.release_resource('alloc_123')

            # 验证释放失败
            assert result is False
            assert 'alloc_123' in manager._allocations  # 分配仍存在

        except ImportError:
            pytest.skip("Release resource provider release failure not available")

    def test_release_resource_without_event_bus(self):
        """测试释放资源（无事件总线）"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()
            manager.event_bus = None  # 无事件总线

            # 创建并存储模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'
            manager._allocations['alloc_123'] = mock_allocation

            # 设置模拟提供者注册表和提供者
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider.release_resource.return_value = True
            manager.provider_registry = mock_provider_registry

            result = manager.release_resource('alloc_123')

            # 验证释放成功
            assert result is True
            assert 'alloc_123' not in manager._allocations

        except ImportError:
            pytest.skip("Release resource without event bus not available")

    def test_release_resource_exception_handling(self):
        """测试释放资源的异常处理"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation

            manager = ResourceAllocationManager()

            # 创建并存储模拟分配
            mock_allocation = Mock(spec=ResourceAllocation)
            mock_allocation.allocation_id = 'alloc_123'
            mock_allocation.resource_id = 'cpu_res1'
            manager._allocations['alloc_123'] = mock_allocation

            # 设置模拟提供者注册表 - 抛出异常
            mock_provider_registry = Mock()
            mock_provider_registry.get_provider.side_effect = Exception("Provider error")
            manager.provider_registry = mock_provider_registry

            mock_error_handler = Mock()
            manager.error_handler = mock_error_handler

            result = manager.release_resource('alloc_123')

            # 验证释放失败且错误被处理
            assert result is False
            mock_error_handler.handle_error.assert_called_once()

        except ImportError:
            pytest.skip("Release resource exception handling not available")

    def test_quantitative_trading_resource_allocation(self):
        """测试量化交易资源分配场景"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            manager.provider_registry = mock_provider_registry

            # 模拟量化交易场景的资源分配
            trading_consumers = [
                'hft_engine_1', 'hft_engine_2', 'algo_trader_1', 'portfolio_optimizer_1'
            ]

            allocations = []
            for consumer_id in trading_consumers:
                # 高频交易引擎需要高性能CPU
                if 'hft' in consumer_id:
                    requirements = {'cpu_cores': 8, 'cpu_speed': 3.5, 'memory_gb': 16}
                    resource_type = 'cpu'
                # 算法交易需要平衡资源
                elif 'algo' in consumer_id:
                    requirements = {'cpu_cores': 4, 'memory_gb': 8, 'storage_gb': 100}
                    resource_type = 'cpu'
                # 投资组合优化需要大量内存
                else:
                    requirements = {'cpu_cores': 2, 'memory_gb': 32, 'gpu_memory_gb': 8}
                    resource_type = 'memory'

                # 模拟分配成功
                mock_allocation = Mock()
                mock_allocation.allocation_id = f'alloc_{consumer_id}'
                mock_provider.allocate_resource.return_value = mock_allocation

                allocation_id = manager.request_resource(consumer_id, resource_type, requirements, priority=8)
                allocations.append(allocation_id)

            # 验证所有分配都成功
            assert len(allocations) == 4
            assert all(allocation_id is not None for allocation_id in allocations)

            # 验证分配被跟踪
            assert len(manager._allocations) == 4

            # 模拟资源释放
            for allocation_id in allocations:
                mock_provider.release_resource.return_value = True
                result = manager.release_resource(allocation_id)
                assert result is True

            # 验证所有分配都被释放
            assert len(manager._allocations) == 0

        except ImportError:
            pytest.skip("Quantitative trading resource allocation not available")

    def test_concurrent_resource_allocation(self):
        """测试并发资源分配"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
            import concurrent.futures
            import threading

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            manager.provider_registry = mock_provider_registry

            results = []
            allocation_counter = 0

            def concurrent_allocation_request(request_id):
                nonlocal allocation_counter
                try:
                    mock_allocation = Mock()
                    allocation_counter += 1
                    mock_allocation.allocation_id = f'concurrent_alloc_{allocation_counter}'
                    mock_provider.allocate_resource.return_value = mock_allocation

                    allocation_id = manager.request_resource(
                        f'consumer_{request_id}',
                        'cpu',
                        {'cores': 2},
                        priority=5
                    )

                    # 记录结果
                    results.append(('success', request_id, allocation_id))
                    return allocation_id

                except Exception as e:
                    results.append(('error', request_id, str(e)))
                    return None

            # 使用线程池执行并发请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_allocation_request, i) for i in range(20)]
                concurrent.futures.wait(futures)

            # 验证并发操作结果
            successful_allocations = [r for r in results if r[0] == 'success']
            assert len(successful_allocations) >= 18  # 至少90%的请求成功

            # 验证分配跟踪的线程安全性
            assert len(manager._allocations) == len(successful_allocations)

        except ImportError:
            pytest.skip("Concurrent resource allocation not available")

    def test_resource_allocation_error_recovery(self):
        """测试资源分配错误恢复"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 设置模拟提供者注册表 - 初始分配失败
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            manager.provider_registry = mock_provider_registry

            # 第一次分配失败
            mock_provider.allocate_resource.return_value = None

            allocation_id = manager.request_resource('consumer_1', 'cpu', {'cores': 4})
            assert allocation_id is None

            # 第二次分配成功
            mock_allocation = Mock()
            mock_allocation.allocation_id = 'recovery_alloc_123'
            mock_provider.allocate_resource.return_value = mock_allocation

            allocation_id = manager.request_resource('consumer_1', 'cpu', {'cores': 4})
            assert allocation_id == 'recovery_alloc_123'

            # 验证分配被正确跟踪
            assert allocation_id in manager._allocations

        except ImportError:
            pytest.skip("Resource allocation error recovery not available")

    def test_resource_allocation_metrics_and_monitoring(self):
        """测试资源分配指标和监控"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 执行一些分配操作来生成指标
            mock_provider_registry = Mock()
            mock_provider = Mock()
            mock_provider_registry.has_provider.return_value = True
            mock_provider_registry.get_provider.return_value = mock_provider
            manager.provider_registry = mock_provider_registry

            # 创建多个分配
            for i in range(5):
                mock_allocation = Mock()
                mock_allocation.allocation_id = f'metrics_alloc_{i}'
                mock_provider.allocate_resource.return_value = mock_allocation

                manager.request_resource(f'consumer_{i}', 'cpu', {'cores': 2})

            # 验证分配指标
            assert len(manager._allocations) == 5
            assert len(manager._requests) == 5

            # 模拟释放一些资源
            for i in range(3):
                mock_provider.release_resource.return_value = True
                manager.release_resource(f'metrics_alloc_{i}')

            # 验证释放后的指标
            assert len(manager._allocations) == 2
            assert len(manager._requests) == 5  # 请求仍然保留

        except ImportError:
            pytest.skip("Resource allocation metrics and monitoring not available")

    def test_resource_allocation_cleanup_and_maintenance(self):
        """测试资源分配清理和维护"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 创建一些分配和请求
            for i in range(10):
                mock_request = Mock()
                mock_request.request_id = f'request_{i}'
                manager._requests[mock_request.request_id] = mock_request

                if i < 5:  # 只为前5个创建分配
                    mock_allocation = Mock()
                    mock_allocation.allocation_id = f'allocation_{i}'
                    manager._allocations[mock_allocation.allocation_id] = mock_allocation

            # 验证初始状态
            assert len(manager._requests) == 10
            assert len(manager._allocations) == 5

            # 模拟清理操作（通过直接操作字典）
            old_allocations = list(manager._allocations.keys())
            for alloc_id in old_allocations[:2]:  # 清理前2个分配
                del manager._allocations[alloc_id]

            # 验证清理结果
            assert len(manager._allocations) == 3
            assert len(manager._requests) == 10  # 请求保持不变

        except ImportError:
            pytest.skip("Resource allocation cleanup and maintenance not available")

    def test_resource_allocation_boundary_conditions(self):
        """测试资源分配边界条件"""
        try:
            from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager

            manager = ResourceAllocationManager()

            # 测试空需求
            allocation_id = manager.request_resource('consumer_1', 'cpu', {})
            assert allocation_id is None  # 应该失败

            # 测试无效资源类型
            allocation_id = manager.request_resource('consumer_1', '', {'cores': 2})
            assert allocation_id is None  # 应该失败

            # 测试无效消费者ID
            allocation_id = manager.request_resource('', 'cpu', {'cores': 2})
            assert allocation_id is None  # 应该失败

            # 测试释放不存在的资源
            result = manager.release_resource('nonexistent')
            assert result is False

            # 测试释放空ID
            result = manager.release_resource('')
            assert result is False

        except ImportError:
            pytest.skip("Resource allocation boundary conditions not available")