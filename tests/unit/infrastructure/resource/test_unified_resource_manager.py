"""
测试统一资源管理器

验证UnifiedResourceManager类的核心功能，包括组件协调、注册管理和分配策略
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any
from datetime import datetime

from src.infrastructure.resource.core.unified_resource_manager import UnifiedResourceManager
from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
from src.infrastructure.resource.core.resource_consumer_registry import ResourceConsumerRegistry
from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation


class TestUnifiedResourceManager:
    """测试UnifiedResourceManager类"""

    def test_unified_resource_manager_initialization(self):
        """测试统一资源管理器初始化"""
        manager = UnifiedResourceManager()

        # 验证核心组件初始化
        assert hasattr(manager, 'provider_registry')
        assert hasattr(manager, 'consumer_registry')
        assert hasattr(manager, 'allocation_manager')
        assert hasattr(manager, 'event_bus')
        assert hasattr(manager, 'container')

        # 验证组件类型
        assert isinstance(manager.provider_registry, ResourceProviderRegistry)
        assert isinstance(manager.consumer_registry, ResourceConsumerRegistry)
        assert isinstance(manager.allocation_manager, ResourceAllocationManager)

    def test_unified_resource_manager_with_config(self):
        """测试带配置的统一资源管理器初始化"""
        config = {
            'allocation_strategy': 'priority_based',
            'max_concurrent_allocations': 10
        }

        manager = UnifiedResourceManager(config=config)

        assert manager.config == config
        assert manager.config['allocation_strategy'] == 'priority_based'

    def test_register_provider(self):
        """测试注册资源提供者"""
        manager = UnifiedResourceManager()

        # 创建模拟提供者
        mock_provider = MagicMock()
        mock_provider.provider_id = "test_provider"
        mock_provider.resource_type = "cpu"

        # 注册提供者
        result = manager.register_provider(mock_provider)

        assert result is True
        # 验证提供者已被注册（使用resource_type作为键）
        assert "cpu" in manager.provider_registry._providers

    def test_register_consumer(self):
        """测试注册资源消费者"""
        manager = UnifiedResourceManager()

        # 创建模拟消费者
        mock_consumer = MagicMock()
        mock_consumer.consumer_id = "test_consumer"
        mock_consumer.resource_requirements = {"cpu": 2}

        # 注册消费者
        result = manager.register_consumer(mock_consumer)

        assert result is True
        # 验证消费者已被注册（检查注册表中是否有消费者）
        assert len(manager.consumer_registry._consumers) > 0
        # 验证我们注册的消费者在注册表中
        assert mock_consumer in manager.consumer_registry._consumers.values()

    def test_request_resource(self):
        """测试资源请求"""
        manager = UnifiedResourceManager()

        # 创建模拟提供者和消费者
        mock_provider = MagicMock()
        mock_provider.provider_id = "cpu_provider"
        mock_provider.resource_type = "cpu"
        
        # 创建正确的ResourceAllocation对象作为返回值
        mock_allocation = ResourceAllocation(
            allocation_id="allocation_123",
            request_id="req_123",
            resource_id="cpu_res_123",
            allocated_resources={"cores": 2},
            allocated_at=datetime.now()
        )
        mock_provider.allocate_resource.return_value = mock_allocation

        mock_consumer = MagicMock()
        mock_consumer.consumer_id = "consumer_1"

        # 注册组件
        manager.register_provider(mock_provider)
        manager.register_consumer(mock_consumer)
        
        # 确保allocation_manager使用正确的provider_registry实例
        allocation_manager = manager.allocation_manager
        allocation_manager.provider_registry = manager.provider_registry

        # 请求资源
        allocation_id = manager.request_resource(
            consumer_id="consumer_1",
            resource_type="cpu",
            requirements={"cores": 2},
            priority=1
        )

        assert allocation_id is not None
        mock_provider.allocate_resource.assert_called_once()

    def test_release_resource(self):
        """测试资源释放"""
        manager = UnifiedResourceManager()

        # 创建模拟提供者
        mock_provider = MagicMock()
        mock_provider.provider_id = "cpu_provider"
        mock_provider.resource_type = "cpu"
        mock_provider.release_resource.return_value = True

        manager.register_provider(mock_provider)
        
        # 确保allocation_manager使用正确的provider_registry实例
        allocation_manager = manager.allocation_manager
        allocation_manager.provider_registry = manager.provider_registry
        
        # 创建一个模拟的分配对象并添加到分配管理器中
        mock_allocation = ResourceAllocation(
            allocation_id="allocation_123",
            request_id="req_123",
            resource_id="cpu_res_123",
            allocated_resources={"cores": 2},
            allocated_at=datetime.now()
        )
        allocation_manager._allocations["allocation_123"] = mock_allocation

        # 释放资源
        result = manager.release_resource("allocation_123")

        assert result is True
        mock_provider.release_resource.assert_called_once_with("allocation_123")

    def test_get_system_status(self):
        """测试获取系统状态"""
        manager = UnifiedResourceManager()

        status = manager.get_system_status()

        # 验证状态包含必要信息
        assert isinstance(status, dict)
        assert 'providers_count' in status
        assert 'consumers_count' in status
        assert 'allocations_count' in status
        assert 'system_health' in status

    def test_optimize_resources(self):
        """测试资源优化"""
        manager = UnifiedResourceManager()

        # 创建模拟优化配置
        config = {
            'memory_optimization': {'enabled': True, 'target_usage': 70.0},
            'cpu_optimization': {'enabled': True, 'target_usage': 60.0}
        }

        result = manager.optimize_resources(config)

        # 验证优化结果
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'optimizations_applied' in result

    @patch('src.infrastructure.resource.core.unified_resource_manager.UnifiedResourceManager._cleanup_allocations')
    def test_shutdown(self, mock_cleanup):
        """测试系统关闭"""
        manager = UnifiedResourceManager()

        # 执行关闭
        manager.shutdown()

        # 验证清理方法被调用
        mock_cleanup.assert_called_once()

    def test_event_bus_integration(self):
        """测试事件总线集成"""
        manager = UnifiedResourceManager()

        # 验证事件总线初始化
        assert manager.event_bus is not None

        # 测试事件发布（如果有相关方法）
        # 这里可以添加更多事件相关的测试

    def test_dependency_container_integration(self):
        """测试依赖注入容器集成"""
        manager = UnifiedResourceManager()

        # 验证容器初始化
        assert manager.container is not None

        # 测试服务注册和解析（如果有相关方法）
        # 这里可以添加更多依赖注入相关的测试


