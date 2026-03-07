#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - Consul服务发现基础测试
测试ConsulServiceDiscovery的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from infrastructure.distributed.consul_service_discovery import (
    ConsulServiceDiscovery, ConsulConfig
)
from infrastructure.distributed.service_mesh import (
    ServiceDiscoveryRequest, ServiceInstance, ServiceStatus
)


class TestConsulConfig:
    """ConsulConfig测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ConsulConfig()

        assert config.host == "localhost"
        assert config.port == 8500
        assert config.scheme == "http"
        assert config.token is None
        assert config.timeout == 30.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.health_check_interval == 30
        assert config.deregister_critical_service_after == "30s"

    def test_custom_config(self):
        """测试自定义配置"""
        config = ConsulConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
            token="test-token",
            timeout=60.0,
            retry_attempts=5,
            health_check_interval=60
        )

        assert config.host == "consul.example.com"
        assert config.port == 8501
        assert config.scheme == "https"
        assert config.token == "test-token"
        assert config.timeout == 60.0
        assert config.retry_attempts == 5
        assert config.health_check_interval == 60


class TestConsulServiceDiscoveryBasic:
    """ConsulServiceDiscovery基础功能测试"""

    @pytest.fixture
    def discovery(self):
        """ConsulServiceDiscovery fixture"""
        return ConsulServiceDiscovery()

    @pytest.fixture
    def sample_instance(self):
        """Sample ServiceInstance fixture"""
        return ServiceInstance(
            id="web-001",
            name="web-service",
            host="192.168.1.100",
            port=8080,
            status=ServiceStatus.HEALTHY,
            metadata={"version": "1.0", "tags": ["web", "api"]},
            weight=10
        )

    @pytest.mark.asyncio
    async def test_register_service_success(self, discovery, sample_instance):
        """测试服务注册成功"""
        result = await discovery.register(sample_instance)

        assert result is True
        assert sample_instance.id in discovery._registered_services
        assert discovery._registered_services[sample_instance.id] == sample_instance

    @pytest.mark.asyncio
    async def test_deregister_service_success(self, discovery, sample_instance):
        """测试服务注销成功"""
        # 先注册服务
        await discovery.register(sample_instance)

        # 注销服务
        result = await discovery.deregister(sample_instance.id)

        assert result is True
        assert sample_instance.id not in discovery._registered_services

    @pytest.mark.asyncio
    async def test_heartbeat_registered_service(self, discovery, sample_instance):
        """测试已注册服务的心跳"""
        await discovery.register(sample_instance)

        result = await discovery.heartbeat(sample_instance.id)

        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_unregistered_service(self, discovery):
        """测试未注册服务的心跳"""
        result = await discovery.heartbeat("unregistered-service")

        assert result is False

    @pytest.mark.asyncio
    async def test_discover_service_with_instances(self, discovery):
        """测试发现有实例的服务"""
        service_name = "test-service"

        # 创建发现请求
        request = ServiceDiscoveryRequest(service_name=service_name)

        # Mock Consul响应
        mock_services = [
            {
                'Service': {
                    'ID': 'inst-001',
                    'Service': service_name,
                    'Address': '10.0.0.1',
                    'Port': 8080,
                    'Meta': {'version': '1.0'},
                    'Weights': {'Passing': 5}
                }
            }
        ]

        with patch.object(discovery, '_make_request', return_value=mock_services):
            instances = await discovery.discover(request)

            assert len(instances) == 1

            # 验证实例
            inst = instances[0]
            assert inst.id == 'inst-001'
            assert inst.name == service_name
            assert inst.host == '10.0.0.1'
            assert inst.port == 8080
            assert inst.weight == 5

    @pytest.mark.asyncio
    async def test_discover_service_no_instances(self, discovery):
        """测试发现无实例的服务"""
        request = ServiceDiscoveryRequest(service_name="empty-service")

        with patch.object(discovery, '_make_request', return_value=[]):
            instances = await discovery.discover(request)

            assert instances == []

    @pytest.mark.asyncio
    async def test_discover_with_request_failure(self, discovery):
        """测试请求失败时的服务发现"""
        request = ServiceDiscoveryRequest(service_name="test-service")

        with patch.object(discovery, '_make_request', return_value=None):
            instances = await discovery.discover(request)

            assert instances == []


class TestConsulServiceDiscoveryMockOperations:
    """ConsulServiceDiscovery Mock操作测试"""

    @pytest.fixture
    def discovery(self):
        """ConsulServiceDiscovery fixture (使用mock实现)"""
        return ConsulServiceDiscovery()

    @pytest.mark.asyncio
    async def test_mock_register_and_discover(self, discovery):
        """测试mock实现的注册和发现"""
        # 注册服务
        instance = ServiceInstance(
            id="mock-service-001",
            name="mock-service",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
            metadata={"version": "1.0"}
        )

        register_result = await discovery.register(instance)
        assert register_result is True

        # 发现服务
        request = ServiceDiscoveryRequest(service_name="mock-service")
        instances = await discovery.discover(request)

        # Mock实现应该返回注册的服务
        assert len(instances) >= 1
        found_instance = next((inst for inst in instances if inst.id == "mock-service-001"), None)
        assert found_instance is not None
        assert found_instance.name == "mock-service"
        assert found_instance.host == "localhost"
        assert found_instance.port == 8080

    @pytest.mark.asyncio
    async def test_mock_deregister(self, discovery):
        """测试mock实现的注销"""
        # 注册服务
        instance = ServiceInstance(
            id="mock-deregister-001",
            name="mock-deregister-service",
            host="localhost",
            port=8080
        )

        await discovery.register(instance)

        # 注销服务
        deregister_result = await discovery.deregister("mock-deregister-001")
        assert deregister_result is True

        # 再次发现应该找不到服务
        request = ServiceDiscoveryRequest(service_name="mock-deregister-service")
        instances = await discovery.discover(request)

        # 服务应该已被移除
        assert not any(inst.id == "mock-deregister-001" for inst in instances)


class TestConsulServiceDiscoveryConcurrency:
    """ConsulServiceDiscovery并发测试"""

    @pytest.fixture
    def discovery(self):
        """ConsulServiceDiscovery fixture"""
        return ConsulServiceDiscovery()

    @pytest.mark.asyncio
    async def test_concurrent_service_registration(self, discovery):
        """测试并发服务注册"""
        async def register_service(service_id: str):
            instance = ServiceInstance(
                id=service_id,
                name="concurrent-service",
                host="localhost",
                port=8000 + int(service_id.split('-')[1]),
                status=ServiceStatus.HEALTHY
            )
            return await discovery.register(instance)

        # 并发注册多个服务
        tasks = [
            register_service(f"service-{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # 所有注册都应该成功
        assert all(results)

        # 所有服务都应该被注册
        assert len(discovery._registered_services) == 5

    @pytest.mark.asyncio
    async def test_concurrent_service_discovery(self, discovery):
        """测试并发服务发现"""
        service_name = "concurrent-discovery-service"

        # 预注册一些服务实例
        for i in range(3):
            instance = ServiceInstance(
                id=f"inst-{i}",
                name=service_name,
                host=f"host-{i}",
                port=8080 + i,
                status=ServiceStatus.HEALTHY
            )
            await discovery.register(instance)

        async def discover_services():
            request = ServiceDiscoveryRequest(service_name=service_name)
            return await discovery.discover(request)

        # 并发发现服务
        tasks = [discover_services() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # 所有发现操作都应该返回相同数量的实例
        for instances in results:
            assert len(instances) == 3


class TestConsulServiceDiscoveryIntegration:
    """ConsulServiceDiscovery集成测试"""

    @pytest.fixture
    def discovery(self):
        """ConsulServiceDiscovery fixture"""
        return ConsulServiceDiscovery()

    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self, discovery):
        """测试服务生命周期管理"""
        service_name = "lifecycle-service"
        instance_id = "lifecycle-001"

        # 1. 注册服务
        instance = ServiceInstance(
            id=instance_id,
            name=service_name,
            host="192.168.1.10",
            port=9000,
            status=ServiceStatus.HEALTHY,
            metadata={"environment": "test", "version": "2.0"}
        )

        register_success = await discovery.register(instance)
        assert register_success

        # 2. 发现服务
        request = ServiceDiscoveryRequest(service_name=service_name)
        instances = await discovery.discover(request)
        assert len(instances) > 0

        found_instance = next((inst for inst in instances if inst.id == instance_id), None)
        assert found_instance is not None

        # 3. 发送心跳
        heartbeat_success = await discovery.heartbeat(instance_id)
        assert heartbeat_success

        # 4. 注销服务
        deregister_success = await discovery.deregister(instance_id)
        assert deregister_success

        # 5. 确认服务已被移除
        instances_after = await discovery.discover(request)
        assert not any(inst.id == instance_id for inst in instances_after)
