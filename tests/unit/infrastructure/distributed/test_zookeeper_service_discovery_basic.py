#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - ZooKeeper服务发现基础测试
测试ZooKeeperServiceDiscovery的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from infrastructure.distributed.zookeeper_service_discovery import (
    ZooKeeperServiceDiscovery, ZooKeeperConfig
)
from infrastructure.distributed.service_mesh import (
    ServiceDiscoveryRequest, ServiceInstance, ServiceStatus
)


class TestZooKeeperConfig:
    """ZooKeeperConfig测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ZooKeeperConfig()

        assert config.hosts == "localhost:2181"
        assert config.session_timeout == 30000
        assert config.connection_timeout == 10000
        assert config.base_path == "/services"
        assert config.auth_scheme is None
        assert config.auth_data is None
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.health_check_interval == 30
        assert config.ephemeral_nodes is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = ZooKeeperConfig(
            hosts="zk1:2181,zk2:2181,zk3:2181",
            session_timeout=60000,
            base_path="/my-services",
            auth_scheme="digest",
            auth_data="user:password",
            ephemeral_nodes=False
        )

        assert config.hosts == "zk1:2181,zk2:2181,zk3:2181"
        assert config.session_timeout == 60000
        assert config.base_path == "/my-services"
        assert config.auth_scheme == "digest"
        assert config.auth_data == "user:password"
        assert config.ephemeral_nodes is False


class TestZooKeeperServiceDiscoveryBasic:
    """ZooKeeperServiceDiscovery基础功能测试"""

    @pytest.fixture
    def discovery(self):
        """ZooKeeperServiceDiscovery fixture"""
        return ZooKeeperServiceDiscovery()

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
    async def test_deregister_unregistered_service(self, discovery):
        """测试注销未注册的服务"""
        result = await discovery.deregister("nonexistent-service")

        assert result is False

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
    async def test_discover_service_mock(self, discovery):
        """测试Mock模式的服务发现"""
        service_name = "test-service"

        # 注册一些服务
        for i in range(3):
            instance = ServiceInstance(
                id=f"inst-{i}",
                name=service_name,
                host=f"host-{i}",
                port=8080 + i,
                status=ServiceStatus.HEALTHY
            )
            await discovery.register(instance)

        # 发现服务
        request = ServiceDiscoveryRequest(service_name=service_name)
        instances = await discovery.discover(request)

        assert len(instances) == 3
        instance_ids = {inst.id for inst in instances}
        expected_ids = {f"inst-{i}" for i in range(3)}
        assert instance_ids == expected_ids

    @pytest.mark.asyncio
    async def test_discover_service_with_tags_filter(self, discovery):
        """测试带标签过滤的服务发现"""
        # 注册不同标签的服务
        web_instance = ServiceInstance(
            id="web-001",
            name="api-service",
            host="host1",
            port=8080,
            metadata={"tags": ["web", "api"]}
        )

        worker_instance = ServiceInstance(
            id="worker-001",
            name="api-service",
            host="host2",
            port=8081,
            metadata={"tags": ["worker"]}
        )

        await discovery.register(web_instance)
        await discovery.register(worker_instance)

        # 只查找web标签的服务
        request = ServiceDiscoveryRequest(
            service_name="api-service",
            tags=["web"]
        )
        instances = await discovery.discover(request)

        assert len(instances) == 1
        assert instances[0].id == "web-001"

    @pytest.mark.asyncio
    async def test_discover_service_only_healthy(self, discovery):
        """测试只发现健康实例"""
        # 注册健康和不健康的服务
        healthy_instance = ServiceInstance(
            id="healthy-001",
            name="test-service",
            host="host1",
            port=8080,
            status=ServiceStatus.HEALTHY
        )

        unhealthy_instance = ServiceInstance(
            id="unhealthy-001",
            name="test-service",
            host="host2",
            port=8081,
            status=ServiceStatus.UNHEALTHY
        )

        await discovery.register(healthy_instance)
        await discovery.register(unhealthy_instance)

        # 只查找健康服务
        request = ServiceDiscoveryRequest(
            service_name="test-service",
            only_healthy=True
        )
        instances = await discovery.discover(request)

        assert len(instances) == 1
        assert instances[0].id == "healthy-001"
        assert instances[0].is_healthy()


class TestZooKeeperServiceDiscoverySerialization:
    """ZooKeeperServiceDiscovery序列化测试"""

    @pytest.fixture
    def discovery(self):
        """ZooKeeperServiceDiscovery fixture"""
        return ZooKeeperServiceDiscovery()

    def test_serialize_instance(self, discovery):
        """测试实例序列化"""
        instance = ServiceInstance(
            id="test-001",
            name="test-service",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
            metadata={"version": "1.0", "env": "prod"},
            weight=5
        )

        serialized = discovery._serialize_instance(instance)
        assert isinstance(serialized, str)

        # 验证包含所有必要字段
        import json
        data = json.loads(serialized)
        assert data['id'] == "test-001"
        assert data['name'] == "test-service"
        assert data['host'] == "localhost"
        assert data['port'] == 8080
        assert data['status'] == "healthy"
        assert data['metadata']['version'] == "1.0"
        assert data['weight'] == 5

    def test_deserialize_instance(self, discovery):
        """测试实例反序列化"""
        json_data = '''{
            "id": "test-001",
            "name": "test-service",
            "host": "localhost",
            "port": 8080,
            "status": "healthy",
            "metadata": {"version": "1.0"},
            "weight": 5
        }'''

        instance = discovery._deserialize_instance(json_data)

        assert instance is not None
        assert instance.id == "test-001"
        assert instance.name == "test-service"
        assert instance.host == "localhost"
        assert instance.port == 8080
        assert instance.status == ServiceStatus.HEALTHY
        assert instance.metadata["version"] == "1.0"
        assert instance.weight == 5

    def test_deserialize_invalid_data(self, discovery):
        """测试反序列化无效数据"""
        invalid_json = '{"invalid": "data"}'

        instance = discovery._deserialize_instance(invalid_json)

        assert instance is None

    def test_matches_tags(self, discovery):
        """测试标签匹配"""
        instance = ServiceInstance(
            id="test",
            name="service",
            host="host",
            port=8080,
            metadata={"tags": ["web", "api", "v1"]}
        )

        # 完全匹配
        assert discovery._matches_tags(instance, ["web"]) is True
        assert discovery._matches_tags(instance, ["api", "v1"]) is True

        # 部分匹配
        assert discovery._matches_tags(instance, ["web", "mobile"]) is False

        # 空标签
        assert discovery._matches_tags(instance, []) is True

        # 不存在的标签
        assert discovery._matches_tags(instance, ["nonexistent"]) is False


class TestZooKeeperServiceDiscoveryConcurrency:
    """ZooKeeperServiceDiscovery并发测试"""

    @pytest.fixture
    def discovery(self):
        """ZooKeeperServiceDiscovery fixture"""
        return ZooKeeperServiceDiscovery()

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


class TestZooKeeperServiceDiscoveryHealthChecks:
    """ZooKeeperServiceDiscovery健康检查测试"""

    @pytest.fixture
    def discovery(self):
        """ZooKeeperServiceDiscovery fixture"""
        return ZooKeeperServiceDiscovery()

    @pytest.fixture
    def sample_instance(self):
        """Sample ServiceInstance fixture"""
        return ServiceInstance(
            id="health-check-service",
            name="test-service",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY
        )

    @pytest.mark.asyncio
    async def test_health_check_task_creation(self, discovery, sample_instance):
        """测试健康检查任务创建"""
        await discovery.register(sample_instance)

        # 应该创建健康检查任务
        assert sample_instance.id in discovery._health_check_tasks

        task = discovery._health_check_tasks[sample_instance.id]
        assert isinstance(task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_health_check_task_cleanup_on_deregister(self, discovery, sample_instance):
        """测试注销时健康检查任务清理"""
        await discovery.register(sample_instance)

        # 确认任务存在
        assert sample_instance.id in discovery._health_check_tasks

        # 注销服务
        await discovery.deregister(sample_instance.id)

        # 任务应该被清理
        assert sample_instance.id not in discovery._health_check_tasks

    @pytest.mark.asyncio
    async def test_perform_health_check(self, discovery, sample_instance):
        """测试执行健康检查"""
        initial_check_time = sample_instance.last_health_check

        await discovery._perform_health_check(sample_instance)

        # 健康检查时间应该被更新
        assert sample_instance.last_health_check is not None
        if initial_check_time is not None:
            assert sample_instance.last_health_check > initial_check_time


class TestZooKeeperServiceDiscoveryIntegration:
    """ZooKeeperServiceDiscovery集成测试"""

    @pytest.fixture
    def discovery(self):
        """ZooKeeperServiceDiscovery fixture"""
        return ZooKeeperServiceDiscovery()

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

    @pytest.mark.asyncio
    async def test_multiple_services_management(self, discovery):
        """测试多服务管理"""
        services = []

        # 注册多个不同服务
        for i in range(3):
            instance = ServiceInstance(
                id=f"multi-service-{i}",
                name=f"service-{i}",
                host=f"host-{i}",
                port=8000 + i,
                status=ServiceStatus.HEALTHY,
                metadata={"group": f"group-{i % 2}"}
            )
            services.append(instance)

            success = await discovery.register(instance)
            assert success

        # 验证所有服务都能被发现
        for service in services:
            request = ServiceDiscoveryRequest(service_name=service.name)
            instances = await discovery.discover(request)

            assert len(instances) == 1
            assert instances[0].id == service.id

    @pytest.mark.asyncio
    async def test_service_health_monitoring_simulation(self, discovery):
        """测试服务健康监控模拟"""
        instance = ServiceInstance(
            id="health-monitor-001",
            name="health-monitor-service",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY
        )

        # 注册服务
        await discovery.register(instance)

        # 模拟健康检查循环
        initial_check_time = instance.last_health_check

        # 等待一段时间让健康检查运行
        await asyncio.sleep(0.1)

        # 执行手动健康检查
        await discovery._perform_health_check(instance)

        # 健康检查时间应该更新
        assert instance.last_health_check is not None
        if initial_check_time is not None:
            assert instance.last_health_check > initial_check_time
