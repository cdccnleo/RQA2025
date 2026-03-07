#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""服务网格测试"""

from src.infrastructure.distributed.service_mesh import (
    ServiceStatus,
    LoadBalanceStrategy,
    ServiceInstance,
    ServiceDiscoveryRequest
)


class TestEnums:
    """测试枚举类"""

    def test_service_status_enum(self):
        """测试服务状态枚举"""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.UNKNOWN.value == "unknown"
        assert ServiceStatus.MAINTENANCE.value == "maintenance"

        # 测试所有枚举值
        expected_values = ["healthy", "unhealthy", "unknown", "maintenance"]
        actual_values = [member.value for member in ServiceStatus]
        assert set(actual_values) == set(expected_values)

    def test_load_balance_strategy_enum(self):
        """测试负载均衡策略枚举"""
        assert LoadBalanceStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalanceStrategy.RANDOM.value == "random"

        # 测试所有枚举值
        expected_values = ["round_robin", "random"]
        actual_values = [member.value for member in LoadBalanceStrategy]
        assert set(actual_values) == set(expected_values)


class TestServiceInstance:
    """测试服务实例"""

    def test_init_default(self):
        """测试默认初始化"""
        instance = ServiceInstance()

        assert instance.id == "-localhost:0"  # ID在__post_init__中生成
        assert instance.name == ""
        assert instance.host == "localhost"
        assert instance.port == 0
        assert instance.status == ServiceStatus.HEALTHY
        assert instance.metadata == {}
        assert instance.weight == 1
        assert instance.service_name == ""  # 在__post_init__中设置
        assert instance.last_health_check is None

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        metadata = {"version": "1.0", "region": "us-east-1"}
        instance = ServiceInstance(
            name="web-service",
            host="10.0.0.1",
            port=8080,
            status=ServiceStatus.HEALTHY,
            metadata=metadata,
            weight=5
        )

        assert instance.name == "web-service"
        assert instance.host == "10.0.0.1"
        assert instance.port == 8080
        assert instance.status == ServiceStatus.HEALTHY
        assert instance.metadata == metadata
        assert instance.weight == 5

    def test_post_init_service_name_fallback(self):
        """测试后初始化服务名称回退"""
        # 测试service_name为空时使用name
        instance = ServiceInstance(name="test-service")
        assert instance.service_name == "test-service"

        # 测试name为空时使用service_name
        instance = ServiceInstance(service_name="another-service")
        assert instance.name == "another-service"

    def test_post_init_id_generation(self):
        """测试ID自动生成"""
        instance = ServiceInstance(
            name="web-service",
            host="10.0.0.1",
            port=8080
        )

        expected_id = "web-service-10.0.0.1:8080"
        assert instance.id == expected_id

    def test_is_healthy(self):
        """测试健康状态检查"""
        # 健康的实例
        healthy_instance = ServiceInstance(status=ServiceStatus.HEALTHY)
        assert healthy_instance.is_healthy() is True

        # 不健康的实例
        unhealthy_instance = ServiceInstance(status=ServiceStatus.UNHEALTHY)
        assert unhealthy_instance.is_healthy() is False

        unknown_instance = ServiceInstance(status=ServiceStatus.UNKNOWN)
        assert unknown_instance.is_healthy() is False

        maintenance_instance = ServiceInstance(status=ServiceStatus.MAINTENANCE)
        assert maintenance_instance.is_healthy() is False

    def test_weight_operations(self):
        """测试权重操作"""
        instance = ServiceInstance(weight=10)
        assert instance.weight == 10

        # 修改权重
        instance.weight = 20
        assert instance.weight == 20

    def test_metadata_operations(self):
        """测试元数据操作"""
        metadata = {"version": "2.0", "env": "prod"}
        instance = ServiceInstance(metadata=metadata)

        assert instance.metadata == metadata

        # 添加元数据
        instance.metadata["owner"] = "team-a"
        assert instance.metadata["owner"] == "team-a"

    def test_last_health_check(self):
        """测试最后健康检查时间"""
        import time

        instance = ServiceInstance()
        assert instance.last_health_check is None

        # 设置健康检查时间
        check_time = time.time()
        instance.last_health_check = check_time
        assert instance.last_health_check == check_time


class TestServiceDiscoveryRequest:
    """测试服务发现请求"""

    def test_init_default(self):
        """测试默认初始化"""
        request = ServiceDiscoveryRequest(service_name="")

        assert request.service_name == ""
        assert request.only_healthy is True

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        request = ServiceDiscoveryRequest(
            service_name="web-service",
            only_healthy=True
        )

        assert request.service_name == "web-service"
        assert request.only_healthy is True

    def test_service_name_operations(self):
        """测试服务名称操作"""
        request = ServiceDiscoveryRequest(service_name="api-gateway")
        assert request.service_name == "api-gateway"

        # 修改服务名称
        request.service_name = "auth-service"
        assert request.service_name == "auth-service"

    def test_only_healthy_operations(self):
        """测试仅健康操作"""
        request = ServiceDiscoveryRequest(service_name="test-service", only_healthy=True)
        assert request.only_healthy is True

        request.only_healthy = False
        assert request.only_healthy is False


class TestServiceMeshIntegration:
    """测试服务网格集成"""

    def test_service_instance_creation(self):
        """测试服务实例创建"""
        instance = ServiceInstance(
            name="web-service",
            host="10.0.0.1",
            port=8080,
            status=ServiceStatus.HEALTHY,
            weight=3
        )

        assert instance.is_healthy() is True
        assert instance.id == "web-service-10.0.0.1:8080"
        assert instance.service_name == "web-service"

    def test_multiple_service_instances(self):
        """测试多个服务实例"""
        instances = [
            ServiceInstance(name="web-1", host="10.0.0.1", port=8080, weight=2),
            ServiceInstance(name="web-2", host="10.0.0.2", port=8080, weight=3),
            ServiceInstance(name="web-3", host="10.0.0.3", port=8080, weight=1, status=ServiceStatus.UNHEALTHY)
        ]

        # 检查健康实例
        healthy_instances = [inst for inst in instances if inst.is_healthy()]
        assert len(healthy_instances) == 2

        # 检查权重
        total_weight = sum(inst.weight for inst in healthy_instances)
        assert total_weight == 5

    def test_service_discovery_request_creation(self):
        """测试服务发现请求创建"""
        request = ServiceDiscoveryRequest(
            service_name="user-service",
            only_healthy=False
        )

        assert request.service_name == "user-service"
        assert request.only_healthy is False

    def test_enum_values_consistency(self):
        """测试枚举值一致性"""
        # ServiceStatus 枚举值应该是字符串
        assert isinstance(ServiceStatus.HEALTHY.value, str)
        assert isinstance(ServiceStatus.UNHEALTHY.value, str)

        # LoadBalanceStrategy 枚举值应该是字符串
        assert isinstance(LoadBalanceStrategy.ROUND_ROBIN.value, str)
        assert isinstance(LoadBalanceStrategy.RANDOM.value, str)
