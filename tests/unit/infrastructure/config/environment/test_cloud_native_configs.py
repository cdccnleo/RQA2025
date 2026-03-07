#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""云原生配置测试"""

from src.infrastructure.config.environment.cloud_native_configs import (
    ServiceMeshType,
    CloudProvider,
    ScalingPolicy,
    ServiceMeshConfig,
    MultiCloudConfig,
    AutoScalingConfig,
    CloudNativeMonitoringConfig
)


class TestEnums:
    """测试枚举类"""

    def test_service_mesh_type_enum(self):
        """测试服务网格类型枚举"""
        assert ServiceMeshType.ISTIO.value == "istio"
        assert ServiceMeshType.LINKERD.value == "linkerd"
        assert ServiceMeshType.CONSUL.value == "consul"
        assert ServiceMeshType.AWS_APP_MESH.value == "aws_app_mesh"
        assert ServiceMeshType.KUMA.value == "kuma"

        # 测试所有枚举值
        expected_values = ["istio", "linkerd", "consul", "aws_app_mesh", "kuma"]
        actual_values = [member.value for member in ServiceMeshType]
        assert set(actual_values) == set(expected_values)

    def test_cloud_provider_enum(self):
        """测试云提供商枚举"""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.ALIBABA.value == "alibaba"
        assert CloudProvider.TENCENT.value == "tencent"
        assert CloudProvider.HUAWEI.value == "huawei"

        # 测试所有枚举值
        expected_values = ["aws", "azure", "gcp", "alibaba", "tencent", "huawei"]
        actual_values = [member.value for member in CloudProvider]
        assert set(actual_values) == set(expected_values)

    def test_scaling_policy_enum(self):
        """测试自动伸缩策略枚举"""
        assert ScalingPolicy.CPU_UTILIZATION.value == "cpu_utilization"
        assert ScalingPolicy.MEMORY_UTILIZATION.value == "memory_utilization"
        assert ScalingPolicy.REQUEST_RATE.value == "request_rate"
        assert ScalingPolicy.CUSTOM_METRIC.value == "custom_metric"
        assert ScalingPolicy.SCHEDULED.value == "scheduled"

        # 测试所有枚举值
        expected_values = ["cpu_utilization", "memory_utilization", "request_rate", "custom_metric", "scheduled"]
        actual_values = [member.value for member in ScalingPolicy]
        assert set(actual_values) == set(expected_values)


class TestServiceMeshConfig:
    """测试服务网格配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = ServiceMeshConfig()

        assert config.enabled is True
        assert config.mesh_type == ServiceMeshType.ISTIO
        assert config.namespace == "istio-system"
        assert config.version == "1.20.0"
        assert config.enable_mtls is True
        assert config.enable_tracing is True
        assert config.enable_metrics is True
        assert config.custom_annotations == {}
        assert config.custom_labels == {}

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        config = ServiceMeshConfig(
            enabled=False,
            mesh_type=ServiceMeshType.LINKERD,
            namespace="linkerd-system",
            version="2.14.0"
        )

        assert config.enabled is False
        assert config.mesh_type == ServiceMeshType.LINKERD
        assert config.namespace == "linkerd-system"
        assert config.version == "2.14.0"

    def test_post_init_with_string_mesh_type(self):
        """测试后初始化处理字符串类型的mesh_type"""
        config = ServiceMeshConfig(mesh_type="consul")

        assert config.mesh_type == ServiceMeshType.CONSUL
        assert isinstance(config.mesh_type, ServiceMeshType)

    def test_custom_annotations_and_labels(self):
        """测试自定义注解和标签"""
        annotations = {"sidecar.istio.io/status": "healthy"}
        labels = {"app": "web", "version": "v1"}

        config = ServiceMeshConfig(
            custom_annotations=annotations,
            custom_labels=labels
        )

        assert config.custom_annotations == annotations
        assert config.custom_labels == labels


class TestMultiCloudConfig:
    """测试多云配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = MultiCloudConfig()

        assert config.enabled is False
        assert config.primary_provider == CloudProvider.AWS
        assert config.secondary_providers == []
        assert config.region_mapping == {}
        assert config.failover_enabled is True
        assert config.load_balancing_strategy == "round_robin"
        assert config.health_check_interval == 30
        assert config.custom_config == {}

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        config = MultiCloudConfig(
            enabled=True,
            primary_provider=CloudProvider.GCP,
            secondary_providers=[CloudProvider.AWS, CloudProvider.AZURE],
            failover_enabled=False
        )

        assert config.enabled is True
        assert config.primary_provider == CloudProvider.GCP
        assert config.secondary_providers == [CloudProvider.AWS, CloudProvider.AZURE]
        assert config.failover_enabled is False

    def test_post_init_with_string_providers(self):
        """测试后初始化处理字符串类型的providers"""
        config = MultiCloudConfig(
            primary_provider="azure",
            secondary_providers=["aws", "gcp"]
        )

        assert config.primary_provider == CloudProvider.AZURE
        assert config.secondary_providers == [CloudProvider.AWS, CloudProvider.GCP]
        assert all(isinstance(p, CloudProvider) for p in config.secondary_providers)

    def test_region_mapping(self):
        """测试区域映射"""
        region_mapping = {
            "us-west-1": "aws",
            "eastus": "azure",
            "us-central1": "gcp"
        }

        config = MultiCloudConfig(region_mapping=region_mapping)
        assert config.region_mapping == region_mapping


class TestAutoScalingConfig:
    """测试自动伸缩配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = AutoScalingConfig()

        assert config.enabled is True
        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.target_cpu_utilization == 70
        assert config.target_memory_utilization == 80
        assert config.scale_up_threshold == 80
        assert config.scale_down_threshold == 30
        assert config.stabilization_window_seconds == 300
        assert config.scaling_policy == ScalingPolicy.CPU_UTILIZATION
        assert config.custom_metrics == []
        assert config.cooldown_period_seconds == 60

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        config = AutoScalingConfig(
            min_replicas=2,
            max_replicas=20,
            scaling_policy=ScalingPolicy.MEMORY_UTILIZATION,
            custom_metrics=["http_requests", "queue_size"]
        )

        assert config.min_replicas == 2
        assert config.max_replicas == 20
        assert config.scaling_policy == ScalingPolicy.MEMORY_UTILIZATION
        assert config.custom_metrics == ["http_requests", "queue_size"]

    def test_post_init_with_string_policy(self):
        """测试后初始化处理字符串类型的scaling_policy"""
        config = AutoScalingConfig(scaling_policy="request_rate")

        assert config.scaling_policy == ScalingPolicy.REQUEST_RATE
        assert isinstance(config.scaling_policy, ScalingPolicy)

    def test_scaling_parameters_validation(self):
        """测试伸缩参数合理性"""
        config = AutoScalingConfig(
            min_replicas=1,
            max_replicas=50,
            target_cpu_utilization=75
        )

        assert config.min_replicas < config.max_replicas
        assert 0 <= config.target_cpu_utilization <= 100


class TestCloudNativeMonitoringConfig:
    """测试云原生监控配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = CloudNativeMonitoringConfig()

        assert config.prometheus_enabled is True
        assert config.grafana_enabled is True
        assert config.alerting_enabled is True
        assert config.log_aggregation is True
        assert config.custom_dashboards == []
        assert config.metrics_retention_days == 30
        assert config.enable_tracing is True
        assert config.custom_metrics == {}

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        config = CloudNativeMonitoringConfig(
            prometheus_enabled=False,
            grafana_enabled=False,
            metrics_retention_days=90,
            custom_dashboards=["dashboard1", "dashboard2"]
        )

        assert config.prometheus_enabled is False
        assert config.grafana_enabled is False
        assert config.metrics_retention_days == 90
        assert config.custom_dashboards == ["dashboard1", "dashboard2"]

    def test_post_init_none_handling(self):
        """测试后初始化处理None值"""
        config = CloudNativeMonitoringConfig()
        config.custom_dashboards = None
        config.custom_metrics = None

        config.__post_init__()

        assert config.custom_dashboards == []
        assert config.custom_metrics == {}

    def test_custom_metrics_operations(self):
        """测试自定义指标操作"""
        custom_metrics = {
            "http_request_duration": {"type": "histogram", "buckets": [0.1, 0.5, 1.0]},
            "database_connections": {"type": "gauge"}
        }

        config = CloudNativeMonitoringConfig(custom_metrics=custom_metrics)
        assert config.custom_metrics == custom_metrics


class TestCloudNativeConfigsIntegration:
    """测试云原生配置集成"""

    def test_all_configs_creation(self):
        """测试所有配置类的创建"""
        mesh_config = ServiceMeshConfig()
        multi_cloud_config = MultiCloudConfig()
        scaling_config = AutoScalingConfig()
        monitoring_config = CloudNativeMonitoringConfig()

        # 验证所有配置都能正常创建
        assert mesh_config is not None
        assert multi_cloud_config is not None
        assert scaling_config is not None
        assert monitoring_config is not None

        # 验证类型
        assert isinstance(mesh_config, ServiceMeshConfig)
        assert isinstance(multi_cloud_config, MultiCloudConfig)
        assert isinstance(scaling_config, AutoScalingConfig)
        assert isinstance(monitoring_config, CloudNativeMonitoringConfig)

    def test_enum_integration(self):
        """测试枚举与配置类的集成"""
        # 服务网格配置使用枚举
        mesh_config = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)
        assert mesh_config.mesh_type == ServiceMeshType.ISTIO

        # 多云配置使用枚举
        multi_config = MultiCloudConfig(
            primary_provider=CloudProvider.AWS,
            secondary_providers=[CloudProvider.AZURE, CloudProvider.GCP]
        )
        assert multi_config.primary_provider == CloudProvider.AWS
        assert CloudProvider.AZURE in multi_config.secondary_providers

        # 自动伸缩配置使用枚举
        scaling_config = AutoScalingConfig(scaling_policy=ScalingPolicy.CPU_UTILIZATION)
        assert scaling_config.scaling_policy == ScalingPolicy.CPU_UTILIZATION
