#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""云服务配置测试"""

from src.infrastructure.config.environment.cloud_configs import (
    CloudNativeMonitoringConfig,
    MultiCloudConfig,
    ServiceMeshConfig
)


class TestCloudNativeMonitoringConfig:
    """测试云原生监控配置"""

    def test_init(self):
        """测试初始化"""
        config = CloudNativeMonitoringConfig()

        assert config.enabled is False
        assert config.endpoint == ""
        assert config.api_key == ""

    def test_attributes_modification(self):
        """测试属性修改"""
        config = CloudNativeMonitoringConfig()

        # 修改属性
        config.enabled = True
        config.endpoint = "https://monitoring.example.com"
        config.api_key = "secret-key-123"

        assert config.enabled is True
        assert config.endpoint == "https://monitoring.example.com"
        assert config.api_key == "secret-key-123"


class TestMultiCloudConfig:
    """测试多云配置"""

    def test_init(self):
        """测试初始化"""
        config = MultiCloudConfig()

        assert config.providers == []
        assert config.default_provider == ""

    def test_attributes_modification(self):
        """测试属性修改"""
        config = MultiCloudConfig()

        # 修改属性
        config.providers = ["aws", "azure", "gcp"]
        config.default_provider = "aws"

        assert config.providers == ["aws", "azure", "gcp"]
        assert config.default_provider == "aws"

    def test_providers_operations(self):
        """测试providers列表操作"""
        config = MultiCloudConfig()

        # 添加provider
        config.providers.append("aws")
        config.providers.append("azure")

        assert len(config.providers) == 2
        assert "aws" in config.providers
        assert "azure" in config.providers

        # 设置默认provider
        config.default_provider = "aws"
        assert config.default_provider == "aws"


class TestServiceMeshConfig:
    """测试服务网格配置"""

    def test_init(self):
        """测试初始化"""
        config = ServiceMeshConfig()

        assert config.enabled is False
        assert config.provider == ""
        assert config.namespace == ""

    def test_attributes_modification(self):
        """测试属性修改"""
        config = ServiceMeshConfig()

        # 修改属性
        config.enabled = True
        config.provider = "istio"
        config.namespace = "production"

        assert config.enabled is True
        assert config.provider == "istio"
        assert config.namespace == "production"

    def test_common_mesh_providers(self):
        """测试常见的服务网格提供商"""
        config = ServiceMeshConfig()

        # 测试Istio
        config.provider = "istio"
        assert config.provider == "istio"

        # 测试Linkerd
        config.provider = "linkerd"
        assert config.provider == "linkerd"

        # 测试Consul
        config.provider = "consul"
        assert config.provider == "consul"


class TestCloudConfigsIntegration:
    """测试云服务配置集成"""

    def test_all_configs_creation(self):
        """测试所有配置类的创建"""
        monitoring_config = CloudNativeMonitoringConfig()
        multi_cloud_config = MultiCloudConfig()
        mesh_config = ServiceMeshConfig()

        # 验证所有配置都能正常创建
        assert monitoring_config is not None
        assert multi_cloud_config is not None
        assert mesh_config is not None

        # 验证类型
        assert isinstance(monitoring_config, CloudNativeMonitoringConfig)
        assert isinstance(multi_cloud_config, MultiCloudConfig)
        assert isinstance(mesh_config, ServiceMeshConfig)

    def test_configs_independence(self):
        """测试配置之间的独立性"""
        config1 = CloudNativeMonitoringConfig()
        config2 = CloudNativeMonitoringConfig()

        # 修改第一个配置
        config1.enabled = True
        config1.endpoint = "endpoint1"

        # 第二个配置应该不受影响
        assert config2.enabled is False
        assert config2.endpoint == ""

        # 修改第二个配置
        config2.enabled = True
        config2.endpoint = "endpoint2"

        # 验证独立性
        assert config1.endpoint == "endpoint1"
        assert config2.endpoint == "endpoint2"
