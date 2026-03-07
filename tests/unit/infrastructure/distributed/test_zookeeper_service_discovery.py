"""
测试ZooKeeper服务发现

覆盖 ZooKeeperServiceDiscovery 和相关类的功能
"""

import pytest
from src.infrastructure.distributed.zookeeper_service_discovery import (
    ZooKeeperConfig,
    ZooKeeperServiceDiscovery
)


class TestZooKeeperConfig:
    """ZooKeeperConfig 数据类测试"""

    def test_initialization_defaults(self):
        """测试默认初始化"""
        config = ZooKeeperConfig()

        assert config.hosts == "localhost:2181"
        assert config.base_path == "/services"
        assert config.session_timeout == 30_000
        assert config.connection_timeout == 10_000
        assert config.auth_scheme is None
        assert config.auth_data is None

    def test_initialization_custom(self):
        """测试自定义初始化"""
        config = ZooKeeperConfig(
            hosts="zk1:2181,zk2:2181,zk3:2181",
            base_path="/myapp/services",
            session_timeout=60_000,
            connection_timeout=15_000,
            auth_scheme="digest",
            auth_data="user:pass"
        )

        assert config.hosts == "zk1:2181,zk2:2181,zk3:2181"
        assert config.base_path == "/myapp/services"
        assert config.session_timeout == 60_000
        assert config.connection_timeout == 15_000
        assert config.auth_scheme == "digest"
        assert config.auth_data == "user:pass"


class TestZooKeeperServiceDiscovery:
    """ZooKeeperServiceDiscovery 类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        discovery = ZooKeeperServiceDiscovery()

        assert isinstance(discovery.config, ZooKeeperConfig)
        assert discovery.config.hosts == "localhost:2181"
        assert discovery._connected == False
        assert discovery._services == {}
        assert discovery._watchers == {}

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = ZooKeeperConfig(hosts="custom:2181", base_path="/custom")
        discovery = ZooKeeperServiceDiscovery(config)

        assert discovery.config == config
        assert discovery.config.hosts == "custom:2181"
        assert discovery.config.base_path == "/custom"

    def test_is_connected(self):
        """测试连接状态检查"""
        discovery = ZooKeeperServiceDiscovery()

        assert not discovery.is_connected()

        discovery._connected = True
        assert discovery.is_connected()