# -*- coding: utf-8 -*-
"""
集群管理器单元测试
测试 ClusterManager 的核心功能
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

from src.distributed.coordinator.cluster_manager import ClusterManager
from src.distributed.coordinator.models import NodeInfo, NodeStatus, ClusterStats


class TestClusterManager:
    """集群管理器测试"""

    def test_cluster_manager_initialization(self):
        """测试集群管理器初始化"""
        manager = ClusterManager()

        assert hasattr(manager, 'nodes')
        assert hasattr(manager, 'stats')
        assert hasattr(manager, '_lock')
        assert isinstance(manager.nodes, dict)
        assert isinstance(manager.stats, ClusterStats)

    def test_register_node(self):
        """测试注册节点"""
        manager = ClusterManager()

        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        result = manager.register_node(node_info)

        assert result is True
        assert "node_001" in manager.nodes
        assert manager.nodes["node_001"] == node_info

    def test_register_duplicate_node(self):
        """测试注册重复节点"""
        manager = ClusterManager()

        node_info1 = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        node_info2 = NodeInfo(
            node_id="node_001",
            hostname="host2",
            ip_address="192.168.1.2",
            port=8081,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        # 注册第一个节点
        result1 = manager.register_node(node_info1)
        assert result1 is True

        # 注册重复节点（应该更新）
        result2 = manager.register_node(node_info2)
        assert result2 is True
        assert manager.nodes["node_001"].hostname == "host2"

    def test_unregister_node(self):
        """测试注销节点"""
        manager = ClusterManager()

        # 先注册节点
        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(node_info)

        # 注销节点
        result = manager.unregister_node("node_001")

        assert result is True
        assert "node_001" not in manager.nodes

    def test_unregister_nonexistent_node(self):
        """测试注销不存在的节点"""
        manager = ClusterManager()

        result = manager.unregister_node("nonexistent_node")

        assert result is False

    def test_get_node(self):
        """测试获取节点"""
        manager = ClusterManager()

        # 添加节点
        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(node_info)

        # 获取节点
        retrieved_node = manager.get_node("node_001")
        assert retrieved_node == node_info

        # 获取不存在的节点
        nonexistent_node = manager.get_node("nonexistent")
        assert nonexistent_node is None

    def test_get_all_nodes(self):
        """测试获取所有节点"""
        manager = ClusterManager()

        # 添加多个节点
        node_info1 = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        node_info2 = NodeInfo(
            node_id="node_002",
            hostname="host2",
            ip_address="192.168.1.2",
            port=8081,
            status=NodeStatus.INACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        manager.register_node(node_info1)
        manager.register_node(node_info2)

        all_nodes = manager.get_all_nodes()

        assert isinstance(all_nodes, dict)
        assert len(all_nodes) == 2
        assert "node_001" in all_nodes
        assert "node_002" in all_nodes

    def test_get_available_nodes(self):
        """测试获取可用节点"""
        manager = ClusterManager()

        # 添加节点（一个活跃，一个非活跃）
        active_node = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        inactive_node = NodeInfo(
            node_id="node_002",
            hostname="host2",
            ip_address="192.168.1.2",
            port=8081,
            status=NodeStatus.INACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        manager.register_node(active_node)
        manager.register_node(inactive_node)

        available_nodes = manager.get_available_nodes()

        assert isinstance(available_nodes, dict)
        assert len(available_nodes) == 1
        assert "node_001" in available_nodes
        assert "node_002" not in available_nodes

    def test_update_node_status(self):
        """测试更新节点状态"""
        manager = ClusterManager()

        # 添加节点
        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(node_info)

        # 更新状态
        result = manager.update_node_status("node_001", NodeStatus.INACTIVE)

        assert result is True
        assert manager.nodes["node_001"].status == NodeStatus.INACTIVE

    def test_update_node_load(self):
        """测试更新节点负载"""
        manager = ClusterManager()

        # 添加节点
        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(node_info)

        # 更新负载
        result = manager.update_node_load("node_001", 0.75)

        assert result is True
        assert manager.nodes["node_001"].load_factor == 0.75

    def test_update_node_heartbeat(self):
        """测试更新节点心跳"""
        manager = ClusterManager()

        # 添加节点
        node_info = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(node_info)

        original_heartbeat = manager.nodes["node_001"].last_heartbeat

        # 更新心跳
        result = manager.update_node_heartbeat("node_001")

        assert result is True
        assert manager.nodes["node_001"].last_heartbeat > original_heartbeat

    def test_check_node_health(self):
        """测试检查节点健康状态"""
        manager = ClusterManager()

        # 添加活跃节点
        active_node = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        manager.register_node(active_node)

        # 检查健康状态
        health = manager.check_node_health("node_001")

        assert health is True

    def test_get_cluster_stats(self):
        """测试获取集群统计"""
        manager = ClusterManager()

        # 添加节点
        node_info1 = NodeInfo(
            node_id="node_001",
            hostname="host1",
            ip_address="192.168.1.1",
            port=8080,
            status=NodeStatus.ACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        node_info2 = NodeInfo(
            node_id="node_002",
            hostname="host2",
            ip_address="192.168.1.2",
            port=8081,
            status=NodeStatus.INACTIVE,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        manager.register_node(node_info1)
        manager.register_node(node_info2)

        stats = manager.get_cluster_stats()

        assert isinstance(stats, ClusterStats)
        assert stats.total_nodes == 2
        assert stats.active_nodes == 1
        assert stats.inactive_nodes == 1

    def test_get_cluster_status(self):
        """测试获取集群状态"""
        manager = ClusterManager()

        status = manager.get_cluster_status()

        assert isinstance(status, dict)
        assert "total_nodes" in status
        assert "active_nodes" in status
        assert "inactive_nodes" in status
        assert "cluster_health" in status
