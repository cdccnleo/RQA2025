import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime
from unittest.mock import Mock

from src.data.distributed.cluster_manager import (
    ClusterManager,
    ClusterInfo,
    ClusterStatus
)


def test_cluster_info_defaults():
    """测试 ClusterInfo（默认值）"""
    info = ClusterInfo(
        cluster_id="test_id",
        name="test_cluster",
        status=ClusterStatus.ACTIVE,
        version="1.0.0",
        created_at=datetime.now(),
        node_count=0,
        active_nodes=0,
        total_cpu=0.0,
        total_memory=0.0,
        metadata={}
    )
    assert info.cluster_id == "test_id"
    assert info.name == "test_cluster"
    assert info.status == ClusterStatus.ACTIVE
    assert info.node_count == 0
    assert info.active_nodes == 0
    assert info.total_cpu == 0.0
    assert info.total_memory == 0.0
    assert info.metadata == {}


def test_cluster_manager_init_none_config():
    """测试 ClusterManager（初始化，None 配置）"""
    manager = ClusterManager(config=None)
    assert manager.config == {}
    assert manager.cluster_info.cluster_id is not None
    assert manager.cluster_info.name == "RQA2025-Cluster"
    assert manager.cluster_info.status == ClusterStatus.ACTIVE


def test_cluster_manager_init_empty_config():
    """测试 ClusterManager（初始化，空配置）"""
    manager = ClusterManager(config={})
    assert manager.config == {}
    assert manager.cluster_info.name == "RQA2025-Cluster"


def test_cluster_manager_init_custom_config():
    """测试 ClusterManager（初始化，自定义配置）"""
    manager = ClusterManager(config={'cluster_name': 'CustomCluster'})
    assert manager.cluster_info.name == "CustomCluster"


def test_cluster_manager_get_cluster_info():
    """测试 ClusterManager（获取集群信息）"""
    manager = ClusterManager()
    info = manager.get_cluster_info()
    assert "cluster_id" in info
    assert "name" in info
    assert "status" in info
    assert "version" in info
    assert "created_at" in info
    assert "node_count" in info
    assert "active_nodes" in info
    assert "total_cpu" in info
    assert "total_memory" in info
    assert "metadata" in info


def test_cluster_manager_register_node_empty_info():
    """测试 ClusterManager（注册节点，空信息）"""
    manager = ClusterManager()
    manager.register_node("node1", {})
    assert "node1" in manager.nodes
    assert manager.cluster_info.node_count == 1


def test_cluster_manager_register_node_with_cpu_memory():
    """测试 ClusterManager（注册节点，带CPU和内存）"""
    manager = ClusterManager()
    manager.register_node("node1", {'cpu_usage': 2.5, 'memory_usage': 8.0})
    assert manager.cluster_info.total_cpu == 2.5
    assert manager.cluster_info.total_memory == 8.0


def test_cluster_manager_register_node_duplicate():
    """测试 ClusterManager（注册节点，重复）"""
    manager = ClusterManager()
    manager.register_node("node1", {})
    manager.register_node("node1", {'cpu_usage': 2.5})
    # 重复注册会增加 node_count，但会覆盖节点信息
    assert manager.cluster_info.node_count == 2  # 每次注册都会增加计数
    assert manager.cluster_info.total_cpu == 2.5


def test_cluster_manager_unregister_node_nonexistent():
    """测试 ClusterManager（注销节点，不存在）"""
    manager = ClusterManager()
    manager.unregister_node("nonexistent")
    assert manager.cluster_info.node_count == 0


def test_cluster_manager_unregister_node_with_cpu_memory():
    """测试 ClusterManager（注销节点，带CPU和内存）"""
    manager = ClusterManager()
    manager.register_node("node1", {'cpu_usage': 2.5, 'memory_usage': 8.0})
    manager.unregister_node("node1")
    assert manager.cluster_info.total_cpu == 0.0
    assert manager.cluster_info.total_memory == 0.0
    assert manager.cluster_info.node_count == 0


def test_cluster_manager_update_cluster_status():
    """测试 ClusterManager（更新集群状态）"""
    manager = ClusterManager()
    manager.update_cluster_status(ClusterStatus.MAINTENANCE)
    assert manager.cluster_info.status == ClusterStatus.MAINTENANCE


def test_cluster_manager_get_node_list_empty():
    """测试 ClusterManager（获取节点列表，空）"""
    manager = ClusterManager()
    nodes = manager.get_node_list()
    assert nodes == []


def test_cluster_manager_get_node_list_with_nodes():
    """测试 ClusterManager（获取节点列表，有节点）"""
    manager = ClusterManager()
    manager.register_node("node1", {'name': 'Node1'})
    manager.register_node("node2", {'name': 'Node2'})
    nodes = manager.get_node_list()
    assert len(nodes) == 2


def test_cluster_manager_get_cluster_stats_empty():
    """测试 ClusterManager（获取集群统计信息，空）"""
    manager = ClusterManager()
    stats = manager.get_cluster_stats()
    assert stats['total_nodes'] == 0
    assert stats['active_nodes'] == 0
    assert stats['total_cpu'] == 0.0
    assert stats['total_memory'] == 0.0
    assert stats['average_cpu'] == 0.0
    assert stats['average_memory'] == 0.0


def test_cluster_manager_get_cluster_stats_with_nodes():
    """测试 ClusterManager（获取集群统计信息，有节点）"""
    manager = ClusterManager()
    manager.register_node("node1", {'cpu_usage': 2.0, 'memory_usage': 4.0})
    manager.register_node("node2", {'cpu_usage': 3.0, 'memory_usage': 6.0})
    stats = manager.get_cluster_stats()
    assert stats['total_nodes'] == 2
    assert stats['total_cpu'] == 5.0
    assert stats['total_memory'] == 10.0
    assert stats['average_cpu'] == 2.5
    assert stats['average_memory'] == 5.0


def test_cluster_manager_status_property():
    """测试 ClusterManager（状态属性）"""
    manager = ClusterManager()
    assert manager.status == "active"
    manager.update_cluster_status(ClusterStatus.INACTIVE)
    assert manager.status == "inactive"


def test_cluster_manager_add_node():
    """测试 ClusterManager（添加节点）"""
    manager = ClusterManager()
    result = manager.add_node("node1", {'name': 'Node1'})
    assert result is True
    assert "node1" in manager.nodes


def test_cluster_manager_remove_node_existing():
    """测试 ClusterManager（移除节点，存在）"""
    manager = ClusterManager()
    manager.register_node("node1", {})
    result = manager.remove_node("node1")
    assert result is True
    assert "node1" not in manager.nodes


def test_cluster_manager_remove_node_nonexistent():
    """测试 ClusterManager（移除节点，不存在）"""
    manager = ClusterManager()
    result = manager.remove_node("nonexistent")
    assert result is False


def test_cluster_manager_get_status():
    """测试 ClusterManager（获取状态）"""
    manager = ClusterManager()
    status = manager.get_status()
    assert "total_nodes" in status
    assert "active_nodes" in status
    assert "status" in status
    assert status['total_nodes'] == 0
    assert status['status'] == "active"

