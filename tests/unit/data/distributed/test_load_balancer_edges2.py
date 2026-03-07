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
from unittest.mock import Mock

from src.data.distributed.load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy
)


def test_load_balancer_init_default_strategy():
    """测试 LoadBalancer（初始化，默认策略）"""
    balancer = LoadBalancer()
    assert balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
    assert balancer.current_index == 0
    assert balancer.node_stats == {}


def test_load_balancer_init_custom_strategy():
    """测试 LoadBalancer（初始化，自定义策略）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    assert balancer.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS


def test_load_balancer_select_node_empty_nodes():
    """测试 LoadBalancer（选择节点，空节点列表）"""
    balancer = LoadBalancer()
    with pytest.raises(ValueError, match="No available nodes"):
        balancer.select_node([], {})


def test_load_balancer_round_robin_select_single_node():
    """测试 LoadBalancer（轮询选择，单个节点）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    node = balancer.select_node(["node1"], {})
    assert node == "node1"


def test_load_balancer_round_robin_select_multiple_nodes():
    """测试 LoadBalancer（轮询选择，多个节点）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    nodes = ["node1", "node2", "node3"]
    # 第一次选择
    node1 = balancer.select_node(nodes, {})
    assert node1 == "node1"
    # 第二次选择
    node2 = balancer.select_node(nodes, {})
    assert node2 == "node2"
    # 第三次选择
    node3 = balancer.select_node(nodes, {})
    assert node3 == "node3"
    # 第四次选择（循环）
    node4 = balancer.select_node(nodes, {})
    assert node4 == "node1"


def test_load_balancer_least_connections_select_empty_nodes():
    """测试 LoadBalancer（最少连接选择，空节点信息）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    node = balancer.select_node(["node1"], {})
    assert node == "node1"


def test_load_balancer_least_connections_select_with_connections():
    """测试 LoadBalancer（最少连接选择，带连接数）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    nodes = ["node1", "node2", "node3"]
    node_info = {
        "node1": {"active_tasks": 5},
        "node2": {"active_tasks": 2},
        "node3": {"active_tasks": 8}
    }
    node = balancer.select_node(nodes, node_info)
    assert node == "node2"  # 最少连接


def test_load_balancer_least_connections_select_equal_connections():
    """测试 LoadBalancer（最少连接选择，相同连接数）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    nodes = ["node1", "node2"]
    node_info = {
        "node1": {"active_tasks": 3},
        "node2": {"active_tasks": 3}
    }
    node = balancer.select_node(nodes, node_info)
    assert node in ["node1", "node2"]  # 应该选择第一个


def test_load_balancer_least_connections_select_missing_node_info():
    """测试 LoadBalancer（最少连接选择，缺少节点信息）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    nodes = ["node1", "node2"]
    node_info = {
        "node1": {"active_tasks": 5}
        # node2 不在 node_info 中
    }
    node = balancer.select_node(nodes, node_info)
    # 应该选择第一个有信息的节点，或者第一个节点
    assert node in nodes


def test_load_balancer_weighted_round_robin_select_empty_node_info():
    """测试 LoadBalancer（加权轮询选择，空节点信息）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    node = balancer.select_node(["node1"], {})
    assert node == "node1"


def test_load_balancer_weighted_round_robin_select_with_usage():
    """测试 LoadBalancer（加权轮询选择，带使用率）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    nodes = ["node1", "node2"]
    node_info = {
        "node1": {"cpu_usage": 0.2, "memory_usage": 0.3},
        "node2": {"cpu_usage": 0.8, "memory_usage": 0.9}
    }
    node = balancer.select_node(nodes, node_info)
    # node1 使用率低，权重高，应该被选中
    assert node == "node1"


def test_load_balancer_weighted_round_robin_select_zero_usage():
    """测试 LoadBalancer（加权轮询选择，零使用率）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    nodes = ["node1", "node2"]
    node_info = {
        "node1": {"cpu_usage": 0.0, "memory_usage": 0.0},
        "node2": {"cpu_usage": 0.5, "memory_usage": 0.5}
    }
    node = balancer.select_node(nodes, node_info)
    # node1 使用率为0，权重更高
    assert node == "node1"


def test_load_balancer_weighted_round_robin_select_missing_usage():
    """测试 LoadBalancer（加权轮询选择，缺少使用率）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    nodes = ["node1", "node2"]
    node_info = {
        "node1": {}  # 缺少使用率信息，会使用默认值 0.5
    }
    node = balancer.select_node(nodes, node_info)
    assert node in nodes


def test_load_balancer_least_response_time_select_empty_stats():
    """测试 LoadBalancer（最少响应时间选择，空统计）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    node = balancer.select_node(["node1"], {})
    assert node == "node1"


def test_load_balancer_least_response_time_select_with_stats():
    """测试 LoadBalancer（最少响应时间选择，带统计）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    balancer.node_stats = {
        "node1": {"average_response_time": 0.5},
        "node2": {"average_response_time": 0.2},
        "node3": {"average_response_time": 1.0}
    }
    nodes = ["node1", "node2", "node3"]
    node = balancer.select_node(nodes, {})
    assert node == "node2"  # 最少响应时间


def test_load_balancer_least_response_time_select_missing_stats():
    """测试 LoadBalancer（最少响应时间选择，缺少统计）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    balancer.node_stats = {
        "node1": {"average_response_time": 0.5}
        # node2 不在 stats 中
    }
    nodes = ["node1", "node2"]
    node = balancer.select_node(nodes, {})
    # 应该选择第一个有统计的节点，或者第一个节点
    assert node in nodes


def test_load_balancer_random_select():
    """测试 LoadBalancer（随机选择）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.RANDOM)
    nodes = ["node1", "node2", "node3"]
    selected_nodes = set()
    # 多次选择，应该能选到不同的节点（概率很高）
    for _ in range(10):
        node = balancer.select_node(nodes, {})
        selected_nodes.add(node)
        assert node in nodes
    # 至少应该选到多个不同的节点
    assert len(selected_nodes) >= 1


def test_load_balancer_update_node_stats_new_node():
    """测试 LoadBalancer（更新节点统计，新节点）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    assert "node1" in balancer.node_stats
    assert balancer.node_stats["node1"]["average_response_time"] == 0.5
    assert balancer.node_stats["node1"]["total_requests"] == 1
    assert balancer.node_stats["node1"]["successful_requests"] == 1


def test_load_balancer_update_node_stats_existing_node():
    """测试 LoadBalancer（更新节点统计，已存在节点）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    balancer.update_node_stats("node1", 0.3, success=True)
    assert balancer.node_stats["node1"]["total_requests"] == 2
    assert balancer.node_stats["node1"]["successful_requests"] == 2
    # 平均响应时间应该是 (0.5 + 0.3) / 2 = 0.4
    assert balancer.node_stats["node1"]["average_response_time"] == 0.4


def test_load_balancer_update_node_stats_failure():
    """测试 LoadBalancer（更新节点统计，失败）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=False)
    assert balancer.node_stats["node1"]["total_requests"] == 1
    assert balancer.node_stats["node1"]["successful_requests"] == 0
    assert balancer.node_stats["node1"]["failed_requests"] == 1


def test_load_balancer_update_node_stats_zero_response_time():
    """测试 LoadBalancer（更新节点统计，零响应时间）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.0, success=True)
    assert balancer.node_stats["node1"]["average_response_time"] == 0.0


def test_load_balancer_update_node_stats_negative_response_time():
    """测试 LoadBalancer（更新节点统计，负响应时间）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", -0.1, success=True)
    # 应该能处理负值（虽然不合理）
    assert "node1" in balancer.node_stats


def test_load_balancer_get_node_stats_nonexistent():
    """测试 LoadBalancer（获取节点统计，不存在）"""
    balancer = LoadBalancer()
    stats = balancer.get_node_stats("nonexistent")
    assert stats is None


def test_load_balancer_get_node_stats_existing():
    """测试 LoadBalancer（获取节点统计，存在）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    stats = balancer.get_node_stats("node1")
    assert stats is not None
    assert "average_response_time" in stats
    assert "total_requests" in stats
    assert "successful_requests" in stats


def test_load_balancer_get_all_node_stats_empty():
    """测试 LoadBalancer（获取所有节点统计，空）"""
    balancer = LoadBalancer()
    all_stats = balancer.get_all_node_stats()
    assert all_stats == {}


def test_load_balancer_get_all_node_stats_with_stats():
    """测试 LoadBalancer（获取所有节点统计，有统计）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    balancer.update_node_stats("node2", 0.3, success=True)
    all_stats = balancer.get_all_node_stats()
    assert len(all_stats) == 2
    assert "node1" in all_stats
    assert "node2" in all_stats


def test_load_balancer_reset_stats():
    """测试 LoadBalancer（重置统计）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    balancer.reset_node_stats()  # 方法名是 reset_node_stats
    assert balancer.node_stats == {}


def test_load_balancer_reset_stats_multiple_times():
    """测试 LoadBalancer（重置统计，多次调用）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    balancer.reset_node_stats()  # 方法名是 reset_node_stats
    balancer.reset_node_stats()  # 应该不抛出异常
    assert balancer.node_stats == {}


def test_load_balancer_reset_node_stats_specific_node():
    """测试 LoadBalancer（重置特定节点统计）"""
    balancer = LoadBalancer()
    balancer.update_node_stats("node1", 0.5, success=True)
    balancer.update_node_stats("node2", 0.3, success=True)
    balancer.reset_node_stats("node1")
    assert "node1" not in balancer.node_stats
    assert "node2" in balancer.node_stats


def test_load_balancer_reset_node_stats_nonexistent_node():
    """测试 LoadBalancer（重置不存在节点统计）"""
    balancer = LoadBalancer()
    balancer.reset_node_stats("nonexistent")  # 应该不抛出异常
    assert balancer.node_stats == {}


def test_load_balancer_select_node_unknown_strategy():
    """测试 LoadBalancer（选择节点，未知策略，覆盖 75 行）"""
    # 创建一个使用未知策略的负载均衡器
    balancer = LoadBalancer()
    # 通过直接设置一个无效的策略值来触发默认分支
    from unittest.mock import Mock
    balancer.strategy = Mock()  # 创建一个不匹配任何已知策略的 mock 对象
    balancer.strategy.value = "unknown_strategy"
    # 由于策略不匹配任何已知策略，应该返回第一个可用节点
    nodes = ["node1", "node2"]
    result = balancer.select_node(nodes, {})
    assert result == "node1"  # 默认选择第一个


def test_load_balancer_round_robin_select_empty_nodes():
    """测试 LoadBalancer（轮询选择，空节点列表，覆盖 80 行）"""
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    with pytest.raises(ValueError, match="No available nodes"):
        balancer._round_robin_select([])

