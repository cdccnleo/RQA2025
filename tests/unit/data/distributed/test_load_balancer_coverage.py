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
from unittest.mock import Mock, patch

from src.data.distributed.load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy
)


def test_load_balancer_select_node_default_strategy():
    """测试select_node的默认策略（75行）"""
    # Create a balancer with an unknown strategy
    balancer = LoadBalancer()
    
    # Mock strategy to be an unknown value
    class UnknownStrategy:
        pass
    
    balancer.strategy = UnknownStrategy()
    
    available_nodes = ['node1', 'node2', 'node3']
    nodes = {}
    
    # Should return first node as default
    result = balancer.select_node(available_nodes, nodes)
    assert result == 'node1'


def test_load_balancer_round_robin_select_empty_nodes():
    """测试_round_robin_select中available_nodes为空（80行）"""
    balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
    
    # Should raise ValueError when available_nodes is empty
    with pytest.raises(ValueError, match="No available nodes"):
        balancer._round_robin_select([])

