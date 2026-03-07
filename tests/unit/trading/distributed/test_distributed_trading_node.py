# -*- coding: utf-8 -*-
"""
分布式交易节点单元测试
测试覆盖率目标: 85%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import threading
import uuid
import sys

# Mock基础设施模块以避免导入错误
sys.modules['src.infrastructure.logging.distributed_lock'] = MagicMock()
sys.modules['src.infrastructure.config.config_center'] = MagicMock()
sys.modules['src.infrastructure.logging.distributed_monitoring'] = MagicMock()

from src.trading.distributed.distributed_distributed_trading_node import (
    DistributedTradingNode,
    TradingNodeInfo,
    TradingTask
)


class TestDistributedTradingNode:
    """分布式交易节点测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.config = {
            'node_id': 'test_node_001',
            'host': 'localhost',
            'port': 8080,
            'distributed_lock': {},
            'config_center': {},
            'distributed_monitoring': {}
        }

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_init_default(self, mock_monitoring, mock_config, mock_lock):
        """测试默认初始化"""
        node = DistributedTradingNode(self.config)
        assert node.node_id == 'test_node_001'
        assert node.host == 'localhost'
        assert node.port == 8080
        assert isinstance(node.nodes, dict)
        assert isinstance(node.tasks, dict)
        assert hasattr(node, '_lock')  # 检查_lock属性存在

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_init_with_defaults(self, mock_monitoring, mock_config, mock_lock):
        """测试使用默认值初始化"""
        config = {}
        node = DistributedTradingNode(config)
        assert node.node_id is not None
        assert node.host == 'localhost'
        assert node.port == 8080

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_init_distributed_components(self, mock_monitoring, mock_config, mock_lock):
        """测试初始化分布式组件"""
        node = DistributedTradingNode(self.config)
        assert node.lock_manager is not None
        assert node.config_manager is not None
        assert node.monitoring_manager is not None

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_init_distributed_components_failure(self, mock_monitoring, mock_config, mock_lock):
        """测试分布式组件初始化失败"""
        mock_lock.side_effect = Exception("Lock init failed")
        with pytest.raises(Exception):
            DistributedTradingNode(self.config)

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_register_node_success(self, mock_monitoring, mock_config, mock_lock):
        """测试注册节点成功"""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_context = MagicMock()
        mock_lock_instance.acquire_lock.return_value.__enter__ = lambda x: None
        mock_lock_instance.acquire_lock.return_value.__exit__ = lambda x, y, z, w: None
        
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        node = DistributedTradingNode(self.config)
        node.lock_manager = mock_lock_instance
        node.config_manager = mock_config_instance
        
        result = node.register_node(['equity', 'futures'])
        assert result is True
        assert node.node_id in node.nodes

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_register_node_with_default_capabilities(self, mock_monitoring, mock_config, mock_lock):
        """测试使用默认能力注册节点"""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_context = MagicMock()
        mock_lock_instance.acquire_lock.return_value.__enter__ = lambda x: None
        mock_lock_instance.acquire_lock.return_value.__exit__ = lambda x, y, z, w: None
        
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        node = DistributedTradingNode(self.config)
        node.lock_manager = mock_lock_instance
        node.config_manager = mock_config_instance
        
        result = node.register_node()
        assert result is True
        node_info = node.nodes[node.node_id]
        assert node_info.capabilities == ['equity']

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_register_node_failure(self, mock_monitoring, mock_config, mock_lock):
        """测试注册节点失败"""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_instance.acquire_lock.side_effect = Exception("Lock failed")
        
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        node = DistributedTradingNode(self.config)
        node.lock_manager = mock_lock_instance
        node.config_manager = mock_config_instance
        
        result = node.register_node()
        assert result is False

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_discover_nodes_success(self, mock_monitoring, mock_config, mock_lock):
        """测试发现节点成功"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # 模拟配置中心返回的节点信息
        mock_config_instance.get_config.return_value = {
            'other_node_001': {
                'node_id': 'other_node_001',
                'host': '192.168.1.100',
                'port': 8081,
                'status': 'active',
                'capabilities': ['equity'],
                'load': 0.5,
                'last_heartbeat': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat()
            }
        }
        
        node = DistributedTradingNode(self.config)
        node.config_manager = mock_config_instance
        
        nodes = node.discover_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == 'other_node_001'

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_discover_nodes_empty(self, mock_monitoring, mock_config, mock_lock):
        """测试发现节点为空"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.get_config.return_value = None
        
        node = DistributedTradingNode(self.config)
        node.config_manager = mock_config_instance
        
        nodes = node.discover_nodes()
        assert len(nodes) == 0

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_discover_nodes_excludes_self(self, mock_monitoring, mock_config, mock_lock):
        """测试发现节点排除自己"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # 包含自己的节点信息
        mock_config_instance.get_config.return_value = {
            'test_node_001': {
                'node_id': 'test_node_001',
                'host': 'localhost',
                'port': 8080,
                'status': 'active',
                'capabilities': ['equity'],
                'load': 0.0,
                'last_heartbeat': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat()
            },
            'other_node_001': {
                'node_id': 'other_node_001',
                'host': '192.168.1.100',
                'port': 8081,
                'status': 'active',
                'capabilities': ['equity'],
                'load': 0.5,
                'last_heartbeat': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat()
            }
        }
        
        node = DistributedTradingNode(self.config)
        node.config_manager = mock_config_instance
        
        nodes = node.discover_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == 'other_node_001'

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_discover_nodes_failure(self, mock_monitoring, mock_config, mock_lock):
        """测试发现节点失败"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.get_config.side_effect = Exception("Config error")
        
        node = DistributedTradingNode(self.config)
        node.config_manager = mock_config_instance
        
        nodes = node.discover_nodes()
        assert len(nodes) == 0

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_submit_task_success(self, mock_monitoring, mock_config, mock_lock):
        """测试提交任务成功"""
        node = DistributedTradingNode(self.config)
        
        task_data = {'order_id': 'test_order_001', 'symbol': '000001.SZ'}
        task_id = node.submit_task('order_execution', task_data, priority=8)
        
        assert task_id is not None
        assert task_id in node.tasks
        task = node.tasks[task_id]
        assert task.task_type == 'order_execution'
        assert task.priority == 8
        assert task.data == task_data

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_submit_task_default_priority(self, mock_monitoring, mock_config, mock_lock):
        """测试使用默认优先级提交任务"""
        node = DistributedTradingNode(self.config)
        
        task_data = {'order_id': 'test_order_002'}
        task_id = node.submit_task('risk_check', task_data)
        
        assert task_id is not None
        task = node.tasks[task_id]
        assert task.priority == 5  # 默认优先级

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_submit_task_exception(self, mock_monitoring, mock_config, mock_lock):
        """测试提交任务异常处理"""
        node = DistributedTradingNode(self.config)
        
        # 模拟config_manager.set_config抛出异常
        node.config_manager.set_config.side_effect = Exception("Config error")
        
        task_data = {'order_id': 'test_order_003'}
        # submit_task在异常时会raise，所以应该捕获异常
        with pytest.raises(Exception):
            node.submit_task('order_execution', task_data)

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_trading_node_info_to_dict(self, mock_monitoring, mock_config, mock_lock):
        """测试TradingNodeInfo转换为字典"""
        node_info = TradingNodeInfo(
            node_id='test_node',
            host='localhost',
            port=8080,
            status='active',
            capabilities=['equity'],
            load=0.5,
            last_heartbeat=datetime.now(),
            created_at=datetime.now()
        )
        
        node_dict = node_info.to_dict()
        assert node_dict['node_id'] == 'test_node'
        assert node_dict['host'] == 'localhost'
        assert node_dict['port'] == 8080
        assert node_dict['status'] == 'active'
        assert node_dict['capabilities'] == ['equity']
        assert node_dict['load'] == 0.5
        assert 'last_heartbeat' in node_dict
        assert 'created_at' in node_dict

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_trading_task_to_dict(self, mock_monitoring, mock_config, mock_lock):
        """测试TradingTask转换为字典"""
        task = TradingTask(
            task_id='test_task_001',
            task_type='order_execution',
            priority=8,
            data={'order_id': 'test_order'},
            created_at=datetime.now(),
            assigned_node='test_node',
            status='pending'
        )
        
        task_dict = task.to_dict()
        assert task_dict['task_id'] == 'test_task_001'
        assert task_dict['task_type'] == 'order_execution'
        assert task_dict['priority'] == 8
        assert task_dict['data'] == {'order_id': 'test_order'}
        assert task_dict['assigned_node'] == 'test_node'
        assert task_dict['status'] == 'pending'
        assert 'created_at' in task_dict

    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedLockManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.ConfigCenterManager')
    @patch('src.trading.distributed.distributed_distributed_trading_node.DistributedMonitoringManager')
    def test_trading_task_default_status(self, mock_monitoring, mock_config, mock_lock):
        """测试TradingTask默认状态"""
        task = TradingTask(
            task_id='test_task_002',
            task_type='risk_check',
            priority=5,
            data={},
            created_at=datetime.now()
        )
        
        assert task.status == 'pending'
        assert task.assigned_node is None

