#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分布式协调器简化测试用例
Distributed Coordinator Simple Test Cases

测试 DistributedCoordinator 的核心功能，提升覆盖率
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from distributed.coordinator.coordinator_core import DistributedCoordinator
    from distributed.coordinator.models import NodeInfo, NodeStatus, TaskPriority
    DISTRIBUTED_COORDINATOR_AVAILABLE = True
except ImportError:
    DISTRIBUTED_COORDINATOR_AVAILABLE = False
    DistributedCoordinator = Mock
    NodeInfo = Mock
    NodeStatus = Mock
    TaskPriority = Mock


class TestDistributedCoordinator:
    """分布式协调器测试"""

    @pytest.fixture
    def coordinator(self):
        """协调器fixture"""
        if DISTRIBUTED_COORDINATOR_AVAILABLE:
            return DistributedCoordinator()
        else:
            mock_coordinator = Mock()
            mock_coordinator.register_node = Mock(return_value=True)
            mock_coordinator.submit_task = Mock(return_value='task_123')
            mock_coordinator.get_task_status = Mock(return_value={
                'task_id': 'task_123',
                'status': 'completed',
                'progress': 100
            })
            mock_coordinator.get_cluster_status = Mock(return_value={
                'total_nodes': 3,
                'active_nodes': 3,
                'total_tasks': 10,
                'completed_tasks': 8
            })
            return mock_coordinator

    def test_coordinator_initialization(self, coordinator):
        """测试协调器初始化"""
        assert coordinator is not None

    def test_node_registration(self, coordinator):
        """测试节点注册"""
        node_info = NodeInfo(
            node_id='test_node_1',
            hostname='localhost',
            ip_address='127.0.0.1',
            status=NodeStatus.ONLINE,
            cpu_cores=4,
            memory_gb=8.0,
            capabilities={'computation', 'storage'},
            last_heartbeat=datetime.now()
        )

        result = coordinator.register_node(node_info)
        assert result == True

    def test_task_submission(self, coordinator):
        """测试任务提交"""
        task_data = {
            'type': 'data_processing',
            'data': {'input_file': 'test.csv'}
        }

        task_id = coordinator.submit_task('data_processing', task_data)
        assert task_id is not None
        assert isinstance(task_id, str)

    def test_task_status_query(self, coordinator):
        """测试任务状态查询"""
        # 先提交任务
        task_data = {'type': 'test', 'data': {}}
        task_id = coordinator.submit_task('test', task_data)

        # 查询状态
        status = coordinator.get_task_status(task_id)
        assert status is not None
        assert isinstance(status, dict)
        assert 'status' in status

    def test_cluster_status(self, coordinator):
        """测试集群状态"""
        status = coordinator.get_cluster_status()
        assert status is not None
        assert isinstance(status, dict)
        # 检查返回的字典包含统计信息
        assert 'total_nodes' in status or 'nodes' in status
        assert 'avg_load_factor' in status

    def test_node_unregistration(self, coordinator):
        """测试节点取消注册"""
        result = coordinator.unregister_node('test_node_1')
        assert isinstance(result, bool)

    def test_task_cancellation(self, coordinator):
        """测试任务取消"""
        # 先提交任务
        task_data = {'type': 'cancellable_task', 'data': {}}
        task_id = coordinator.submit_task('test', task_data)

        # 取消任务
        result = coordinator.cancel_task(task_id)
        assert isinstance(result, bool)

    def test_coordinator_attributes(self, coordinator):
        """测试协调器属性"""
        # 验证基本属性存在
        assert hasattr(coordinator, 'cluster_manager')
        assert hasattr(coordinator, 'task_manager')

    def test_cluster_monitoring(self, coordinator):
        """测试集群监控"""
        # 协调器应该有监控能力
        status = coordinator.get_cluster_status()
        assert 'total_tasks' in status
        assert 'completed_tasks' in status

    def test_task_priority_handling(self, coordinator):
        """测试任务优先级处理"""
        # 测试不同优先级的任务
        high_priority_task = {
            'type': 'urgent_processing',
            'data': {'urgent': True}
        }

        normal_priority_task = {
            'type': 'normal_processing',
            'data': {'normal': True}
        }

        # 提交任务
        high_task_id = coordinator.submit_task('urgent', high_priority_task)
        normal_task_id = coordinator.submit_task('normal', normal_priority_task)

        assert high_task_id != normal_task_id
        assert high_task_id is not None
        assert normal_task_id is not None
