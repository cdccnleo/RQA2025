#!/usr/bin/env python3
"""
分布式协调器

从coordinator.py拆分后的主入口模块。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

# 导入所有组件
from .models import (
    NodeStatus,
    TaskStatus,
    TaskPriority,
    NodeInfo,
    DistributedTask,
    ClusterStats
)
from .scheduling_engine import SchedulingEngine
from .queue_engine import QueueEngine
from .priority_engine import PriorityEngine
from .load_balancer import LoadBalancer
from .coordinator_core import (
    DistributedCoordinator,
    get_distributed_coordinator,
    submit_distributed_task,
    get_cluster_status
)

__all__ = [
    # 枚举和数据模型
    'NodeStatus',
    'TaskStatus',
    'TaskPriority',
    'NodeInfo',
    'DistributedTask',
    'ClusterStats',
    
    # 引擎组件
    'SchedulingEngine',
    'QueueEngine',
    'PriorityEngine',
    'LoadBalancer',
    
    # 主协调器
    'DistributedCoordinator',
    'get_distributed_coordinator',
    'submit_distributed_task',
    'get_cluster_status'
]
