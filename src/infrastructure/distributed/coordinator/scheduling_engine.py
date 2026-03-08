"""
调度引擎

负责任务调度策略和算法实现。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import DistributedTask, NodeInfo, TaskStatus, TaskPriority, NodeStatus
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


class SchedulingEngine:
    """
    调度引擎

    负责任务调度策略和算法实现
    """

    def __init__(self):
        self.scheduling_history = []
        self.scheduling_policies = {}
        self.default_policy = 'round_robin'

        # 调度策略注册
        self.scheduling_policies['round_robin'] = self._round_robin_schedule
        self.scheduling_policies['least_loaded'] = self._least_loaded_schedule
        self.scheduling_policies['capability_based'] = self._capability_based_schedule
        self.scheduling_policies['priority_first'] = self._priority_first_schedule

        logger.info("调度引擎初始化完成")

    def schedule_task(self,
                      task: DistributedTask,
                      nodes: Dict[str, NodeInfo],
                      policy: str = None) -> Optional[str]:
        """
        调度任务到节点

        Args:
            task: 待调度任务
            nodes: 可用节点字典
            policy: 调度策略

        Returns:
            选中的节点ID，如果无法调度则返回None
        """
        if not nodes:
            logger.warning(f"无可用节点，任务 {task.task_id} 调度失败")
            return None

        # 选择调度策略
        policy = policy or self.default_policy
        if policy not in self.scheduling_policies:
            logger.warning(f"未知调度策略 {policy}，使用默认策略")
            policy = self.default_policy

        # 执行调度
        scheduling_func = self.scheduling_policies[policy]
        selected_node = scheduling_func(task, nodes)

        # 记录调度历史
        self._record_scheduling(task, selected_node, policy)

        return selected_node

    def _round_robin_schedule(self,
                              task: DistributedTask,
                              nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """轮询调度"""
        # 获取在线节点
        online_nodes = [
            node_id for node_id, node in nodes.items()
            if node.status == NodeStatus.ONLINE
        ]

        if not online_nodes:
            return None

        # 轮询选择
        index = len(self.scheduling_history) % len(online_nodes)
        return online_nodes[index]

    def _least_loaded_schedule(self,
                               task: DistributedTask,
                               nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """最小负载调度"""
        # 获取在线节点
        online_nodes = {
            node_id: node for node_id, node in nodes.items()
            if node.status == NodeStatus.ONLINE
        }

        if not online_nodes:
            return None

        # 选择负载最小的节点
        return min(online_nodes.items(), key=lambda x: x[1].load_factor)[0]

    def _capability_based_schedule(self,
                                   task: DistributedTask,
                                   nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """基于能力的调度"""
        # 获取任务需求
        required_capabilities = task.data.get('required_capabilities', set())

        # 过滤符合能力要求的节点
        capable_nodes = {}
        for node_id, node in nodes.items():
            if node.status == NodeStatus.ONLINE:
                if not required_capabilities or required_capabilities.issubset(node.capabilities):
                    capable_nodes[node_id] = node

        if not capable_nodes:
            return None

        # 选择负载最小的符合要求的节点
        return min(capable_nodes.items(), key=lambda x: x[1].load_factor)[0]

    def _priority_first_schedule(self,
                                 task: DistributedTask,
                                 nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """优先级优先调度"""
        # 对于高优先级任务，选择性能最好的节点
        if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            # 选择CPU核心最多的节点
            online_nodes = {
                node_id: node for node_id, node in nodes.items()
                if node.status == NodeStatus.ONLINE
            }

            if not online_nodes:
                return None

            return max(online_nodes.items(), key=lambda x: x[1].cpu_cores)[0]

        # 对于普通优先级任务，使用最小负载调度
        return self._least_loaded_schedule(task, nodes)

    def _record_scheduling(self, task: DistributedTask, node_id: Optional[str], policy: str):
        """记录调度历史"""
        record = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'node_id': node_id,
            'policy': policy,
            'timestamp': datetime.now(),
            'priority': task.priority.value
        }

        self.scheduling_history.append(record)

        # 保持历史记录在合理范围内
        if len(self.scheduling_history) > 10000:
            self.scheduling_history = self.scheduling_history[-10000:]

    def get_scheduling_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        if not self.scheduling_history:
            return {}

        recent_history = self.scheduling_history[-1000:]

        policy_counts = {}
        for record in recent_history:
            policy = record['policy']
            policy_counts[policy] = policy_counts.get(policy, 0) + 1

        return {
            'total_schedulings': len(self.scheduling_history),
            'recent_schedulings': len(recent_history),
            'policy_distribution': policy_counts,
            'default_policy': self.default_policy
        }


__all__ = ['SchedulingEngine']

