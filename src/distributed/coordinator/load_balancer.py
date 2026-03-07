"""
负载均衡器

负责节点选择和负载均衡策略实现。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Dict, List, Any, Optional
import secrets

from .models import DistributedTask, NodeInfo, NodeStatus, TaskPriority

logger = logging.getLogger(__name__)


class LoadBalancer:
    """
    高级负载均衡器

    支持多种负载均衡策略：
    - adaptive: 自适应
    - round_robin: 轮询
    - weighted_round_robin: 加权轮询
    - least_loaded: 最小负载
    - capability_based: 能力匹配
    - random: 随机
    - response_time: 响应时间
    """

    def __init__(self):
        # adaptive, round_robin, weighted_round_robin, least_loaded, capability_based, random, response_time
        self.strategy = "adaptive"
        self.round_robin_index = 0  # 轮询索引
        self.node_weights = {}  # 节点权重
        self.response_times = {}  # 节点响应时间历史
        self.max_response_time_history = 100  # 最大响应时间历史记录数
        self.adaptive_threshold = 0.8  # 自适应切换阈值

    def select_node(self, task: DistributedTask, available_nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """选择执行任务的节点"""
        if not available_nodes:
            return None

        # 过滤出符合条件的节点
        eligible_nodes = self._filter_eligible_nodes(task, available_nodes)
        if not eligible_nodes:
            return None

        # 自适应策略选择
        if self.strategy == "adaptive":
            return self._adaptive_selection(task, eligible_nodes)

        # 具体策略选择
        strategy_map = {
            "round_robin": self._round_robin_selection,
            "weighted_round_robin": self._weighted_round_robin_selection,
            "least_loaded": self._least_loaded_selection,
            "capability_based": self._capability_based_selection,
            "random": self._random_selection,
            "response_time": self._response_time_selection
        }

        selector = strategy_map.get(self.strategy, self._round_robin_selection)

        if self.strategy in ["round_robin", "weighted_round_robin", "random"]:
            return selector(list(eligible_nodes.keys()), eligible_nodes)
        else:
            return selector(task, eligible_nodes)

    def _filter_eligible_nodes(self, task: DistributedTask, nodes: Dict[str, NodeInfo]) -> Dict[str, NodeInfo]:
        """过滤符合条件的节点"""
        eligible = {}

        for node_id, node in nodes.items():
            if node.status != NodeStatus.ONLINE:
                continue

            # 检查节点负载
            if node.load_factor > 0.9:  # 负载超过90%的节点不分配任务
                continue

            # 检查任务依赖
            if not self._check_task_dependencies(task, node):
                continue

            eligible[node_id] = node

        return eligible

    def _round_robin_selection(self, node_ids: List[str], nodes: Optional[Dict[str, NodeInfo]] = None) -> str:
        """轮询选择"""
        if not node_ids:
            return None

        # 使用内部索引实现真正的轮询
        index = self.round_robin_index % len(node_ids)
        selected_node = node_ids[index]
        self.round_robin_index += 1
        return selected_node

    def _least_loaded_selection(self, task: Optional[DistributedTask] = None, nodes: Optional[Dict[str, NodeInfo]] = None) -> str:
        """选择负载最小的节点"""
        if nodes is None:
            return None
        return min(nodes.items(), key=lambda x: x[1].load_factor)[0]

    def _capability_based_selection(self, task: DistributedTask, nodes: Dict[str, NodeInfo]) -> str:
        """基于能力的节点选择"""
        # 根据任务类型选择最合适的节点
        task_requirements = self._get_task_requirements(task)

        best_node = None
        best_score = 0

        for node_id, node in nodes.items():
            score = self._calculate_node_score(node, task_requirements)
            if score > best_score:
                best_score = score
                best_node = node_id

        return best_node

    def _check_task_dependencies(self, task: DistributedTask, node: NodeInfo) -> bool:
        """检查任务依赖"""
        # 这里应该实现任务依赖检查逻辑
        # 例如：检查依赖的任务是否在此节点上运行
        return True

    def _get_task_requirements(self, task: DistributedTask) -> Dict[str, Any]:
        """获取任务要求"""
        requirements = {
            'cpu_required': 1,
            'memory_required': 1.0,
            'gpu_required': False,
            'capabilities_required': set()
        }

        # 根据任务类型设置要求
        if task.task_type == 'ml_training':
            requirements.update({
                'cpu_required': 4,
                'memory_required': 8.0,
                'gpu_required': True
            })
        elif task.task_type == 'data_processing':
            requirements.update({
                'cpu_required': 2,
                'memory_required': 4.0
            })

        return requirements

    def _calculate_node_score(self, node: NodeInfo, requirements: Dict[str, Any]) -> float:
        """计算节点匹配分数"""
        score = 0.0

        # CPU分数
        cpu_score = min(node.cpu_cores / requirements['cpu_required'], 1.0)
        score += cpu_score * 0.3

        # 内存分数
        memory_score = min(node.memory_gb / requirements['memory_required'], 1.0)
        score += memory_score * 0.3

        # GPU分数
        if requirements['gpu_required']:
            gpu_score = 1.0 if node.gpu_devices else 0.0
        else:
            gpu_score = 0.5  # 如果不需要GPU，给中性分数
        score += gpu_score * 0.2

        # 负载分数（负载越低分数越高）
        load_score = 1.0 - node.load_factor
        score += load_score * 0.2

        return score

    def _adaptive_selection(self, task: DistributedTask, eligible_nodes: Dict[str, NodeInfo]) -> Optional[str]:
        """自适应负载均衡策略"""
        if not eligible_nodes:
            return None

        # 计算系统负载状态
        avg_load = sum(node.load_factor for node in eligible_nodes.values()) / len(eligible_nodes)

        # 根据负载状态选择策略
        if avg_load > self.adaptive_threshold:
            # 高负载时，使用最小负载策略
            return self._least_loaded_selection(task, eligible_nodes)
        elif task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            # 高优先级任务使用能力匹配
            return self._capability_based_selection(task, eligible_nodes)
        else:
            # 正常负载使用轮询
            return self._round_robin_selection(list(eligible_nodes.keys()), eligible_nodes)

    def _weighted_round_robin_selection(self, node_ids: List[str], nodes: Dict[str, NodeInfo]) -> str:
        """加权轮询选择"""
        if not node_ids:
            return None

        # 计算节点权重（基于CPU核心数和可用内存）
        weights = {}
        for node_id in node_ids:
            node = nodes[node_id]
            # 权重 = CPU核心数 + 内存GB数 + GPU数量
            weight = node.cpu_cores + node.memory_gb + len(node.gpu_devices)
            weights[node_id] = max(weight, 1)  # 最小权重为1

        # 使用加权轮询算法
        total_weight = sum(weights.values())
        if total_weight == 0:
            return node_ids[0]

        # 找到当前应该选择的节点
        current_weight_sum = 0
        for node_id in node_ids:
            current_weight_sum += weights[node_id]
            if current_weight_sum >= (self.round_robin_index % total_weight) + 1:
                self.round_robin_index += 1
                return node_id

        # 默认返回第一个
        self.round_robin_index += 1
        return node_ids[0]

    def _random_selection(self, node_ids: List[str], nodes: Dict[str, NodeInfo]) -> str:
        """随机选择"""
        return secrets.choice(node_ids)

    def _response_time_selection(self, task: DistributedTask, eligible_nodes: Dict[str, NodeInfo]) -> str:
        """基于响应时间的选择"""
        if not eligible_nodes:
            return None

        # 计算每个节点的平均响应时间
        node_scores = {}
        for node_id, node in eligible_nodes.items():
            avg_response_time = self._get_average_response_time(node_id)
            # 响应时间越短分数越高（0-1之间）
            score = 1.0 / (1.0 + avg_response_time) if avg_response_time > 0 else 1.0
            # 考虑负载因子
            load_penalty = node.load_factor
            final_score = score * (1 - load_penalty)
            node_scores[node_id] = final_score

        # 选择分数最高的节点
        return max(node_scores.items(), key=lambda x: x[1])[0]

    def record_response_time(self, node_id: str, response_time: float):
        """记录节点响应时间"""
        if node_id not in self.response_times:
            self.response_times[node_id] = []

        self.response_times[node_id].append(response_time)

        # 保持历史记录数量限制
        if len(self.response_times[node_id]) > self.max_response_time_history:
            self.response_times[node_id].pop(0)

    def _get_average_response_time(self, node_id: str) -> float:
        """获取节点的平均响应时间"""
        if node_id not in self.response_times or not self.response_times[node_id]:
            return 1.0  # 默认1秒

        return sum(self.response_times[node_id]) / len(self.response_times[node_id])

    def update_node_weight(self, node_id: str, weight: int):
        """更新节点权重"""
        self.node_weights[node_id] = max(weight, 1)

    def get_strategy_stats(self) -> Dict[str, Any]:
        """获取负载均衡策略统计信息"""
        return {
            "current_strategy": self.strategy,
            "round_robin_index": self.round_robin_index,
            "node_weights": self.node_weights.copy(),
            "response_time_nodes": list(self.response_times.keys()),
            "adaptive_threshold": self.adaptive_threshold
        }


__all__ = ['LoadBalancer']

