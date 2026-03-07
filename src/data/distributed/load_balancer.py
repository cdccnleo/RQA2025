import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 负载均衡器

实现负载均衡功能：
- 轮询负载均衡
- 最少连接负载均衡
- 加权轮询负载均衡
- 响应时间负载均衡
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import time
import secrets


logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):

    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """
        初始化负载均衡器

        Args:
            strategy: 负载均衡策略
        """
        self.strategy = strategy
        self.current_index = 0
        self.node_stats: Dict[str, Dict[str, Any]] = {}

        logger.info(f"LoadBalancer initialized with strategy: {self.strategy.value}")

    def select_node(self, available_nodes: List[str], nodes: Dict[str, Any]) -> str:
        """
        选择节点

        Args:
            available_nodes: 可用节点列表
            nodes: 节点信息字典

        Returns:
            str: 选中的节点ID
        """
        if not available_nodes:
            raise ValueError("No available nodes")

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes, nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_nodes, nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(available_nodes, nodes)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_select(available_nodes)
        else:
            return available_nodes[0]  # 默认选择第一个

    def _round_robin_select(self, available_nodes: List[str]) -> str:
        """轮询选择"""
        if not available_nodes:
            raise ValueError("No available nodes")

        selected = available_nodes[self.current_index % len(available_nodes)]
        self.current_index += 1
        return selected

    def _least_connections_select(self, available_nodes: List[str],


                                  nodes: Dict[str, Any]) -> str:
        """最少连接选择"""
        min_connections = float('inf')
        selected_node = available_nodes[0]

        for node_id in available_nodes:
            if node_id in nodes:
                connections = nodes[node_id].get('active_tasks', 0)
                if connections < min_connections:
                    min_connections = connections
                    selected_node = node_id

        return selected_node

    def _weighted_round_robin_select(self, available_nodes: List[str],


                                     nodes: Dict[str, Any]) -> str:
        """加权轮询选择"""
        # 基于CPU和内存使用率计算权重
        node_weights = {}
        total_weight = 0

        for node_id in available_nodes:
            if node_id in nodes:
                node = nodes[node_id]
                # 权重 = 1 / (CPU使用率 + 内存使用率 + 0.1)
                cpu_usage = node.get('cpu_usage', 0.5)
                memory_usage = node.get('memory_usage', 0.5)
                weight = 1 / (cpu_usage + memory_usage + 0.1)
                node_weights[node_id] = weight
                total_weight += weight

        if not node_weights:
            return available_nodes[0]

        # 选择权重最高的节点
        selected_node = max(node_weights.items(), key=lambda x: x[1])[0]
        return selected_node

    def _least_response_time_select(self, available_nodes: List[str],


                                    nodes: Dict[str, Any]) -> str:
        """最少响应时间选择"""
        min_response_time = float('inf')
        selected_node = available_nodes[0]

        for node_id in available_nodes:
            if node_id in self.node_stats:
                response_time = self.node_stats[node_id].get('average_response_time', 1.0)
                if response_time < min_response_time:
                    min_response_time = response_time
                    selected_node = node_id

        return selected_node

    def _random_select(self, available_nodes: List[str]) -> str:
        """随机选择"""
        return secrets.choice(available_nodes)

    def update_node_stats(self, node_id: str, response_time: float, success: bool = True):
        """更新节点统计信息"""
        if node_id not in self.node_stats:
            self.node_stats[node_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_response_time': 0.0,
                'average_response_time': 0.0,
                'last_update': time.time()
            }

        stats = self.node_stats[node_id]
        stats['total_requests'] += 1
        stats['total_response_time'] += response_time

        if success:
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1

        # 更新平均响应时间
        stats['average_response_time'] = stats['total_response_time'] / stats['total_requests']
        stats['last_update'] = time.time()

        logger.debug(
            f"Updated stats for node {node_id}: avg_response_time={stats['average_response_time']:.3f}s")

    def get_node_stats(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点统计信息"""
        return self.node_stats.get(node_id)

    def get_all_node_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有节点统计信息"""
        return self.node_stats.copy()

    def reset_node_stats(self, node_id: Optional[str] = None):
        """重置节点统计信息"""
        if node_id:
            if node_id in self.node_stats:
                del self.node_stats[node_id]
                logger.info(f"Reset stats for node {node_id}")
        else:
            self.node_stats.clear()
            logger.info("Reset all node stats")
