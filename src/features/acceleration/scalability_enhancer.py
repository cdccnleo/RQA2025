#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
扩展性增强模块
提供系统扩展能力和负载均衡功能
"""

import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ScalabilityEnhancer:

    """扩展性增强器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化扩展性增强器"""
        self.config = config or {}
        self.nodes = []
        self.load_balancer = LoadBalancer()
        self.auto_scaling = AutoScaling()

        logger.info("扩展性增强器初始化完成")

    def add_node(self, node_info: Dict[str, Any]):
        """添加节点"""
        self.nodes.append(node_info)
        self.load_balancer.update_nodes(self.nodes)
        logger.info(f"添加节点: {node_info.get('id', 'unknown')}")

    def remove_node(self, node_id: str):
        """移除节点"""
        self.nodes = [n for n in self.nodes if n.get('id') != node_id]
        self.load_balancer.update_nodes(self.nodes)
        logger.info(f"移除节点: {node_id}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes if n.get('status') == 'active']),
            "load_balancer_status": self.load_balancer.get_status(),
            "auto_scaling_status": self.auto_scaling.get_status()
        }

    def scale_out(self, count: int = 1):
        """扩容"""
        for i in range(count):
            new_node = self._create_node()
            self.add_node(new_node)
        logger.info(f"扩容 {count} 个节点")

    def scale_in(self, count: int = 1):
        """缩容"""
        for i in range(count):
            if self.nodes:
                node = self.nodes.pop()
                self.remove_node(node.get('id', ''))
        logger.info(f"缩容 {count} 个节点")

    def _create_node(self) -> Dict[str, Any]:
        """创建新节点"""
        return {
            'id': f"node_{int(time.time())}",
            'status': 'active',
            'capacity': 1000,
            'current_load': 0
        }


class LoadBalancer:

    """负载均衡器"""

    def __init__(self):
        """初始化负载均衡器"""
        self.nodes = []
        self.strategy = 'round_robin'
        self.current_index = 0

    def update_nodes(self, nodes: List[Dict[str, Any]]):
        """更新节点列表"""
        self.nodes = nodes

    def get_next_node(self) -> Optional[Dict[str, Any]]:
        """获取下一个节点"""
        if not self.nodes:
            return None

        if self.strategy == 'round_robin':
            node = self.nodes[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.nodes)
            return node
        elif self.strategy == 'least_connections':
            return min(self.nodes, key=lambda x: x.get('current_load', 0))
        else:
            return self.nodes[0]

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "strategy": self.strategy,
            "total_nodes": len(self.nodes),
            "current_index": self.current_index
        }


class AutoScaling:

    """自动扩缩容"""

    def __init__(self):
        """初始化自动扩缩容"""
        self.enabled = False
        self.min_nodes = 2
        self.max_nodes = 10
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 30.0

    def enable(self):
        """启用自动扩缩容"""
        self.enabled = True
        logger.info("自动扩缩容已启用")

    def disable(self):
        """禁用自动扩缩容"""
        self.enabled = False
        logger.info("自动扩缩容已禁用")

    def check_and_scale(self, current_load: float, node_count: int):
        """检查并执行扩缩容"""
        if not self.enabled:
            return

        avg_load = current_load / node_count if node_count > 0 else 0

        if avg_load > self.scale_up_threshold and node_count < self.max_nodes:
            logger.info(f"触发扩容，当前负载: {avg_load:.1f}%")
            return "scale_up"
        elif avg_load < self.scale_down_threshold and node_count > self.min_nodes:
            logger.info(f"触发缩容，当前负载: {avg_load:.1f}%")
            return "scale_down"

        return None

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "enabled": self.enabled,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold
        }
