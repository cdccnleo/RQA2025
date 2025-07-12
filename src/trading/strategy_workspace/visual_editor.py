import json
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """策略节点类型枚举"""
    DATA_SOURCE = "data_source"
    FEATURE = "feature"
    MODEL = "model"
    TRADE = "trade"
    RISK = "risk"

@dataclass
class StrategyNode:
    """策略节点基类"""
    node_id: str
    node_type: NodeType
    name: str
    params: Dict
    next_nodes: List[str]  # 下游节点ID列表

class VisualStrategyEditor:
    """策略可视化编辑器"""

    def __init__(self):
        self.nodes: Dict[str, StrategyNode] = {}
        self.connections: List[Dict] = []

    def add_node(self, node: StrategyNode) -> bool:
        """添加策略节点"""
        if node.node_id in self.nodes:
            logger.warning(f"Node {node.node_id} already exists")
            return False

        self.nodes[node.node_id] = node
        return True

    def remove_node(self, node_id: str) -> bool:
        """移除策略节点"""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found")
            return False

        # 移除相关连接
        self.connections = [
            conn for conn in self.connections
            if conn['source'] != node_id and conn['target'] != node_id
        ]

        # 从其他节点的next_nodes中移除
        for node in self.nodes.values():
            if node_id in node.next_nodes:
                node.next_nodes.remove(node_id)

        del self.nodes[node_id]
        return True

    def connect_nodes(self, source_id: str, target_id: str) -> bool:
        """连接两个策略节点"""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning("Source or target node not found")
            return False

        # 检查是否已存在连接
        if any(
            conn['source'] == source_id and conn['target'] == target_id
            for conn in self.connections
        ):
            logger.warning("Connection already exists")
            return False

        # 添加到连接列表
        self.connections.append({
            'source': source_id,
            'target': target_id
        })

        # 添加到下游节点列表
        self.nodes[source_id].next_nodes.append(target_id)
        return True

    def disconnect_nodes(self, source_id: str, target_id: str) -> bool:
        """断开节点连接"""
        # 从连接列表中移除
        self.connections = [
            conn for conn in self.connections
            if not (conn['source'] == source_id and conn['target'] == target_id)
        ]

        # 从下游节点列表中移除
        if source_id in self.nodes and target_id in self.nodes[source_id].next_nodes:
            self.nodes[source_id].next_nodes.remove(target_id)

        return True

    def update_node_params(self, node_id: str, new_params: Dict) -> bool:
        """更新节点参数"""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found")
            return False

        self.nodes[node_id].params.update(new_params)
        return True

    def get_node(self, node_id: str) -> Optional[StrategyNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[StrategyNode]:
        """获取所有节点"""
        return list(self.nodes.values())

    def get_connections(self) -> List[Dict]:
        """获取所有连接"""
        return self.connections.copy()

    def validate_strategy(self) -> bool:
        """验证策略有效性"""
        # 检查是否有孤立节点
        connected_nodes = set()
        for conn in self.connections:
            connected_nodes.add(conn['source'])
            connected_nodes.add(conn['target'])

        for node_id in self.nodes:
            if node_id not in connected_nodes and len(self.nodes[node_id].next_nodes) > 0:
                logger.error(f"Isolated node found: {node_id}")
                return False

        # 检查数据流闭环
        if self._has_cycle():
            logger.error("Circular dependency detected")
            return False

        return True

    def _has_cycle(self) -> bool:
        """检测图中是否有环"""
        visited = set()
        recursion_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            recursion_stack.add(node_id)

            for neighbor in self.nodes[node_id].next_nodes:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True

            recursion_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def export_strategy(self) -> Dict:
        """导出策略配置"""
        return {
            'nodes': {
                node_id: {
                    'node_type': node.node_type.value,
                    'name': node.name,
                    'params': node.params,
                    'next_nodes': node.next_nodes
                }
                for node_id, node in self.nodes.items()
            },
            'connections': self.connections
        }

    def import_strategy(self, config: Dict) -> bool:
        """导入策略配置"""
        try:
            # 清空当前策略
            self.nodes = {}
            self.connections = []

            # 导入节点
            for node_id, node_data in config['nodes'].items():
                node = StrategyNode(
                    node_id=node_id,
                    node_type=NodeType(node_data['node_type']),
                    name=node_data['name'],
                    params=node_data['params'],
                    next_nodes=node_data['next_nodes']
                )
                self.add_node(node)

            # 导入连接
            self.connections = config['connections']

            return True
        except Exception as e:
            logger.error(f"Failed to import strategy: {e}")
            return False

    def visualize(self):
        """可视化策略图(简化版)"""
        print("Strategy Visualization:")
        for node_id, node in self.nodes.items():
            print(f"Node {node_id} ({node.node_type.value}): {node.name}")
            if node.next_nodes:
                print(f"  -> {', '.join(node.next_nodes)}")
