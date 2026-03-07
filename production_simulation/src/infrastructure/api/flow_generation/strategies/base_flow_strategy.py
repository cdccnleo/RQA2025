"""
流程生成策略基类

定义流程生成的抽象接口和通用方法
"""

from abc import ABC
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class FlowNode:
    """流程节点"""
    node_id: str
    label: str
    node_type: str  # start, end, process, decision, api_call
    description: str = ""
    position: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowEdge:
    """流程边"""
    edge_id: str
    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    arrow_type: str = "normal"


@dataclass
class FlowDiagram:
    """流程图"""
    flow_id: str
    title: str
    description: str
    nodes: List[FlowNode] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)
    layout: str = "horizontal"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFlowStrategy(ABC):
    """
    流程生成策略基类
    
    定义流程生成的标准接口和通用方法，
    子类实现具体服务的流程生成逻辑。
    """
    
    def __init__(self):
        """初始化流程策略"""
        self.nodes: List[FlowNode] = []
        self.edges: List[FlowEdge] = []
    
    def generate_flow(self) -> FlowDiagram:
        """
        生成流程图（子类必须实现）
        
        Returns:
            FlowDiagram: 生成的流程图
        """
        raise NotImplementedError("子类必须实现 generate_flow 方法")
    
    def _add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        description: str = "",
        position: Dict[str, int] = None
    ):
        """添加节点的辅助方法"""
        node = FlowNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            description=description,
            position=position or {}
        )
        self.nodes.append(node)
        return node
    
    def _add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        label: str = "",
        condition: str = "",
        arrow_type: str = "normal"
    ):
        """添加边的辅助方法"""
        edge = FlowEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            label=label,
            condition=condition,
            arrow_type=arrow_type
        )
        self.edges.append(edge)
        return edge
    
    def _create_start_node(self, description: str = "开始"):
        """创建开始节点"""
        return self._add_node("start", "开始", "start", description)
    
    def _create_end_node(self, description: str = "结束"):
        """创建结束节点"""
        return self._add_node("end", "结束", "end", description)
    
    def _create_process_node(self, node_id: str, label: str, description: str = ""):
        """创建处理节点"""
        return self._add_node(node_id, label, "process", description)
    
    def _create_decision_node(self, node_id: str, label: str, description: str = ""):
        """创建决策节点"""
        return self._add_node(node_id, label, "decision", description)
    
    def _create_api_call_node(self, node_id: str, label: str, description: str = ""):
        """创建API调用节点"""
        return self._add_node(node_id, label, "api_call", description)
    
    def _connect_nodes(
        self,
        from_id: str,
        to_id: str,
        label: str = "",
        condition: str = ""
    ):
        """连接两个节点"""
        edge_id = f"{from_id}_to_{to_id}"
        return self._add_edge(edge_id, from_id, to_id, label, condition)

