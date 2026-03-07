"""
流程节点构建器

职责：创建和管理流程图节点和边
"""

from typing import Dict, List
from .models import APIFlowNode, APIFlowEdge


class FlowNodeBuilder:
    """流程节点构建器"""
    
    def __init__(self):
        """初始化节点构建器"""
        self.nodes: List[APIFlowNode] = []
        self.edges: List[APIFlowEdge] = []
    
    def create_node(
        self,
        id: str,
        label: str,
        node_type: str,
        description: str = "",
        position: Dict[str, float] = None
    ) -> APIFlowNode:
        """
        创建节点
        
        Args:
            id: 节点ID
            label: 节点标签
            node_type: 节点类型
            description: 描述
            position: 位置坐标
            
        Returns:
            APIFlowNode: 创建的节点
        """
        node = APIFlowNode(
            id=id,
            label=label,
            node_type=node_type,
            description=description,
            position=position or {}
        )
        self.nodes.append(node)
        return node
    
    def create_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        edge_type: str = "normal",
        condition: str = ""
    ) -> APIFlowEdge:
        """
        创建边
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            label: 边标签
            edge_type: 边类型
            condition: 条件
            
        Returns:
            APIFlowEdge: 创建的边
        """
        edge = APIFlowEdge(
            source=source,
            target=target,
            label=label,
            edge_type=edge_type,
            condition=condition
        )
        self.edges.append(edge)
        return edge
    
    def create_start_node(self, id: str = "start", position: Dict = None) -> APIFlowNode:
        """创建开始节点"""
        return self.create_node(
            id=id,
            label="开始",
            node_type="start",
            description="流程开始",
            position=position or {"x": 100, "y": 100}
        )
    
    def create_end_node(self, id: str = "end", position: Dict = None) -> APIFlowNode:
        """创建结束节点"""
        return self.create_node(
            id=id,
            label="结束",
            node_type="end",
            description="流程结束",
            position=position or {"x": 900, "y": 100}
        )
    
    def create_process_node(
        self,
        id: str,
        label: str,
        description: str = "",
        position: Dict = None
    ) -> APIFlowNode:
        """创建处理节点"""
        return self.create_node(
            id=id,
            label=label,
            node_type="process",
            description=description,
            position=position or {}
        )
    
    def create_decision_node(
        self,
        id: str,
        label: str,
        description: str = "",
        position: Dict = None
    ) -> APIFlowNode:
        """创建决策节点"""
        return self.create_node(
            id=id,
            label=label,
            node_type="decision",
            description=description,
            position=position or {}
        )
    
    def create_api_call_node(
        self,
        id: str,
        label: str,
        description: str = "",
        position: Dict = None
    ) -> APIFlowNode:
        """创建API调用节点"""
        return self.create_node(
            id=id,
            label=label,
            node_type="api_call",
            description=description,
            position=position or {}
        )
    
    def get_nodes(self) -> List[APIFlowNode]:
        """获取所有节点"""
        return self.nodes.copy()
    
    def get_edges(self) -> List[APIFlowEdge]:
        """获取所有边"""
        return self.edges.copy()
    
    def clear(self):
        """清空节点和边"""
        self.nodes.clear()
        self.edges.clear()

