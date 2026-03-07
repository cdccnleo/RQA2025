"""
API流程图数据模型

定义流程图的基础数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class APIFlowNode:
    """API流程节点"""
    id: str
    label: str
    node_type: str  # start, process, decision, end, api_call, database, cache
    description: str = ""
    position: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIFlowEdge:
    """API流程边"""
    source: str
    target: str
    label: str = ""
    edge_type: str = "normal"  # normal, success, error, conditional
    condition: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIFlowDiagram:
    """API流程图"""
    id: str
    title: str
    description: str
    nodes: List[APIFlowNode] = field(default_factory=list)
    edges: List[APIFlowEdge] = field(default_factory=list)
    layout: str = "horizontal"  # horizontal, vertical, radial
    theme: str = "default"

