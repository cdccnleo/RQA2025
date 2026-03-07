"""
API流程图生成框架

将APIFlowDiagramGenerator拆分为专注的类：
- FlowModels: 数据模型
- FlowNodeBuilder: 节点构建器
- FlowGenerators: 流程生成器
- FlowExporter: 导出器
- APIFlowCoordinator: 协调器（对外接口）
"""

from .models import APIFlowNode, APIFlowEdge, APIFlowDiagram
from .node_builder import FlowNodeBuilder
from .flow_generators import (
    DataServiceFlowGenerator,
    TradingFlowGenerator,
    FeatureEngineeringFlowGenerator
)
from .exporter import FlowExporter
from .coordinator import APIFlowCoordinator

# 向后兼容
APIFlowDiagramGenerator = APIFlowCoordinator

__all__ = [
    'APIFlowNode',
    'APIFlowEdge',
    'APIFlowDiagram',
    'FlowNodeBuilder',
    'DataServiceFlowGenerator',
    'TradingFlowGenerator',
    'FeatureEngineeringFlowGenerator',
    'FlowExporter',
    'APIFlowCoordinator',
    'APIFlowDiagramGenerator'
]

