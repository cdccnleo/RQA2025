"""
流程图生成相关配置

提供流程图生成所需的各类配置对象
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig, ValidationResult, ExportFormat


@dataclass
class FlowNodeConfig(BaseConfig):
    """流程节点配置"""
    
    node_id: str
    label: str
    node_type: str  # start, end, process, decision, api_call
    description: Optional[str] = None
    position: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    style: Optional[Dict[str, str]] = None
    
    def _validate_impl(self, result: ValidationResult):
        """验证节点配置"""
        if not self.node_id:
            result.add_error("节点ID不能为空")
        
        if not self.label:
            result.add_error("节点标签不能为空")
        
        valid_types = ['start', 'end', 'process', 'decision', 'api_call', 'data']
        if self.node_type not in valid_types:
            result.add_error(f"节点类型必须是 {valid_types} 之一")
        
        if self.position:
            if 'x' not in self.position or 'y' not in self.position:
                result.add_error("位置信息必须包含x和y坐标")


@dataclass
class FlowEdgeConfig(BaseConfig):
    """流程边配置"""
    
    edge_id: str
    from_node: str
    to_node: str
    label: Optional[str] = None
    condition: Optional[str] = None
    arrow_type: str = "normal"  # normal, thick, dashed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _validate_impl(self, result: ValidationResult):
        """验证边配置"""
        if not self.edge_id:
            result.add_error("边ID不能为空")
        
        if not self.from_node:
            result.add_error("起始节点不能为空")
        
        if not self.to_node:
            result.add_error("目标节点不能为空")
        
        valid_arrow_types = ['normal', 'thick', 'dashed', 'dotted']
        if self.arrow_type not in valid_arrow_types:
            result.add_error(f"箭头类型必须是 {valid_arrow_types} 之一")


@dataclass
class FlowGenerationConfig(BaseConfig):
    """流程图生成配置"""
    
    title: str
    flow_type: str
    flow_id: Optional[str] = None
    description: Optional[str] = None
    nodes: List[FlowNodeConfig] = field(default_factory=list)
    edges: List[FlowEdgeConfig] = field(default_factory=list)
    layout: str = "horizontal"  # horizontal, vertical, hierarchical
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 样式配置
    theme: str = "default"  # default, dark, light
    show_timestamps: bool = False
    show_metadata: bool = False
    
    def __post_init__(self):
        self.flow_type = (self.flow_type or "").lower()
        self._flow_id_provided = bool(self.flow_id)
        if not self.flow_id:
            self.flow_id = self._generate_flow_id()
        super().__post_init__()

    def _generate_flow_id(self) -> str:
        """根据标题生成默认的流程ID"""
        base = (self.title or "flow").strip().lower().replace(" ", "_")
        return f"{self.flow_type}_{base}" if self.flow_type else base or "flow_default"

    def _validate_impl(self, result: ValidationResult):
        """验证流程生成配置"""
        if not self.flow_id:
            result.add_error("流程ID不能为空")
        elif not getattr(self, "_flow_id_provided", False) and self._validation_mode == "strict":
            result.add_error("流程ID不能为空")
        
        if not self.title:
            result.add_error("流程标题不能为空")
        
        valid_types = ['data_service', 'trading', 'feature_engineering', 'monitoring', 'custom']
        if self.flow_type not in valid_types:
            result.add_error(f"流程类型必须是 {valid_types} 之一")
        
        valid_layouts = ['horizontal', 'vertical', 'hierarchical', 'radial']
        if self.layout not in valid_layouts:
            result.add_error(f"布局类型必须是 {valid_layouts} 之一")
        
        # 验证节点
        if not self.nodes:
            result.add_warning("流程图没有定义任何节点")
        
        for node in self.nodes:
            node_result = node.validate()
            result.merge(node_result)
        
        # 验证边
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            edge_result = edge.validate()
            result.merge(edge_result)
            
            if edge.from_node not in node_ids:
                result.add_error(f"边 {edge.edge_id} 的起始节点 {edge.from_node} 不存在")
            
            if edge.to_node not in node_ids:
                result.add_error(f"边 {edge.edge_id} 的目标节点 {edge.to_node} 不存在")
        
        # 检查是否有孤立节点
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)
        
        isolated_nodes = node_ids - connected_nodes
        if isolated_nodes:
            result.add_warning(f"存在孤立节点: {isolated_nodes}")
    
    def add_node(self, node: FlowNodeConfig):
        """添加节点"""
        self.nodes.append(node)
    
    def add_edge(self, edge: FlowEdgeConfig):
        """添加边"""
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[FlowNodeConfig]:
        """获取节点"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


@dataclass
class FlowExportConfig(BaseConfig):
    """流程图导出配置"""
    
    format: str
    output_dir: Optional[str] = None
    output_path: Optional[str] = None
    include_metadata: bool = True
    include_statistics: bool = False
    compress_output: bool = False
    encoding: str = "utf-8"
    pretty_print: bool = True
    theme: str = "default"

    def __post_init__(self):
        if isinstance(self.format, ExportFormat):
            self.format = self.format.value
        if self.format:
            self.format = self.format.lower()
        target_dir = self.output_dir or self.output_path or ""
        self.output_dir = target_dir
        self.output_path = target_dir
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证导出配置"""
        if not self.output_dir:
            result.add_error("输出路径不能为空")
        
        valid_formats = ['json', 'yaml', 'markdown', 'mermaid']
        if self.format not in valid_formats:
            result.add_error("不支持的导出格式")

