"""
流程图导出器

职责：将流程图导出为各种格式（Mermaid、JSON等）
"""

import json
from pathlib import Path
from typing import Dict, List
from .models import APIFlowDiagram


class FlowExporter:
    """流程图导出器"""
    
    def __init__(self):
        """初始化导出器"""
        pass
    
    def export_to_mermaid(
        self,
        diagram: APIFlowDiagram,
        output_path: str
    ) -> str:
        """
        导出为Mermaid格式
        
        Args:
            diagram: 流程图对象
            output_path: 输出路径
            
        Returns:
            str: Mermaid代码
        """
        mermaid_code = self._generate_mermaid(diagram)
        
        # 保存文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(mermaid_code, encoding='utf-8')
        
        return mermaid_code
    
    def _generate_mermaid(self, diagram: APIFlowDiagram) -> str:
        """生成Mermaid代码"""
        lines = ["```mermaid"]
        
        # 流程图类型
        if diagram.layout == "horizontal":
            lines.append("graph LR")
        elif diagram.layout == "vertical":
            lines.append("graph TD")
        else:
            lines.append("graph LR")
        
        # 节点定义
        for node in diagram.nodes:
            node_shape = self._get_node_shape(node.node_type)
            lines.append(f"    {node.id}{node_shape[0]}{node.label}{node_shape[1]}")
        
        # 边定义
        for edge in diagram.edges:
            arrow = self._get_arrow_type(edge.edge_type)
            if edge.label:
                lines.append(f"    {edge.source} {arrow}|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} {arrow} {edge.target}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _get_node_shape(self, node_type: str) -> tuple:
        """获取节点形状"""
        shapes = {
            "start": ("([", "])"),
            "end": ("([", "])"),
            "process": ("[", "]"),
            "decision": ("{", "}"),
            "api_call": ("[[", "]]"),
            "database": ("[(", ")]"),
            "cache": ("[/", "/]")
        }
        return shapes.get(node_type, ("[", "]"))
    
    def _get_arrow_type(self, edge_type: str) -> str:
        """获取箭头类型"""
        arrows = {
            "normal": "-->",
            "success": "==>",
            "error": "-.->",
            "conditional": "-->"
        }
        return arrows.get(edge_type, "-->")
    
    def export_to_json(
        self,
        diagram: APIFlowDiagram,
        output_path: str
    ) -> Dict:
        """
        导出为JSON格式

        Args:
            diagram: 流程图对象
            output_path: 输出路径

        Returns:
            Dict: JSON数据
        """
        # 创建数据字典
        data = self._create_diagram_data(diagram)

        # 保存文件
        self._save_json_file(data, output_path)

        return data

    def _create_diagram_data(self, diagram: APIFlowDiagram) -> Dict:
        """创建流程图数据字典"""
        return {
            "id": diagram.id,
            "title": diagram.title,
            "description": diagram.description,
            "layout": diagram.layout,
            "theme": diagram.theme,
            "nodes": self._create_nodes_data(diagram.nodes),
            "edges": self._create_edges_data(diagram.edges)
        }

    def _create_nodes_data(self, nodes) -> List[Dict]:
        """创建节点数据列表"""
        return [
            {
                "id": node.id,
                "label": node.label,
                "type": node.node_type,
                "description": node.description,
                "position": node.position,
                "metadata": node.metadata
            }
            for node in nodes
        ]

    def _create_edges_data(self, edges) -> List[Dict]:
        """创建边数据列表"""
        return [
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "type": edge.edge_type,
                "condition": edge.condition,
                "metadata": edge.metadata
            }
            for edge in edges
        ]

    def _save_json_file(self, data: Dict, output_path: str):
        """保存JSON文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def get_statistics(self, diagram: APIFlowDiagram) -> Dict:
        """获取流程图统计信息"""
        node_types = {}
        for node in diagram.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        edge_types = {}
        for edge in diagram.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        
        return {
            "total_nodes": len(diagram.nodes),
            "total_edges": len(diagram.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "complexity": len(diagram.nodes) + len(diagram.edges)
        }

