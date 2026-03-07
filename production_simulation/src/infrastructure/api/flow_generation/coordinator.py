"""
API流程图生成协调器

职责：协调各个组件生成完整的流程图
向后兼容：提供与APIFlowDiagramGenerator相同的接口
"""

from pathlib import Path
from typing import Dict
from .models import APIFlowDiagram
from .node_builder import FlowNodeBuilder
from .flow_generators import (
    DataServiceFlowGenerator,
    TradingFlowGenerator,
    FeatureEngineeringFlowGenerator
)
from .exporter import FlowExporter


class APIFlowCoordinator:
    """API流程图生成协调器"""
    
    def __init__(self):
        """初始化协调器"""
        # 初始化组件
        self.node_builder = FlowNodeBuilder()
        self.exporter = FlowExporter()
        
        # 初始化生成器
        self.data_gen = DataServiceFlowGenerator(self.node_builder)
        self.trading_gen = TradingFlowGenerator(self.node_builder)
        self.feature_gen = FeatureEngineeringFlowGenerator(self.node_builder)
        
        # 存储已生成的流程图
        self.diagrams: Dict[str, APIFlowDiagram] = {}
    
    def create_data_service_flow(self) -> APIFlowDiagram:
        """创建数据服务流程图"""
        diagram = self.data_gen.generate()
        self.diagrams[diagram.id] = diagram
        return diagram
    
    def create_trading_flow(self) -> APIFlowDiagram:
        """创建交易流程图"""
        diagram = self.trading_gen.generate()
        self.diagrams[diagram.id] = diagram
        return diagram
    
    def create_feature_engineering_flow(self) -> APIFlowDiagram:
        """创建特征工程流程图"""
        diagram = self.feature_gen.generate()
        self.diagrams[diagram.id] = diagram
        return diagram
    
    def generate_all_flows(self) -> Dict[str, APIFlowDiagram]:
        """生成所有流程图"""
        self.create_data_service_flow()
        self.create_trading_flow()
        self.create_feature_engineering_flow()
        return self.diagrams.copy()
    
    def export_to_mermaid(self, output_dir: str = "docs/api/flows") -> Dict[str, str]:
        """
        导出所有流程图为Mermaid格式
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 文件路径字典
        """
        if not self.diagrams:
            self.generate_all_flows()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        for diagram_id, diagram in self.diagrams.items():
            file_path = output_path / f"{diagram_id}.md"
            self.exporter.export_to_mermaid(diagram, str(file_path))
            files[diagram_id] = str(file_path)
        
        return files
    
    def export_to_json(self, output_dir: str = "docs/api/flows") -> Dict[str, str]:
        """
        导出所有流程图为JSON格式
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 文件路径字典
        """
        if not self.diagrams:
            self.generate_all_flows()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        for diagram_id, diagram in self.diagrams.items():
            file_path = output_path / f"{diagram_id}.json"
            self.exporter.export_to_json(diagram, str(file_path))
            files[diagram_id] = str(file_path)
        
        return files
    
    def get_flow_statistics(self) -> Dict[str, any]:
        """获取所有流程图的统计信息"""
        if not self.diagrams:
            self.generate_all_flows()
        
        stats = {
            "total_flows": len(self.diagrams),
            "flows": {}
        }
        
        for diagram_id, diagram in self.diagrams.items():
            stats["flows"][diagram_id] = self.exporter.get_statistics(diagram)
        
        return stats


# 向后兼容别名
APIFlowDiagramGenerator = APIFlowCoordinator

