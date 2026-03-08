"""
Lineage Graph Module

提供数据血缘图谱的构建和查询功能。
"""

import logging
from typing import Dict, List, Optional, Set
from collections import deque

from ..models.lineage_models import (
    DataAsset,
    LineageEdge,
    LineageNode,
    LineagePath,
    LineageQuery,
    ImpactAnalysisResult
)


logger = logging.getLogger(__name__)


class LineageGraph:
    """
    血缘图谱
    
    管理数据血缘关系的图结构，支持构建、查询和分析。
    
    功能：
    1. 添加节点和边
    2. 查询血缘关系（上游/下游）
    3. 查找血缘路径
    4. 影响分析
    """
    
    def __init__(self):
        """初始化血缘图谱"""
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        self.adjacency: Dict[str, List[str]] = {}  # source -> [targets]
        self.reverse_adjacency: Dict[str, List[str]] = {}  # target -> [sources]
        
    def add_node(self, node: LineageNode) -> None:
        """
        添加节点
        
        Args:
            node: 血缘节点
        """
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
        if node.id not in self.reverse_adjacency:
            self.reverse_adjacency[node.id] = []
            
    def add_edge(self, edge: LineageEdge) -> None:
        """
        添加边
        
        Args:
            edge: 血缘边
        """
        self.edges[edge.id] = edge
        
        # 更新邻接表
        if edge.source_id not in self.adjacency:
            self.adjacency[edge.source_id] = []
        self.adjacency[edge.source_id].append(edge.target_id)
        
        # 更新反向邻接表
        if edge.target_id not in self.reverse_adjacency:
            self.reverse_adjacency[edge.target_id] = []
        self.reverse_adjacency[edge.target_id].append(edge.source_id)
        
    def get_upstream(self, asset_id: str, depth: int = -1) -> List[DataAsset]:
        """
        获取上游依赖
        
        Args:
            asset_id: 资产ID
            depth: 查询深度，-1表示无限制
            
        Returns:
            上游资产列表
        """
        return self._traverse(asset_id, "upstream", depth)
        
    def get_downstream(self, asset_id: str, depth: int = -1) -> List[DataAsset]:
        """
        获取下游影响
        
        Args:
            asset_id: 资产ID
            depth: 查询深度，-1表示无限制
            
        Returns:
            下游资产列表
        """
        return self._traverse(asset_id, "downstream", depth)
        
    def _traverse(self, start_id: str, direction: str, depth: int) -> List[DataAsset]:
        """
        遍历图谱
        
        Args:
            start_id: 起始节点ID
            direction: 遍历方向（upstream/downstream）
            depth: 遍历深度
            
        Returns:
            遍历到的资产列表
        """
        if start_id not in self.nodes:
            return []
            
        result = []
        visited = {start_id}
        queue = deque([(start_id, 0)])
        
        adj = self.reverse_adjacency if direction == "upstream" else self.adjacency
        
        while queue:
            node_id, current_depth = queue.popleft()
            
            if depth != -1 and current_depth >= depth:
                continue
                
            for neighbor_id in adj.get(node_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    if neighbor_id in self.nodes:
                        result.append(self.nodes[neighbor_id].asset)
                    queue.append((neighbor_id, current_depth + 1))
                    
        return result
        
    def find_path(self, source_id: str, target_id: str) -> Optional[LineagePath]:
        """
        查找血缘路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            血缘路径或None
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
            
        # BFS查找路径
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                # 构建路径对象
                return self._build_path(path)
                
            for neighbor_id in self.adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
                    
        return None
        
    def _build_path(self, node_ids: List[str]) -> LineagePath:
        """
        构建路径对象
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            血缘路径
        """
        nodes = []
        edges = []
        
        for i, node_id in enumerate(node_ids):
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.level = i
                nodes.append(node)
                
        # 查找边
        for i in range(len(node_ids) - 1):
            source_id = node_ids[i]
            target_id = node_ids[i + 1]
            
            for edge in self.edges.values():
                if edge.source_id == source_id and edge.target_id == target_id:
                    edges.append(edge)
                    break
                    
        return LineagePath(
            nodes=nodes,
            edges=edges,
            distance=len(node_ids) - 1
        )
        
    def analyze_impact(self, asset_id: str) -> ImpactAnalysisResult:
        """
        影响分析
        
        Args:
            asset_id: 资产ID
            
        Returns:
            影响分析结果
        """
        upstream = self.get_upstream(asset_id)
        downstream = self.get_downstream(asset_id)
        
        return ImpactAnalysisResult(
            asset_id=asset_id,
            upstream=upstream,
            downstream=downstream,
            total_affected=len(upstream) + len(downstream)
        )
        
    def query(self, query: LineageQuery) -> Dict[str, List[DataAsset]]:
        """
        查询血缘关系
        
        Args:
            query: 查询参数
            
        Returns:
            查询结果
        """
        result = {}
        
        if query.direction in ("upstream", "both"):
            result["upstream"] = self.get_upstream(query.asset_id, query.depth)
            
        if query.direction in ("downstream", "both"):
            result["downstream"] = self.get_downstream(query.asset_id, query.depth)
            
        return result
        
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            包含图谱数据的字典
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
        
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "avg_out_degree": sum(len(v) for v in self.adjacency.values()) / max(len(self.adjacency), 1),
            "avg_in_degree": sum(len(v) for v in self.reverse_adjacency.values()) / max(len(self.reverse_adjacency), 1)
        }
