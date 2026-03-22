#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征血缘追踪模块

提供特征血缘关系追踪、依赖图谱构建、血缘查询等功能。
实现特征的可追溯性和依赖关系可视化。
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FeatureLineageNode:
    """特征血缘节点"""
    feature_name: str
    feature_type: str  # 'raw', 'derived', 'aggregated', 'transformed'
    source_features: List[str] = field(default_factory=list)  # 父特征列表
    derived_features: List[str] = field(default_factory=list)  # 子特征列表
    transformation_logic: str = ""  # 转换逻辑描述
    parameters: Dict[str, Any] = field(default_factory=dict)  # 参数
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """血缘关系边"""
    source: str  # 源特征
    target: str  # 目标特征
    relationship: str  # 'derived_from', 'depends_on', 'transformed_from'
    transformation: str = ""  # 转换描述
    created_at: datetime = field(default_factory=datetime.now)


class FeatureLineageTracker:
    """
    特征血缘追踪器
    
    功能：
    - 记录特征血缘关系
    - 构建血缘图谱
    - 查询特征依赖链
    - 血缘关系可视化
    - 影响分析
    """
    
    def __init__(self, storage_dir: str = "feature_lineage"):
        """
        初始化血缘追踪器
        
        Args:
            storage_dir: 血缘数据存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存中的血缘图谱
        self._nodes: Dict[str, FeatureLineageNode] = {}
        self._edges: List[LineageEdge] = []
        
        # 加载已有数据
        self._load_lineage_data()
        
        logger.info(f"✅ 特征血缘追踪器初始化完成: {storage_dir}")
    
    def _load_lineage_data(self):
        """加载血缘数据"""
        try:
            nodes_file = self.storage_dir / "nodes.json"
            edges_file = self.storage_dir / "edges.json"
            
            if nodes_file.exists():
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    nodes_data = json.load(f)
                    for name, data in nodes_data.items():
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        self._nodes[name] = FeatureLineageNode(**data)
            
            if edges_file.exists():
                with open(edges_file, 'r', encoding='utf-8') as f:
                    edges_data = json.load(f)
                    for edge_data in edges_data:
                        edge_data['created_at'] = datetime.fromisoformat(edge_data['created_at'])
                        self._edges.append(LineageEdge(**edge_data))
            
            logger.info(f"📊 加载血缘数据: {len(self._nodes)} 个节点, {len(self._edges)} 条边")
            
        except Exception as e:
            logger.error(f"❌ 加载血缘数据失败: {e}")
    
    def _save_lineage_data(self):
        """保存血缘数据"""
        try:
            # 保存节点
            nodes_data = {}
            for name, node in self._nodes.items():
                node_dict = asdict(node)
                node_dict['created_at'] = node.created_at.isoformat()
                nodes_data[name] = node_dict
            
            with open(self.storage_dir / "nodes.json", 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2, ensure_ascii=False)
            
            # 保存边
            edges_data = []
            for edge in self._edges:
                edge_dict = asdict(edge)
                edge_dict['created_at'] = edge.created_at.isoformat()
                edges_data.append(edge_dict)
            
            with open(self.storage_dir / "edges.json", 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ 保存血缘数据失败: {e}")
    
    def register_feature(
        self,
        feature_name: str,
        feature_type: str,
        source_features: Optional[List[str]] = None,
        transformation_logic: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册特征血缘
        
        Args:
            feature_name: 特征名称
            feature_type: 特征类型
            source_features: 源特征列表
            transformation_logic: 转换逻辑描述
            parameters: 参数
            tags: 标签
            metadata: 元数据
            
        Returns:
            bool: 是否成功
        """
        try:
            # 创建或更新节点
            node = FeatureLineageNode(
                feature_name=feature_name,
                feature_type=feature_type,
                source_features=source_features or [],
                transformation_logic=transformation_logic,
                parameters=parameters or {},
                tags=tags or [],
                metadata=metadata or {}
            )
            
            self._nodes[feature_name] = node
            
            # 创建血缘边
            if source_features:
                for source in source_features:
                    edge = LineageEdge(
                        source=source,
                        target=feature_name,
                        relationship='derived_from',
                        transformation=transformation_logic
                    )
                    self._edges.append(edge)
                    
                    # 更新源特征的派生列表
                    if source in self._nodes:
                        if feature_name not in self._nodes[source].derived_features:
                            self._nodes[source].derived_features.append(feature_name)
            
            # 保存数据
            self._save_lineage_data()
            
            logger.info(f"✅ 特征血缘已注册: {feature_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 注册特征血缘失败: {e}")
            return False
    
    def get_lineage(self, feature_name: str) -> Optional[FeatureLineageNode]:
        """
        获取特征血缘信息
        
        Args:
            feature_name: 特征名称
            
        Returns:
            FeatureLineageNode: 血缘节点信息
        """
        return self._nodes.get(feature_name)
    
    def get_upstream_features(self, feature_name: str, depth: int = -1) -> List[str]:
        """
        获取上游特征（依赖的特征）
        
        Args:
            feature_name: 特征名称
            depth: 查询深度，-1表示无限
            
        Returns:
            List[str]: 上游特征列表
        """
        upstream = []
        visited = set()
        
        def _get_upstream_recursive(name: str, current_depth: int):
            if current_depth == 0 or name in visited:
                return
            
            visited.add(name)
            node = self._nodes.get(name)
            
            if node and node.source_features:
                for source in node.source_features:
                    if source not in upstream:
                        upstream.append(source)
                        _get_upstream_recursive(source, current_depth - 1 if current_depth > 0 else -1)
        
        _get_upstream_recursive(feature_name, depth)
        return upstream
    
    def get_downstream_features(self, feature_name: str, depth: int = -1) -> List[str]:
        """
        获取下游特征（被依赖的特征）
        
        Args:
            feature_name: 特征名称
            depth: 查询深度，-1表示无限
            
        Returns:
            List[str]: 下游特征列表
        """
        downstream = []
        visited = set()
        
        def _get_downstream_recursive(name: str, current_depth: int):
            if current_depth == 0 or name in visited:
                return
            
            visited.add(name)
            node = self._nodes.get(name)
            
            if node and node.derived_features:
                for derived in node.derived_features:
                    if derived not in downstream:
                        downstream.append(derived)
                        _get_downstream_recursive(derived, current_depth - 1 if current_depth > 0 else -1)
        
        _get_downstream_recursive(feature_name, depth)
        return downstream
    
    def get_lineage_graph(self, feature_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        获取特征血缘图谱
        
        Args:
            feature_name: 特征名称
            depth: 查询深度
            
        Returns:
            Dict: 血缘图谱数据
        """
        graph = {
            "center": feature_name,
            "nodes": [],
            "edges": [],
            "upstream": [],
            "downstream": []
        }
        
        # 获取上游特征
        upstream = self.get_upstream_features(feature_name, depth)
        graph["upstream"] = upstream
        
        # 获取下游特征
        downstream = self.get_downstream_features(feature_name, depth)
        graph["downstream"] = downstream
        
        # 收集所有相关节点
        all_features = set([feature_name] + upstream + downstream)
        
        for name in all_features:
            node = self._nodes.get(name)
            if node:
                graph["nodes"].append({
                    "name": name,
                    "type": node.feature_type,
                    "tags": node.tags
                })
        
        # 收集相关边
        for edge in self._edges:
            if edge.source in all_features and edge.target in all_features:
                graph["edges"].append({
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "transformation": edge.transformation
                })
        
        return graph
    
    def analyze_impact(self, feature_name: str) -> Dict[str, Any]:
        """
        分析特征变更影响范围
        
        Args:
            feature_name: 特征名称
            
        Returns:
            Dict: 影响分析结果
        """
        downstream = self.get_downstream_features(feature_name, depth=-1)
        
        return {
            "feature": feature_name,
            "affected_features": downstream,
            "affected_count": len(downstream),
            "impact_level": "high" if len(downstream) > 10 else "medium" if len(downstream) > 5 else "low",
            "recommendation": "需要全面测试" if len(downstream) > 10 else "建议回归测试"
        }
    
    def get_lineage_summary(self) -> Dict[str, Any]:
        """
        获取血缘统计摘要
        
        Returns:
            Dict: 统计信息
        """
        feature_types = defaultdict(int)
        for node in self._nodes.values():
            feature_types[node.feature_type] += 1
        
        return {
            "total_features": len(self._nodes),
            "total_relationships": len(self._edges),
            "feature_types": dict(feature_types),
            "storage_dir": str(self.storage_dir)
        }
    
    def export_lineage(self, format: str = "json") -> str:
        """
        导出血缘数据
        
        Args:
            format: 导出格式 ('json', 'dot')
            
        Returns:
            str: 导出的数据
        """
        if format == "json":
            data = {
                "nodes": [asdict(node) for node in self._nodes.values()],
                "edges": [asdict(edge) for edge in self._edges]
            }
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        
        elif format == "dot":
            # GraphViz DOT格式
            lines = ["digraph FeatureLineage {"]
            lines.append("  rankdir=TB;")
            lines.append("  node [shape=box];")
            
            # 添加节点
            for name, node in self._nodes.items():
                color = "lightblue" if node.feature_type == "raw" else "lightgreen"
                lines.append(f'  "{name}" [style=filled, fillcolor={color}];')
            
            # 添加边
            for edge in self._edges:
                lines.append(f'  "{edge.source}" -> "{edge.target}";')
            
            lines.append("}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局血缘追踪器实例
_lineage_tracker: Optional[FeatureLineageTracker] = None


def get_lineage_tracker(storage_dir: str = "feature_lineage") -> FeatureLineageTracker:
    """
    获取全局血缘追踪器实例（单例模式）
    
    Args:
        storage_dir: 存储目录
        
    Returns:
        FeatureLineageTracker: 血缘追踪器实例
    """
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = FeatureLineageTracker(storage_dir)
    return _lineage_tracker
