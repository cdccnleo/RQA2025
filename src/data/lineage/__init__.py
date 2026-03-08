"""
Data Lineage Module - 数据血缘追踪模块

提供数据血缘追踪功能，记录数据从产生到消费的完整生命周期，
包括数据来源、转换过程、依赖关系等。

属于数据管理层，与数据管道、数据同步等模块协同工作。

主要功能:
1. 血缘元数据采集 - 自动采集数据处理过程中的血缘信息
2. 血缘图谱构建 - 构建数据血缘关系图谱
3. 血缘查询分析 - 支持血缘追溯和影响分析
4. 可视化展示 - 提供血缘关系可视化

作者: RQA2025 Architecture Team
版本: 1.0.0
日期: 2026-03-08
"""

from .core.lineage_tracker import LineageTracker
from .core.lineage_graph import LineageGraph
from .models.lineage_models import (
    DataAsset,
    LineageEdge,
    LineageNode,
    LineageType,
    Transformation
)

__version__ = "1.0.0"
__all__ = [
    # 核心组件
    "LineageTracker",
    "LineageGraph",
    # 数据模型
    "DataAsset",
    "LineageEdge",
    "LineageNode",
    "LineageType",
    "Transformation",
]
