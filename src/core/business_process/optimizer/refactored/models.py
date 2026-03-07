"""
共享数据模型

重构后组件使用的通用数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum


class ProcessStage(Enum):
    """流程阶段枚举"""
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_GENERATION = "order_generation"
    EXECUTION_OPTIMIZATION = "execution_optimization"
    POSITION_MANAGEMENT = "position_management"
    PERFORMANCE_EVALUATION = "performance_evaluation"


@dataclass
class ProcessContext:
    """流程上下文"""
    process_id: str
    start_time: datetime
    current_stage: ProcessStage
    market_data: Dict[str, Any] = field(default_factory=dict)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    stage: ProcessStage
    recommendation_type: str
    description: str
    confidence: float
    expected_impact: Dict[str, Any]
    implementation_steps: List[str]
    priority: str  # "high", "medium", "low"
    timestamp: datetime

