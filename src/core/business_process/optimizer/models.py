"""
智能业务流程优化器数据模型

定义优化器使用的核心数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


# ==================== 枚举类型 ====================

class ProcessStage(Enum):
    """流程阶段枚举"""
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_GENERATION = "order_generation"
    EXECUTION_OPTIMIZATION = "execution_optimization"
    POSITION_MANAGEMENT = "position_management"
    PERFORMANCE_EVALUATION = "performance_evaluation"


class OptimizationStatus(Enum):
    """优化状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DecisionStrategy(Enum):
    """决策策略枚举"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    AI_OPTIMIZED = "ai_optimized"


class DecisionType(Enum):
    """决策类型"""
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    HOLD_SIGNAL = "hold_signal"
    RISK_ADJUSTMENT = "risk_adjustment"
    POSITION_REBALANCE = "position_rebalance"
    MARKET_EXIT = "market_exit"


# ==================== 数据类 ====================

@dataclass
class ProcessContext:
    """
    流程上下文

    包含流程执行所需的所有上下文信息
    """
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

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'process_id': self.process_id,
            'start_time': self.start_time.isoformat(),
            'current_stage': self.current_stage.value,
            'market_data': self.market_data,
            'signals': self.signals,
            'risk_assessment': self.risk_assessment,
            'orders': self.orders,
            'execution_results': self.execution_results,
            'performance_metrics': self.performance_metrics,
            'decisions': self.decisions,
            'metadata': self.metadata
        }


@dataclass
class OptimizationResult:
    """
    优化结果

    包含流程优化的完整结果
    """
    process_id: str
    status: OptimizationStatus
    stages: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    performance: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Any] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'process_id': self.process_id,
            'status': self.status.value,
            'stages': self.stages,
            'decisions': self.decisions,
            'performance': self.performance,
            'recommendations': [
                rec.to_dict() if hasattr(rec, 'to_dict') else rec
                for rec in self.recommendations
            ],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'errors': self.errors,
            'metadata': self.metadata
        }

    @property
    def is_successful(self) -> bool:
        """判断是否成功"""
        return self.status == OptimizationStatus.COMPLETED and not self.errors

    @property
    def overall_score(self) -> float:
        """综合评分"""
        return self.performance.get('overall_score', 0.0)


@dataclass
class StageResult:
    """
    阶段结果

    单个流程阶段的执行结果
    """
    stage: ProcessStage
    status: str  # 'completed', 'failed', 'skipped'
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage': self.stage.value,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'output_data': self.output_data,
            'decisions': self.decisions,
            'metrics': self.metrics,
            'errors': self.errors
        }


@dataclass
class PerformanceMetrics:
    """
    性能指标

    流程性能的详细指标
    """
    process_id: str
    timestamp: datetime
    overall_score: float
    execution_efficiency: float
    decision_quality: float
    resource_utilization: float
    error_rate: float
    success_rate: float
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'process_id': self.process_id,
            'timestamp': self.timestamp.isoformat(),
            'overall_score': self.overall_score,
            'execution_efficiency': self.execution_efficiency,
            'decision_quality': self.decision_quality,
            'resource_utilization': self.resource_utilization,
            'error_rate': self.error_rate,
            'success_rate': self.success_rate,
            'stage_metrics': self.stage_metrics,
            'custom_metrics': self.custom_metrics
        }


# ==================== 辅助函数 ====================

def create_process_context(process_id: str,
                          market_data: Dict[str, Any],
                          **kwargs) -> ProcessContext:
    """
    创建流程上下文（便捷函数）

    Args:
        process_id: 流程ID
        market_data: 市场数据
        **kwargs: 其他参数

    Returns:
        ProcessContext: 流程上下文对象
    """
    return ProcessContext(
        process_id=process_id,
        start_time=datetime.now(),
        current_stage=ProcessStage.MARKET_ANALYSIS,
        market_data=market_data,
        **kwargs
    )


def create_optimization_result(process_id: str,
                               status: OptimizationStatus = OptimizationStatus.PENDING) -> OptimizationResult:
    """
    创建优化结果（便捷函数）

    Args:
        process_id: 流程ID
        status: 初始状态

    Returns:
        OptimizationResult: 优化结果对象
    """
    return OptimizationResult(
        process_id=process_id,
        status=status,
        start_time=datetime.now()
    )


def create_stage_result(stage: ProcessStage,
                       status: str = 'pending') -> StageResult:
    """
    创建阶段结果（便捷函数）

    Args:
        stage: 流程阶段
        status: 状态

    Returns:
        StageResult: 阶段结果对象
    """
    return StageResult(
        stage=stage,
        status=status,
        start_time=datetime.now()
    )
