"""
AI Performance Optimizer - 数据模型

从原 ai_performance_optimizer.py 提取的数据模型和枚举定义
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


class OptimizationMode(Enum):
    """优化模式"""
    REACTIVE = "reactive"      # 反应式优化
    PREDICTIVE = "predictive"  # 预测性优化
    PROACTIVE = "proactive"    # 主动式优化
    ADAPTIVE = "adaptive"      # 自适应优化


class PerformanceMetric(Enum):
    """性能指标"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CONNECTION_COUNT = "connection_count"


@dataclass
class PerformanceData:
    """性能数据"""
    timestamp: datetime
    metrics: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """优化动作"""
    action_id: str
    action_type: str
    target_component: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"


@dataclass
class PerformanceInsight:
    """性能洞察"""
    insight_id: str
    insight_type: str
    description: str
    severity: str
    confidence: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


__all__ = [
    'OptimizationMode',
    'PerformanceMetric',
    'PerformanceData',
    'OptimizationAction',
    'PerformanceInsight',
]

