#!/usr/bin/env python3
"""
RQA2025 自动化层核心模型
Automation Layer Core Models

定义自动化层的数据模型和基础结构。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AutomationType(Enum):

    """自动化类型"""
    RULE_BASED = "rule_based"
    AI_DECISION = "ai_decision"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"


class ExecutionStatus(Enum):

    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):

    """任务优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AutomationRule:

    """自动化规则"""
    rule_id: str
    name: str
    description: str
    automation_type: AutomationType
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: TaskPriority = TaskPriority.MEDIUM
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutomationTask:

    """自动化任务"""
    task_id: str
    rule_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AutomationWorkflow:

    """自动化工作流"""
    workflow_id: str
    name: str
    description: str
    tasks: List[str]  # 任务ID列表
    dependencies: Dict[str, List[str]]  # 任务依赖关系
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class AutomationMetrics:

    """自动化指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    avg_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    rule_execution_count: Dict[str, int] = field(default_factory=dict)


@dataclass
class AIDecisionContext:

    """AI决策上下文"""
    context_id: str
    input_data: Dict[str, Any]
    decision_criteria: Dict[str, Any]
    historical_decisions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_threshold: float = 0.8
    max_processing_time_ms: int = 5000


@dataclass
class DeploymentConfig:

    """部署配置"""
    config_id: str
    service_name: str
    version: str
    environment: str
    config_data: Dict[str, Any]
    rollback_config: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:

    """扩展决策"""
    decision_id: str
    service_name: str
    current_load: Dict[str, Any]
    scaling_action: str  # scale_up, scale_down, no_action
    target_instances: int
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


__all__ = [
    'AutomationType', 'ExecutionStatus', 'TaskPriority',
    'AutomationRule', 'AutomationTask', 'AutomationWorkflow',
    'AutomationMetrics', 'AIDecisionContext', 'DeploymentConfig', 'ScalingDecision'
]
