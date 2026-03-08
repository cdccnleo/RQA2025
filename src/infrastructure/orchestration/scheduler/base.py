"""
调度器基础类和接口定义

提供调度器的基础抽象类和通用数据结构
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"           # 等待中
    RUNNING = "running"           # 运行中
    PAUSED = "paused"             # 已暂停
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消


class JobType(Enum):
    """任务类型枚举"""
    # 数据层任务
    DATA_COLLECTION = "data_collection"       # 数据采集
    DATA_CLEANING = "data_cleaning"           # 数据清洗
    DATA_VALIDATION = "data_validation"       # 数据验证

    # 特征层任务
    FEATURE_EXTRACTION = "feature_extraction" # 特征提取
    FEATURE_ENGINEERING = "feature_engineering" # 特征工程
    FEATURE_VALIDATION = "feature_validation" # 特征验证

    # 模型层任务
    MODEL_TRAINING = "model_training"         # 模型训练
    MODEL_VALIDATION = "model_validation"     # 模型验证
    MODEL_INFERENCE = "model_inference"       # 模型推理
    MODEL_DEPLOYMENT = "model_deployment"     # 模型部署

    # 策略层任务
    STRATEGY_BACKTEST = "strategy_backtest"   # 策略回测
    STRATEGY_OPTIMIZATION = "strategy_optimization" # 策略优化
    STRATEGY_VALIDATION = "strategy_validation" # 策略验证

    # 信号层任务
    SIGNAL_GENERATION = "signal_generation"   # 信号生成
    SIGNAL_FILTERING = "signal_filtering"     # 信号过滤
    SIGNAL_AGGREGATION = "signal_aggregation" # 信号聚合

    # 交易执行层任务
    ORDER_PREPARATION = "order_preparation"   # 订单准备
    ORDER_VALIDATION = "order_validation"     # 订单验证
    ORDER_EXECUTION = "order_execution"       # 订单执行
    ORDER_CONFIRMATION = "order_confirmation" # 订单确认

    # 风险控制层任务
    RISK_CALCULATION = "risk_calculation"     # 风险计算
    RISK_MONITORING = "risk_monitoring"       # 风险监控
    RISK_ALERTING = "risk_alerting"           # 风险告警
    POSITION_LIMIT_CHECK = "position_limit_check" # 仓位限制检查

    # 组合管理层任务
    PORTFOLIO_CONSTRUCTION = "portfolio_construction" # 组合构建
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"   # 组合再平衡
    PORTFOLIO_ANALYSIS = "portfolio_analysis"         # 组合分析

    # 传统任务类型（向后兼容）
    BACKTEST = "backtest"                     # 回测（兼容旧版）
    OPTIMIZATION = "optimization"             # 优化（兼容旧版）


class TriggerType(Enum):
    """触发器类型枚举"""
    INTERVAL = "interval"         # 间隔触发
    CRON = "cron"                 # Cron表达式
    DATE = "date"                 # 指定日期
    ONCE = "once"                 # 一次性


@dataclass
class Task:
    """任务数据类"""
    id: str
    type: str
    status: TaskStatus
    priority: int
    created_at: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None

    # 超时和重试相关字段
    timeout_seconds: Optional[int] = None           # 任务超时时间（秒）
    max_retries: int = 0                            # 最大重试次数
    retry_count: int = 0                            # 当前重试次数
    retry_delay_seconds: int = 0                    # 重试延迟（秒）
    deadline: Optional[datetime] = None             # 任务截止时间

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

        # 如果设置了超时时间但没有设置截止时间，自动计算截止时间
        if self.timeout_seconds and not self.deadline:
            self.deadline = self.created_at + __import__('datetime').timedelta(seconds=self.timeout_seconds)

    def is_timeout(self) -> bool:
        """检查任务是否已超时"""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline

    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED

    def get_remaining_time(self) -> Optional[float]:
        """获取剩余时间（秒）"""
        if not self.deadline:
            return None
        remaining = (self.deadline - datetime.now()).total_seconds()
        return max(0, remaining)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "worker_id": self.worker_id,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "is_timeout": self.is_timeout(),
            "remaining_time": self.get_remaining_time()
        }


@dataclass
class Job:
    """定时任务数据类"""
    id: str
    name: str
    job_type: JobType
    trigger_type: TriggerType
    trigger_config: Dict[str, Any]
    handler: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "job_type": self.job_type.value,
            "trigger_type": self.trigger_type.value,
            "trigger_config": self.trigger_config,
            "config": self.config,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count
        }


@dataclass
class WorkerInfo:
    """工作进程信息数据类"""
    id: str
    status: str  # "idle", "busy", "stopped"
    current_task: Optional[str] = None
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    task_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "status": self.status,
            "current_task": self.current_task,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "task_count": self.task_count
        }


class BaseScheduler(ABC):
    """
    调度器抽象基类
    
    定义所有调度器必须实现的接口
    """
    
    @abstractmethod
    async def start(self) -> bool:
        """
        启动调度器
        
        Returns:
            bool: 启动是否成功
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        停止调度器
        
        Returns:
            bool: 停止是否成功
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        检查调度器是否运行中
        
        Returns:
            bool: 是否运行中
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            Dict: 状态信息
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        pass


def generate_task_id() -> str:
    """生成唯一任务ID"""
    return f"task-{uuid4().hex[:8]}"


def generate_job_id() -> str:
    """生成唯一任务ID"""
    return f"job-{uuid4().hex[:8]}"


def generate_worker_id(index: int) -> str:
    """生成工作进程ID"""
    return f"worker-{index + 1}"
