"""
调度任务定义模块

提供调度任务的配置、状态管理和执行信息
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable


class JobStatus(Enum):
    """
    任务状态枚举
    
    定义调度任务的生命周期状态
    """
    PENDING = auto()      # 等待调度
    RUNNING = auto()      # 执行中
    COMPLETED = auto()    # 完成
    FAILED = auto()       # 失败
    PAUSED = auto()       # 暂停
    CANCELLED = auto()    # 已取消
    SCHEDULED = auto()    # 已调度


class TriggerType(Enum):
    """
    触发器类型枚举
    
    支持定时触发和事件触发两种模式
    """
    CRON = "cron"         # Cron表达式定时触发
    INTERVAL = "interval" # 间隔触发
    DATE = "date"         # 指定日期触发
    EVENT = "event"       # 事件触发
    ONCE = "once"         # 一次性执行


@dataclass
class JobTrigger:
    """
    任务触发器配置
    
    定义任务的触发方式和参数
    
    Attributes:
        trigger_type: 触发器类型
        cron_expression: Cron表达式（用于CRON类型）
        interval_seconds: 间隔秒数（用于INTERVAL类型）
        run_date: 指定执行日期（用于DATE类型）
        event_name: 事件名称（用于EVENT类型）
        event_filter: 事件过滤条件
        timezone: 时区设置
        jitter: 随机延迟秒数（用于分散负载）
        max_instances: 最大并发实例数
    """
    trigger_type: TriggerType
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_date: Optional[datetime] = None
    event_name: Optional[str] = None
    event_filter: Optional[Dict[str, Any]] = None
    timezone: str = "Asia/Shanghai"
    jitter: int = 0
    max_instances: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含触发器配置的字典
        """
        return {
            "trigger_type": self.trigger_type.value,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "run_date": self.run_date.isoformat() if self.run_date else None,
            "event_name": self.event_name,
            "event_filter": self.event_filter,
            "timezone": self.timezone,
            "jitter": self.jitter,
            "max_instances": self.max_instances
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobTrigger':
        """
        从字典创建触发器配置
        
        Args:
            data: 包含触发器配置的字典
            
        Returns:
            JobTrigger实例
        """
        run_date = None
        if data.get("run_date"):
            run_date = datetime.fromisoformat(data["run_date"])
        
        return cls(
            trigger_type=TriggerType(data.get("trigger_type", "once")),
            cron_expression=data.get("cron_expression"),
            interval_seconds=data.get("interval_seconds"),
            run_date=run_date,
            event_name=data.get("event_name"),
            event_filter=data.get("event_filter"),
            timezone=data.get("timezone", "Asia/Shanghai"),
            jitter=data.get("jitter", 0),
            max_instances=data.get("max_instances", 1)
        )
    
    @classmethod
    def cron(
        cls,
        expression: str,
        timezone: str = "Asia/Shanghai",
        jitter: int = 0
    ) -> 'JobTrigger':
        """
        创建Cron触发器
        
        Args:
            expression: Cron表达式（如 "0 9 * * 1-5" 表示工作日9点）
            timezone: 时区
            jitter: 随机延迟秒数
            
        Returns:
            Cron类型的JobTrigger实例
        """
        return cls(
            trigger_type=TriggerType.CRON,
            cron_expression=expression,
            timezone=timezone,
            jitter=jitter
        )
    
    @classmethod
    def interval(
        cls,
        seconds: int,
        timezone: str = "Asia/Shanghai"
    ) -> 'JobTrigger':
        """
        创建间隔触发器
        
        Args:
            seconds: 间隔秒数
            timezone: 时区
            
        Returns:
            INTERVAL类型的JobTrigger实例
        """
        return cls(
            trigger_type=TriggerType.INTERVAL,
            interval_seconds=seconds,
            timezone=timezone
        )
    
    @classmethod
    def event(
        cls,
        event_name: str,
        event_filter: Optional[Dict[str, Any]] = None
    ) -> 'JobTrigger':
        """
        创建事件触发器
        
        Args:
            event_name: 事件名称
            event_filter: 事件过滤条件
            
        Returns:
            EVENT类型的JobTrigger实例
        """
        return cls(
            trigger_type=TriggerType.EVENT,
            event_name=event_name,
            event_filter=event_filter
        )
    
    @classmethod
    def once(cls, run_date: Optional[datetime] = None) -> 'JobTrigger':
        """
        创建一次性触发器
        
        Args:
            run_date: 执行日期，None表示立即执行
            
        Returns:
            ONCE类型的JobTrigger实例
        """
        return cls(
            trigger_type=TriggerType.ONCE,
            run_date=run_date
        )


@dataclass
class JobExecutionHistory:
    """
    任务执行历史记录
    
    记录单次任务执行的详细信息
    
    Attributes:
        execution_id: 执行实例ID
        start_time: 开始时间
        end_time: 结束时间
        status: 执行状态
        result: 执行结果
        error_message: 错误信息
        retry_count: 重试次数
    """
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """
        计算执行时长
        
        Returns:
            执行时长（秒），如果未结束返回None
        """
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含执行历史的字典
        """
        return {
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status.name,
            "result": str(self.result) if self.result else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }


@dataclass
class ScheduleJob:
    """
    调度任务类
    
    定义调度任务的完整配置和状态，支持定时触发和事件触发
    
    Attributes:
        job_id: 任务唯一标识
        name: 任务名称
        description: 任务描述
        trigger: 触发器配置
        pipeline_config: 管道配置
        status: 当前状态
        created_at: 创建时间
        updated_at: 更新时间
        next_run_time: 下次执行时间
        last_run_time: 上次执行时间
        execution_count: 执行次数
        max_executions: 最大执行次数（None表示无限制）
        execution_history: 执行历史记录
        metadata: 元数据
        enabled: 是否启用
        timeout_seconds: 执行超时时间
        retry_count: 失败重试次数
        retry_delay_seconds: 重试间隔秒数
    """
    job_id: str
    name: str
    description: str = ""
    trigger: JobTrigger = field(default_factory=lambda: JobTrigger.once())
    pipeline_config: Optional[Dict[str, Any]] = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    next_run_time: Optional[datetime] = None
    last_run_time: Optional[datetime] = None
    execution_count: int = 0
    max_executions: Optional[int] = None
    execution_history: List[JobExecutionHistory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: int = 3600
    retry_count: int = 3
    retry_delay_seconds: int = 60
    
    def __post_init__(self):
        """
        初始化后处理
        
        如果没有提供job_id，自动生成UUID
        """
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含任务完整信息的字典
        """
        return {
            "job_id": self.job_id,
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger.to_dict(),
            "pipeline_config": self.pipeline_config,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "execution_count": self.execution_count,
            "max_executions": self.max_executions,
            "execution_history": [h.to_dict() for h in self.execution_history[-10:]],  # 只保留最近10条
            "metadata": self.metadata,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleJob':
        """
        从字典创建任务实例
        
        Args:
            data: 包含任务配置的字典
            
        Returns:
            ScheduleJob实例
        """
        trigger_data = data.get("trigger", {})
        trigger = JobTrigger.from_dict(trigger_data)
        
        # 解析时间字段
        next_run_time = None
        if data.get("next_run_time"):
            next_run_time = datetime.fromisoformat(data["next_run_time"])
        
        last_run_time = None
        if data.get("last_run_time"):
            last_run_time = datetime.fromisoformat(data["last_run_time"])
        
        created_at = datetime.now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        updated_at = datetime.now()
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])
        
        # 解析执行历史
        history = []
        for h in data.get("execution_history", []):
            history.append(JobExecutionHistory(
                execution_id=h["execution_id"],
                start_time=datetime.fromisoformat(h["start_time"]),
                end_time=datetime.fromisoformat(h["end_time"]) if h.get("end_time") else None,
                status=JobStatus[h.get("status", "PENDING")],
                result=h.get("result"),
                error_message=h.get("error_message"),
                retry_count=h.get("retry_count", 0)
            ))
        
        return cls(
            job_id=data.get("job_id", str(uuid.uuid4())),
            name=data.get("name", "unnamed_job"),
            description=data.get("description", ""),
            trigger=trigger,
            pipeline_config=data.get("pipeline_config"),
            status=JobStatus[data.get("status", "PENDING")],
            created_at=created_at,
            updated_at=updated_at,
            next_run_time=next_run_time,
            last_run_time=last_run_time,
            execution_count=data.get("execution_count", 0),
            max_executions=data.get("max_executions"),
            execution_history=history,
            metadata=data.get("metadata", {}),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 3600),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 60)
        )
    
    def update_next_run_time(self, next_time: datetime) -> None:
        """
        更新下次执行时间
        
        Args:
            next_time: 下次执行时间
        """
        self.next_run_time = next_time
        self.updated_at = datetime.now()
    
    def record_execution(
        self,
        execution_id: str,
        status: JobStatus,
        result: Optional[Any] = None,
        error_message: Optional[str] = None
    ) -> JobExecutionHistory:
        """
        记录任务执行
        
        Args:
            execution_id: 执行实例ID
            status: 执行状态
            result: 执行结果
            error_message: 错误信息
            
        Returns:
            执行历史记录
        """
        now = datetime.now()
        
        # 查找或创建执行记录
        history = None
        for h in self.execution_history:
            if h.execution_id == execution_id:
                history = h
                break
        
        if history is None:
            history = JobExecutionHistory(
                execution_id=execution_id,
                start_time=now,
                status=status
            )
            self.execution_history.append(history)
        
        # 更新记录
        history.status = status
        history.result = result
        history.error_message = error_message
        
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            history.end_time = now
            self.last_run_time = now
            self.execution_count += 1
        
        self.status = status
        self.updated_at = now
        
        # 限制历史记录数量
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return history
    
    def can_execute(self) -> bool:
        """
        检查任务是否可以执行
        
        Returns:
            是否可以执行
        """
        if not self.enabled:
            return False
        
        if self.status == JobStatus.PAUSED:
            return False
        
        if self.max_executions is not None and self.execution_count >= self.max_executions:
            return False
        
        return True
    
    def pause(self) -> None:
        """
        暂停任务
        """
        self.status = JobStatus.PAUSED
        self.enabled = False
        self.updated_at = datetime.now()
    
    def resume(self) -> None:
        """
        恢复任务
        """
        self.status = JobStatus.SCHEDULED
        self.enabled = True
        self.updated_at = datetime.now()
    
    def cancel(self) -> None:
        """
        取消任务
        """
        self.status = JobStatus.CANCELLED
        self.enabled = False
        self.updated_at = datetime.now()
    
    def get_last_execution(self) -> Optional[JobExecutionHistory]:
        """
        获取最近一次执行记录
        
        Returns:
            最近的执行历史记录，如果没有返回None
        """
        if self.execution_history:
            return self.execution_history[-1]
        return None
    
    def get_success_rate(self) -> float:
        """
        计算成功率
        
        Returns:
            成功率（0.0 - 1.0）
        """
        if not self.execution_history:
            return 0.0
        
        completed = sum(
            1 for h in self.execution_history
            if h.status == JobStatus.COMPLETED
        )
        return completed / len(self.execution_history)


# 常用Cron表达式预设
CRON_PRESETS = {
    "minutely": "* * * * *",           # 每分钟
    "hourly": "0 * * * *",             # 每小时
    "daily": "0 0 * * *",              # 每天零点
    "daily_9am": "0 9 * * *",          # 每天9点
    "weekly": "0 0 * * 0",             # 每周日零点
    "weekdays_9am": "0 9 * * 1-5",     # 工作日9点
    "monthly": "0 0 1 * *",            # 每月1号零点
    "quarterly": "0 0 1 1,4,7,10 *"    # 每季度首月1号
}


def create_job(
    name: str,
    trigger: JobTrigger,
    pipeline_config: Optional[Dict[str, Any]] = None,
    description: str = "",
    **kwargs
) -> ScheduleJob:
    """
    创建调度任务的工厂函数
    
    Args:
        name: 任务名称
        trigger: 触发器配置
        pipeline_config: 管道配置
        description: 任务描述
        **kwargs: 其他任务参数
        
    Returns:
        ScheduleJob实例
    """
    return ScheduleJob(
        job_id=str(uuid.uuid4()),
        name=name,
        description=description,
        trigger=trigger,
        pipeline_config=pipeline_config,
        **kwargs
    )
