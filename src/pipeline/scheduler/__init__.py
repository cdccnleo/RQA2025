"""
Pipeline Scheduler 模块

提供统一的任务调度管理功能，支持定时触发（Cron）和事件触发（Event）两种模式。

主要组件:
    - UnifiedScheduler: 统一调度器，管理任务的创建、执行、暂停、恢复和删除
    - ScheduleJob: 调度任务类，定义任务配置、状态和执行信息
    - JobTrigger: 任务触发器，支持Cron、Interval、Date、Event等多种触发方式

使用示例:
    >>> from src.pipeline.scheduler import UnifiedScheduler, JobTrigger
    >>> 
    >>> # 创建调度器
    >>> scheduler = UnifiedScheduler(max_workers=4)
    >>> scheduler.start()
    >>> 
    >>> # 创建定时任务（Cron表达式）
    >>> job = scheduler.create_job(
    ...     name="daily_training",
    ...     trigger=JobTrigger.cron("0 2 * * *"),  # 每天凌晨2点
    ...     pipeline_config={...}
    ... )
    >>> 
    >>> # 创建事件触发任务
    >>> event_job = scheduler.create_job(
    ...     name="on_data_ready",
    ...     trigger=JobTrigger.event("data_ready"),
    ...     pipeline_config={...}
    ... )
    >>> 
    >>> # 触发事件
    >>> scheduler.emit_event("data_ready", {"symbol": "000001.SZ"})
    >>> 
    >>> # 暂停/恢复任务
    >>> scheduler.pause_job(job.job_id)
    >>> scheduler.resume_job(job.job_id)
    >>> 
    >>> # 停止调度器
    >>> scheduler.stop()
"""

from .schedule_job import (
    ScheduleJob,
    JobTrigger,
    JobStatus,
    TriggerType,
    JobExecutionHistory,
    CRON_PRESETS,
    create_job
)
from .unified_scheduler import UnifiedScheduler, SchedulerException

__all__ = [
    # 核心类
    "UnifiedScheduler",
    "ScheduleJob",
    "JobTrigger",
    
    # 枚举
    "JobStatus",
    "TriggerType",
    
    # 数据类
    "JobExecutionHistory",
    
    # 异常
    "SchedulerException",
    
    # 常量
    "CRON_PRESETS",
    
    # 工具函数
    "create_job"
]

__version__ = "1.0.0"
