"""
统一调度器模块

提供系统唯一的调度器实现，支持任务调度、定时任务、工作进程管理等功能。

使用示例:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    # 获取调度器实例
    scheduler = get_unified_scheduler()
    
    # 启动调度器
    await scheduler.start()
    
    # 提交任务
    task_id = await scheduler.submit_task(
        task_type="data_collection",
        payload={"source": "alpha_vantage"},
        priority=5
    )
    
    # 获取任务状态
    task = scheduler.get_task_detail(task_id)
    
    # 停止调度器
    await scheduler.stop()
"""

from .base import (
    TaskStatus,
    JobType,
    TriggerType,
    Task,
    Job,
    WorkerInfo,
    BaseScheduler,
    generate_task_id,
    generate_job_id,
    generate_worker_id
)

from .task_manager import TaskManager
from .worker_manager import WorkerManager
from .unified_scheduler import UnifiedScheduler, get_unified_scheduler

__all__ = [
    # 枚举类型
    "TaskStatus",
    "JobType",
    "TriggerType",
    
    # 数据类
    "Task",
    "Job",
    "WorkerInfo",
    
    # 管理器类
    "TaskManager",
    "WorkerManager",
    
    # 调度器
    "BaseScheduler",
    "UnifiedScheduler",
    "get_unified_scheduler",
    
    # 工具函数
    "generate_task_id",
    "generate_job_id",
    "generate_worker_id",
]

__version__ = "1.0.0"
