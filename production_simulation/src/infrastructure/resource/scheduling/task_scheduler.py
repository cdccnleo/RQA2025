"""
task_scheduler 模块

提供 task_scheduler 相关功能和接口。
"""

"""
任务调度器 (重构版本)

此文件现在使用重构后的模块化设计。
原来的大类TaskScheduler已被拆分为多个职责单一的专用类。

重构详情:
- TaskManager: 任务生命周期管理
- TaskQueueManager: 队列管理
- TaskWorkerManager: 工作线程管理
- TaskSchedulerCore: 调度核心逻辑
- TaskMonitor: 任务监控和统计
- TaskSchedulerFacade: 门面类 (向后兼容)

保持向后兼容性，现有代码无需修改。
"""

from .task_scheduler_refactored import (
    TaskScheduler as TaskSchedulerImpl,
    TaskPriority,
    TaskStatus,
    Task,
    TaskNotFoundException,
    TaskNotCompletedError
)

# 向后兼容的别名
TaskScheduler = TaskSchedulerImpl

# 导出其他兼容性符号
__all__ = [
    'TaskScheduler',
    'TaskPriority',
    'TaskStatus',
    'Task',
    'TaskNotFoundException',
    'TaskNotCompletedError'
]
