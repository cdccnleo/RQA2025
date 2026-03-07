"""
任务调度器模块（别名模块）
提供向后兼容的导入路径

实际实现在 resource/scheduling/task_scheduler.py 中
"""

try:
    from .scheduling.task_scheduler import TaskScheduler
except ImportError:
    try:
        from .scheduling.task_scheduler_refactored import TaskScheduler
    except ImportError:
        # 提供基础实现
        class TaskScheduler:
            pass

__all__ = ['TaskScheduler']

