"""
任务调度数据模型

任务调度相关的枚举和数据类。

从task_scheduler.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTask:
    """
    Scheduled Task Class
    调度任务类

    Represents a task to be executed at a specific time or with specific priority
    表示要在特定时间或具有特定优先级执行的任务
    """

    def __init__(self,
                 task_id: str,
                 func: Callable,
                 args: Optional[tuple] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 scheduled_time: Optional[datetime] = None,
                 timeout: Optional[float] = None,
                 retry_count: int = 0,
                 max_retries: int = 3):
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.scheduled_time = scheduled_time or datetime.now()
        self.timeout = timeout
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None

    def __lt__(self, other: 'ScheduledTask') -> bool:
        """比较优先级（用于优先级队列）"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.scheduled_time < other.scheduled_time

    def __repr__(self) -> str:
        return f"ScheduledTask(id={self.task_id}, priority={self.priority}, status={self.status})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'priority': self.priority.value,
            'status': self.status.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


__all__ = ['TaskPriority', 'TaskStatus', 'ScheduledTask']

