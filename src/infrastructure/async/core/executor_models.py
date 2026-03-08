"""
执行器数据模型

执行器的枚举和指标类。

从executor_manager.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Dict, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Types of executors"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    SINGLE_THREAD = "single_thread"


class ExecutorStatus(Enum):
    """Executor status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutorMetrics:
    """
    Executor Metrics Class
    执行器指标类

    Tracks performance metrics for executors
    跟踪执行器的性能指标
    """

    def __init__(self):
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self.peak_active_threads = 0
        self.current_active_threads = 0
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def record_task_start(self):
        """Record task start"""
        self.tasks_submitted += 1
        self.current_active_threads += 1
        self.peak_active_threads = max(self.peak_active_threads, self.current_active_threads)
        self.last_updated = datetime.now()

    def record_task_completion(self, execution_time: float):
        """Record task completion"""
        self.tasks_completed += 1
        self.current_active_threads -= 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.tasks_completed
        self.last_updated = datetime.now()

    def record_task_failure(self):
        """Record task failure"""
        self.tasks_failed += 1
        self.current_active_threads -= 1
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'tasks_submitted': self.tasks_submitted,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'success_rate': (self.tasks_completed / max(self.tasks_submitted, 1)) * 100,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.average_execution_time,
            'peak_active_threads': self.peak_active_threads,
            'current_active_threads': self.current_active_threads,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


__all__ = ['ExecutorType', 'ExecutorStatus', 'ExecutorMetrics']

