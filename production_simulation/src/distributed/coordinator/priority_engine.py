"""
优先级引擎

负责任务优先级管理和动态优先级调整。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .models import DistributedTask, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class PriorityEngine:
    """
    优先级引擎

    负责任务优先级管理和动态优先级调整
    """

    def __init__(self):
        self.priority_adjustments = []
        self.aging_enabled = True  # 启用任务老化（等待久的任务提升优先级）
        self.aging_threshold = timedelta(minutes=30)  # 老化阈值

        logger.info("优先级引擎初始化完成")

    def adjust_priority(self,
                       task: DistributedTask,
                       reason: str = 'manual') -> TaskPriority:
        """
        调整任务优先级

        Args:
            task: 待调整优先级的任务
            reason: 调整原因

        Returns:
            调整后的优先级
        """
        original_priority = task.priority

        # 根据原因调整优先级
        if reason == 'timeout_approaching':
            # 超时临近，提升优先级
            task.priority = self._increase_priority(task.priority)

        elif reason == 'retry':
            # 重试任务，提升优先级
            if task.retry_count > 1:
                task.priority = self._increase_priority(task.priority)

        elif reason == 'dependency_completed':
            # 依赖完成，保持或降低优先级
            pass

        elif reason == 'aging':
            # 任务老化，提升优先级
            wait_time = datetime.now() - task.created_time
            if wait_time > self.aging_threshold:
                task.priority = self._increase_priority(task.priority)

        # 记录优先级调整
        self._record_priority_adjustment(task, original_priority, reason)

        logger.info(
            f"任务 {task.task_id} 优先级调整: "
            f"{original_priority.name} → {task.priority.name} (原因: {reason})"
        )

        return task.priority

    def _increase_priority(self, current_priority: TaskPriority) -> TaskPriority:
        """提升优先级"""
        if current_priority == TaskPriority.LOW:
            return TaskPriority.NORMAL
        elif current_priority == TaskPriority.NORMAL:
            return TaskPriority.HIGH
        elif current_priority == TaskPriority.HIGH:
            return TaskPriority.CRITICAL
        else:
            return TaskPriority.CRITICAL

    def _decrease_priority(self, current_priority: TaskPriority) -> TaskPriority:
        """降低优先级"""
        if current_priority == TaskPriority.CRITICAL:
            return TaskPriority.HIGH
        elif current_priority == TaskPriority.HIGH:
            return TaskPriority.NORMAL
        elif current_priority == TaskPriority.NORMAL:
            return TaskPriority.LOW
        else:
            return TaskPriority.LOW

    def check_aging_tasks(self, tasks: List[DistributedTask]) -> List[DistributedTask]:
        """检查并调整老化任务的优先级"""
        if not self.aging_enabled:
            return []

        aged_tasks = []

        for task in tasks:
            if task.status != TaskStatus.PENDING:
                continue

            wait_time = datetime.now() - task.created_time

            if wait_time > self.aging_threshold:
                self.adjust_priority(task, reason='aging')
                aged_tasks.append(task)

        return aged_tasks

    def _record_priority_adjustment(self,
                                   task: DistributedTask,
                                   original_priority: TaskPriority,
                                   reason: str):
        """记录优先级调整"""
        record = {
            'task_id': task.task_id,
            'original_priority': original_priority.name,
            'new_priority': task.priority.name,
            'reason': reason,
            'timestamp': datetime.now()
        }

        self.priority_adjustments.append(record)

        # 保持历史记录在合理范围内
        if len(self.priority_adjustments) > 10000:
            self.priority_adjustments = self.priority_adjustments[-10000:]

    def get_priority_stats(self) -> Dict[str, Any]:
        """获取优先级统计"""
        if not self.priority_adjustments:
            return {}

        recent_adjustments = self.priority_adjustments[-1000:]

        reason_counts = {}
        for record in recent_adjustments:
            reason = record['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'total_adjustments': len(self.priority_adjustments),
            'recent_adjustments': len(recent_adjustments),
            'reason_distribution': reason_counts,
            'aging_enabled': self.aging_enabled,
            'aging_threshold_minutes': self.aging_threshold.total_seconds() / 60
        }


__all__ = ['PriorityEngine']

