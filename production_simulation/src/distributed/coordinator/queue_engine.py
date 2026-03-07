"""
队列引擎

负责任务队列管理和优先级队列实现。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import heapq

from .models import DistributedTask, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class QueueEngine:
    """
    队列引擎

    负责任务队列管理和优先级队列实现
    """

    def __init__(self):
        self.task_queue = []  # 优先级队列
        self.queue_stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'peak_queue_size': 0
        }

        logger.info("队列引擎初始化完成")

    def enqueue_task(self, task: DistributedTask):
        """入队任务"""
        # 计算优先级分数
        priority_score = self._calculate_priority_score(task)

        # 使用堆队列保持优先级顺序
        heapq.heappush(self.task_queue, (-priority_score, task.created_time, task))

        # 更新统计
        self.queue_stats['total_enqueued'] += 1
        if len(self.task_queue) > self.queue_stats['peak_queue_size']:
            self.queue_stats['peak_queue_size'] = len(self.task_queue)

        logger.debug(f"任务 {task.task_id} 已入队，优先级分数: {priority_score}")

    def dequeue_task(self) -> Optional[DistributedTask]:
        """出队任务"""
        if not self.task_queue:
            return None

        # 从堆队列中取出优先级最高的任务
        _, _, task = heapq.heappop(self.task_queue)

        # 更新统计
        self.queue_stats['total_dequeued'] += 1

        logger.debug(f"任务 {task.task_id} 已出队")

        return task

    def peek_task(self) -> Optional[DistributedTask]:
        """查看队首任务（不出队）"""
        if not self.task_queue:
            return None

        # 查看堆顶任务
        _, _, task = self.task_queue[0]
        return task

    def remove_task(self, task_id: str) -> bool:
        """从队列中移除指定任务"""
        # 查找任务
        for i, (score, created_time, task) in enumerate(self.task_queue):
            if task.task_id == task_id:
                del self.task_queue[i]
                heapq.heapify(self.task_queue)
                logger.info(f"任务 {task_id} 已从队列移除")
                return True

        return False

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return len(self.task_queue)

    def get_queued_tasks(self) -> List[DistributedTask]:
        """获取所有排队任务"""
        return [task for _, _, task in self.task_queue]

    def _calculate_priority_score(self, task: DistributedTask) -> int:
        """计算任务优先级分数"""
        # 基础优先级
        base_score = task.priority.value * 100

        # 等待时间惩罚（越久等待，优先级越高）
        wait_time = (datetime.now() - task.created_time).total_seconds()
        age_bonus = int(wait_time / 60)  # 每分钟+1分

        # 重试次数惩罚（失败重试次数越多，优先级越高）
        retry_bonus = task.retry_count * 10

        return base_score + age_bonus + retry_bonus

    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        return {
            'current_size': len(self.task_queue),
            'total_enqueued': self.queue_stats['total_enqueued'],
            'total_dequeued': self.queue_stats['total_dequeued'],
            'peak_size': self.queue_stats['peak_queue_size']
        }

    def clear_queue(self):
        """清空队列"""
        self.task_queue.clear()
        logger.warning("任务队列已清空")


__all__ = ['QueueEngine']

