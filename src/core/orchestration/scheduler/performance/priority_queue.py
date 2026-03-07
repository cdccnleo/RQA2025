"""
优先级任务队列

基于堆实现的优先级队列，支持O(log n)的插入和提取操作
"""

import heapq
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 1    # 关键任务（如风险控制）
    HIGH = 2        # 高优先级（如交易执行）
    NORMAL = 3      # 普通优先级（如数据同步）
    LOW = 4         # 低优先级（如报表生成）
    BACKGROUND = 5  # 后台任务（如数据清理）


@dataclass(order=True)
class PrioritizedTask:
    """
    带优先级的任务

    使用dataclass自动实现比较方法，用于堆排序
    """
    priority_value: int  # 优先级数值（越小优先级越高）
    sequence: int       # 序列号，用于相同优先级的FIFO排序
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    created_at: datetime = field(compare=False, default_factory=datetime.now)
    timeout_seconds: Optional[int] = field(compare=False, default=None)


class PriorityTaskQueue:
    """
    优先级任务队列

    基于堆实现的高效优先级队列，特点：
    - O(log n) 插入和提取
    - 支持动态优先级调整
    - 支持按任务类型分组统计
    - 线程安全
    """

    def __init__(self):
        """初始化优先级队列"""
        self._heap: List[PrioritizedTask] = []
        self._task_map: Dict[str, PrioritizedTask] = {}  # 任务ID到任务的映射
        self._lock = threading.RLock()
        self._sequence = 0  # 序列号计数器
        self._stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'priority_distribution': {p: 0 for p in TaskPriority}
        }

    def enqueue(
        self,
        task_id: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        payload: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """
        将任务加入队列

        Args:
            task_id: 任务ID
            task_type: 任务类型
            priority: 任务优先级
            payload: 任务数据
            timeout_seconds: 超时时间

        Returns:
            bool: 是否成功加入队列
        """
        with self._lock:
            if task_id in self._task_map:
                return False  # 任务已存在

            # 生成序列号保证FIFO
            self._sequence += 1

            # 创建优先级任务
            task = PrioritizedTask(
                priority_value=priority.value,
                sequence=self._sequence,
                task_id=task_id,
                task_type=task_type,
                payload=payload or {},
                timeout_seconds=timeout_seconds
            )

            # 加入堆
            heapq.heappush(self._heap, task)
            self._task_map[task_id] = task

            # 更新统计
            self._stats['total_enqueued'] += 1
            self._stats['priority_distribution'][priority] += 1

            return True

    def dequeue(self) -> Optional[PrioritizedTask]:
        """
        从队列中提取优先级最高的任务

        Returns:
            Optional[PrioritizedTask]: 优先级最高的任务，队列为空则返回None
        """
        with self._lock:
            while self._heap:
                task = heapq.heappop(self._heap)

                # 检查任务是否仍在映射中（可能已被移除）
                if task.task_id in self._task_map:
                    del self._task_map[task.task_id]
                    self._stats['total_dequeued'] += 1
                    return task

            return None

    def peek(self) -> Optional[PrioritizedTask]:
        """
        查看队列中优先级最高的任务（不移除）

        Returns:
            Optional[PrioritizedTask]: 优先级最高的任务
        """
        with self._lock:
            while self._heap:
                task = self._heap[0]
                if task.task_id in self._task_map:
                    return task
                # 任务已被移除，弹出
                heapq.heappop(self._heap)
            return None

    def remove(self, task_id: str) -> bool:
        """
        从队列中移除指定任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            if task_id not in self._task_map:
                return False

            # 从映射中移除（实际从堆中移除会在dequeue时处理）
            task = self._task_map.pop(task_id)

            # 更新统计
            priority = TaskPriority(task.priority_value)
            self._stats['priority_distribution'][priority] -= 1

            return True

    def update_priority(
        self,
        task_id: str,
        new_priority: TaskPriority
    ) -> bool:
        """
        更新任务优先级

        Args:
            task_id: 任务ID
            new_priority: 新优先级

        Returns:
            bool: 是否成功更新
        """
        with self._lock:
            if task_id not in self._task_map:
                return False

            old_task = self._task_map[task_id]
            old_priority = TaskPriority(old_task.priority_value)

            # 如果优先级相同，无需更新
            if old_task.priority_value == new_priority.value:
                return True

            # 创建新任务（堆不支持直接更新）
            self._sequence += 1
            new_task = PrioritizedTask(
                priority_value=new_priority.value,
                sequence=self._sequence,
                task_id=old_task.task_id,
                task_type=old_task.task_type,
                payload=old_task.payload,
                created_at=old_task.created_at,
                timeout_seconds=old_task.timeout_seconds
            )

            # 标记旧任务为已移除
            del self._task_map[task_id]

            # 加入新任务
            heapq.heappush(self._heap, new_task)
            self._task_map[task_id] = new_task

            # 更新统计
            self._stats['priority_distribution'][old_priority] -= 1
            self._stats['priority_distribution'][new_priority] += 1

            return True

    def get_position(self, task_id: str) -> Optional[int]:
        """
        获取任务在队列中的位置

        Args:
            task_id: 任务ID

        Returns:
            Optional[int]: 位置（0为队首），不存在则返回None
        """
        with self._lock:
            if task_id not in self._task_map:
                return None

            # 遍历堆查找位置
            position = 0
            for task in sorted(self._heap):
                if task.task_id == task_id:
                    return position
                if task.task_id in self._task_map:
                    position += 1
            return None

    def get_task(self, task_id: str) -> Optional[PrioritizedTask]:
        """
        获取任务信息

        Args:
            task_id: 任务ID

        Returns:
            Optional[PrioritizedTask]: 任务信息
        """
        with self._lock:
            return self._task_map.get(task_id)

    def is_empty(self) -> bool:
        """
        检查队列是否为空

        Returns:
            bool: 是否为空
        """
        with self._lock:
            return len(self._task_map) == 0

    def size(self) -> int:
        """
        获取队列大小

        Returns:
            int: 队列中的任务数
        """
        with self._lock:
            return len(self._task_map)

    def clear(self) -> int:
        """
        清空队列

        Returns:
            int: 清空的任务数
        """
        with self._lock:
            count = len(self._task_map)
            self._heap.clear()
            self._task_map.clear()
            self._stats = {
                'total_enqueued': 0,
                'total_dequeued': 0,
                'priority_distribution': {p: 0 for p in TaskPriority}
            }
            return count

    def get_all_tasks(self) -> List[PrioritizedTask]:
        """
        获取队列中所有任务（按优先级排序）

        Returns:
            List[PrioritizedTask]: 任务列表
        """
        with self._lock:
            return sorted(
                [t for t in self._heap if t.task_id in self._task_map]
            )

    def get_tasks_by_type(self, task_type: str) -> List[PrioritizedTask]:
        """
        获取指定类型的所有任务

        Args:
            task_type: 任务类型

        Returns:
            List[PrioritizedTask]: 任务列表
        """
        with self._lock:
            return [
                t for t in self._heap
                if t.task_id in self._task_map and t.task_type == task_type
            ]

    def get_tasks_by_priority(self, priority: TaskPriority) -> List[PrioritizedTask]:
        """
        获取指定优先级的所有任务

        Args:
            priority: 优先级

        Returns:
            List[PrioritizedTask]: 任务列表
        """
        with self._lock:
            return [
                t for t in self._heap
                if t.task_id in self._task_map and t.priority_value == priority.value
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取队列统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            return {
                'size': len(self._task_map),
                'total_enqueued': self._stats['total_enqueued'],
                'total_dequeued': self._stats['total_dequeued'],
                'priority_distribution': {
                    p.name: count
                    for p, count in self._stats['priority_distribution'].items()
                }
            }

    def get_wait_time_stats(self) -> Dict[str, float]:
        """
        获取任务等待时间统计

        Returns:
            Dict[str, float]: 等待时间统计（秒）
        """
        with self._lock:
            if not self._task_map:
                return {'avg_wait_time': 0, 'max_wait_time': 0, 'min_wait_time': 0}

            now = datetime.now()
            wait_times = [
                (now - task.created_at).total_seconds()
                for task in self._task_map.values()
            ]

            return {
                'avg_wait_time': sum(wait_times) / len(wait_times),
                'max_wait_time': max(wait_times),
                'min_wait_time': min(wait_times)
            }
