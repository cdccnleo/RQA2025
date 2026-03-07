"""
独立的TaskConcurrencyController测试模块
"""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import threading
import time
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TaskConcurrencyController:

    """
    任务并发控制器 - 防止资源竞争和死锁

    Task concurrency controller to prevent resource competition and deadlocks
    """

    def __init__(self, max_concurrent_tasks: int = 10, deadlock_timeout: float = 300.0):
        """
        初始化任务并发控制器

        Initialize task concurrency controller

        Args:
            max_concurrent_tasks: 最大并发任务数
                                Maximum number of concurrent tasks
            deadlock_timeout: 死锁检测超时时间（秒）
                           Deadlock detection timeout (seconds)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.deadlock_timeout = deadlock_timeout

        # 任务状态管理
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: deque = deque(maxlen=1000)

        # 同步原语
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)

        # 资源依赖管理（用于死锁检测）
        self.resource_dependencies: Dict[str, set] = defaultdict(set)
        self.task_resources: Dict[str, set] = defaultdict(set)

        # 性能统计
        self.stats = {
            'total_tasks_processed': 0,
            'deadlock_detected': 0,
            'queue_full_rejections': 0,
            'timeout_rejections': 0,
            'average_wait_time': 0.0,
            'max_wait_time': 0.0
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def acquire_task_slot(self, task_id: str, required_resources: Optional[set] = None) -> bool:
        """
        获取任务执行槽位

        Acquire task execution slot

        Args:
            task_id: 任务ID
                    Task ID
            required_resources: 需要的资源集合
                              Required resources set

        Returns:
            bool: 是否成功获取
                 Whether acquisition was successful
        """
        start_time = time.time()

        with self.condition:
            # 检查是否已存在相同任务
            if task_id in self.active_tasks:
                self.logger.warning(f"Task {task_id} is already running")
                return False

            # 检查死锁风险
            if required_resources and self._would_cause_deadlock(task_id, required_resources):
                self.stats['deadlock_detected'] += 1
                self.logger.warning(f"Deadlock risk detected for task {task_id}")
                return False

            # 等待可用槽位
            while len(self.active_tasks) >= self.max_concurrent_tasks:
                if not self.condition.wait(timeout=self.deadlock_timeout):
                    self.stats['timeout_rejections'] += 1
                    self.logger.warning(f"Timeout waiting for task slot: {task_id}")
                    return False

                # 再次检查死锁风险（因为等待期间状态可能改变）
                if required_resources and self._would_cause_deadlock(task_id, required_resources):
                    self.stats['deadlock_detected'] += 1
                    self.logger.warning(f"Deadlock risk detected after wait for task {task_id}")
                    return False

            # 获取槽位
            self.active_tasks[task_id] = {
                'start_time': datetime.now(),
                'required_resources': required_resources or set(),
                'wait_time': time.time() - start_time
            }

            # 更新资源依赖
            if required_resources:
                self.task_resources[task_id] = required_resources
                for resource in required_resources:
                    self.resource_dependencies[resource].add(task_id)

            # 更新统计信息
            wait_time = time.time() - start_time
            self.stats['total_tasks_processed'] += 1
            self.stats['average_wait_time'] = (
                (self.stats['average_wait_time'] *
                 (self.stats['total_tasks_processed'] - 1)) + wait_time
            ) / self.stats['total_tasks_processed']
            self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)

            self.logger.info(f"Task slot acquired: {task_id}")
            return True

    def release_task_slot(self, task_id: str):
        """
        释放任务执行槽位

        Release task execution slot

        Args:
            task_id: 任务ID
                    Task ID
        """
        with self.condition:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                execution_time = (datetime.now() - task_info['start_time']).total_seconds()

                # 记录完成的任务
                self.completed_tasks.append({
                    'task_id': task_id,
                    'execution_time': execution_time,
                    'wait_time': task_info['wait_time'],
                    'completed_at': datetime.now()
                })

                # 清理资源依赖
                required_resources = task_info.get('required_resources', set())
                for resource in required_resources:
                    self.resource_dependencies[resource].discard(task_id)
                    if not self.resource_dependencies[resource]:
                        del self.resource_dependencies[resource]

                if task_id in self.task_resources:
                    del self.task_resources[task_id]

                # 释放槽位
                del self.active_tasks[task_id]

                # 通知等待的线程
                self.condition.notify()

                self.logger.info(
                    f"Task slot released: {task_id} (execution: {execution_time:.2f}s)")
            else:
                self.logger.warning(f"Attempt to release non - existent task: {task_id}")

    def _would_cause_deadlock(self, task_id: str, required_resources: set) -> bool:
        """
        检查是否会造成死锁

        Check if operation would cause deadlock

        Args:
            task_id: 任务ID
                    Task ID
            required_resources: 需要的资源
                              Required resources

        Returns:
            bool: 是否会造成死锁
                 Whether it would cause deadlock
        """
        # 简单的死锁检测：检查是否有循环等待
        # 这里实现一个简化的死锁检测算法

        # 获取所有当前活跃任务
        active_task_ids = set(self.active_tasks.keys())

        # 检查是否存在资源冲突
        for resource in required_resources:
            holders = self.resource_dependencies.get(resource, set())
            # 如果某个资源被其他任务持有，检查是否存在循环依赖
            for holder in holders:
                if holder in active_task_ids:
                    # 检查holder任务是否也在等待当前任务持有的资源
                    holder_resources = self.task_resources.get(holder, set())
                    if holder_resources.intersection(self._get_task_held_resources(task_id)):
                        return True

        return False

    def _get_task_held_resources(self, task_id: str) -> set:
        """
        获取任务持有的资源

        Get resources held by task

        Args:
            task_id: 任务ID

        Returns:
            set: 持有的资源集合
        """
        if task_id not in self.active_tasks:
            return set()

        return self.active_tasks[task_id].get('required_resources', set())

    async def execute_with_control(self, task_id: str, task_func: Callable, *args,
                                   required_resources: Optional[set] = None, **kwargs):
        """
        受控执行任务（异步版本）

        Controlled task execution (async version)

        Args:
            task_id: 任务ID
                    Task ID
            task_func: 任务函数
                      Task function
            required_resources: 需要的资源
                              Required resources
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务执行结果
            Task execution result
        """
        if not self.acquire_task_slot(task_id, required_resources):
            raise RuntimeError(f"Failed to acquire task slot for {task_id}")

        try:
            # 执行任务
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(executor, task_func, *args, **kwargs)

            return result

        finally:
            self.release_task_slot(task_id)

    def execute_with_control_sync(self, task_id: str, task_func: Callable, *args,


                                  required_resources: Optional[set] = None, **kwargs):
        """
        受控执行任务（同步版本）

        Controlled task execution (sync version)

        Args:
            task_id: 任务ID
                    Task ID
            task_func: 任务函数
                      Task function
            required_resources: 需要的资源
                              Required resources
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务执行结果
            Task execution result
        """
        if not self.acquire_task_slot(task_id, required_resources):
            raise RuntimeError(f"Failed to acquire task slot for {task_id}")

        try:
            result = task_func(*args, **kwargs)
            return result
        finally:
            self.release_task_slot(task_id)

    def get_controller_stats(self) -> Dict[str, Any]:
        """
        获取控制器统计信息

        Get controller statistics

        Returns:
            dict: 统计信息
                 Statistics
        """
        with self.lock:
            current_stats = self.stats.copy()
            current_stats.update({
                'active_tasks_count': len(self.active_tasks),
                'queued_tasks_count': len(self.task_queue),
                'resource_conflicts': len(self.resource_dependencies),
                'active_task_details': {
                    task_id: {
                        'start_time': info['start_time'].isoformat(),
                        'wait_time': info['wait_time'],
                        'resources': list(info.get('required_resources', []))
                    }
                    for task_id, info in self.active_tasks.items()
                }
            })

        return current_stats

    def force_release_stuck_tasks(self, max_age_seconds: float = 3600.0):
        """
        强制释放卡住的任务

        Force release stuck tasks

        Args:
            max_age_seconds: 最大任务年龄（秒）
                           Maximum task age (seconds)
        """
        with self.condition:
            current_time = datetime.now()
            stuck_tasks = []

            for task_id, task_info in self.active_tasks.items():
                age = (current_time - task_info['start_time']).total_seconds()
                if age > max_age_seconds:
                    stuck_tasks.append(task_id)

            for task_id in stuck_tasks:
                self.logger.warning(f"Forcing release of stuck task: {task_id}")
                self.release_task_slot(task_id)

            if stuck_tasks:
                self.condition.notify_all()

            return len(stuck_tasks)

    def set_max_concurrent_tasks(self, max_tasks: int):
        """
        设置最大并发任务数

        Set maximum concurrent tasks

        Args:
            max_tasks: 最大并发任务数
                      Maximum concurrent tasks
        """
        with self.condition:
            old_max = self.max_concurrent_tasks
            self.max_concurrent_tasks = max_tasks
            self.logger.info(f"Max concurrent tasks changed: {old_max} -> {max_tasks}")

            # 如果增加并发数，通知等待的线程
            if max_tasks > old_max:
                self.condition.notify_all()
