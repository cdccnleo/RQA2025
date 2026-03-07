import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征任务调度器

提供分布式特征计算的任务调度功能。
"""

import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue


logger = logging.getLogger(__name__)


class TaskStatus(Enum):

    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):

    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FeatureTask:

    """特征任务"""
    task_id: str
    task_type: str
    data: Any
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None


class FeatureTaskScheduler:

    """特征任务调度器"""

    def __init__(self, max_queue_size: int = 1000):
        """
        初始化任务调度器

        Args:
            max_queue_size: 最大队列大小
        """
        self.max_queue_size = max_queue_size
        self._task_queue = PriorityQueue(maxsize=max_queue_size)
        self._tasks: Dict[str, FeatureTask] = {}
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread = None

        # 统计信息
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "pending_tasks": 0,
            "running_tasks": 0
        }

    def submit_task(self,


                    task_type: str,
                    data: Any,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())

        task = FeatureTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )

        with self._lock:
            if len(self._tasks) >= self.max_queue_size:
                raise ValueError("任务队列已满")

            self._tasks[task_id] = task
            self._task_queue.put((-priority.value, task_id))  # 负值用于优先级队列
            self._stats["total_tasks"] += 1
            self._stats["pending_tasks"] += 1

        logger.info(f"提交任务: {task_id}, 类型: {task_type}, 优先级: {priority.value}")
        return task_id

    def get_task(self, worker_id: str) -> Optional[FeatureTask]:
        """获取任务（供工作节点调用）"""
        with self._lock:
            if self._task_queue.empty():
                return None

            # 获取最高优先级的任务
            _, task_id = self._task_queue.get()
            task = self._tasks.get(task_id)

            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                task.worker_id = worker_id

                self._stats["pending_tasks"] -= 1
                self._stats["running_tasks"] += 1

                logger.info(f"分配任务: {task_id} -> 工作节点: {worker_id}")
                return task

        return None

    def complete_task(self, task_id: str, result: Any, error: Optional[str] = None) -> None:
        """完成任务（供工作节点调用）"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return

            task.completed_at = datetime.now()
            task.result = result
            task.error = error

            if error:
                task.status = TaskStatus.FAILED
                self._stats["failed_tasks"] += 1
                logger.error(f"任务失败: {task_id}, 错误: {error}")
            else:
                task.status = TaskStatus.COMPLETED
                self._stats["completed_tasks"] += 1
                logger.info(f"任务完成: {task_id}")

            self._stats["running_tasks"] -= 1

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            if task.status == TaskStatus.PENDING:
                self._stats["pending_tasks"] -= 1
            elif task.status == TaskStatus.RUNNING:
                self._stats["running_tasks"] -= 1

            logger.info(f"任务取消: {task_id}")
            return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.status if task else None

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            return None

    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """注册工作节点"""
        with self._lock:
            self._workers[worker_id] = {
                "capabilities": capabilities,
                "registered_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "status": "active"
            }
            logger.info(f"注册工作节点: {worker_id}")

    def unregister_worker(self, worker_id: str) -> None:
        """注销工作节点"""
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info(f"注销工作节点: {worker_id}")

    def update_worker_heartbeat(self, worker_id: str) -> None:
        """更新工作节点心跳"""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id]["last_heartbeat"] = datetime.now()

    def get_available_workers(self) -> List[str]:
        """获取可用工作节点"""
        with self._lock:
            current_time = datetime.now()
            available_workers = []

            for worker_id, worker_info in self._workers.items():
                # 检查心跳时间（超过30秒认为离线）
                if (current_time - worker_info["last_heartbeat"]).seconds < 30:
                    available_workers.append(worker_id)

            return available_workers

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                "queue_size": self._task_queue.qsize(),
                "active_workers": len(self.get_available_workers()),
                "total_workers": len(self._workers)
            })
            return stats

    def get_task_history(self, limit: Optional[int] = None) -> List[FeatureTask]:
        """获取任务历史"""
        with self._lock:
            tasks = list(self._tasks.values())
            tasks.sort(key=lambda t: t.created_at, reverse=True)

            if limit:
                tasks = tasks[:limit]

            return tasks

    def clear_completed_tasks(self, older_than_hours: int = 24) -> int:
        """清理已完成的任务"""
        with self._lock:
            current_time = datetime.now()
            # 使用timedelta来正确计算时间差
            from datetime import timedelta
            cutoff_time = current_time - timedelta(hours=older_than_hours)

            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                        and task.completed_at and task.completed_at < cutoff_time):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._tasks[task_id]

            logger.info(f"清理了 {len(tasks_to_remove)} 个已完成的任务")
            return len(tasks_to_remove)

    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("任务调度器已启动")

    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join()
        logger.info("任务调度器已停止")

    def _scheduler_loop(self) -> None:
        """调度器主循环"""
        while self._running:
            try:
                # 检查工作节点状态
                self._check_worker_status()

                # 处理超时任务
                self._handle_timeout_tasks()

                time.sleep(1)  # 每秒检查一次

            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                time.sleep(5)  # 错误时等待更长时间

    def _check_worker_status(self) -> None:
        """检查工作节点状态"""
        current_time = datetime.now()

        with self._lock:
            for worker_id, worker_info in self._workers.items():
                # 检查心跳超时
                if (current_time - worker_info["last_heartbeat"]).seconds > 60:
                    worker_info["status"] = "offline"
                    logger.warning(f"工作节点离线: {worker_id}")

    def _handle_timeout_tasks(self) -> None:
        """处理超时任务"""
        current_time = datetime.now()

        with self._lock:
            for task in self._tasks.values():
                if (task.status == TaskStatus.RUNNING
                    and task.started_at
                        and (current_time - task.started_at).seconds > 300):  # 5分钟超时

                    task.status = TaskStatus.FAILED
                    task.error = "任务执行超时"
                    task.completed_at = current_time

                    self._stats["failed_tasks"] += 1
                    self._stats["running_tasks"] -= 1

                    logger.warning(f"任务超时: {task.task_id}")


# 全局任务调度器实例
_task_scheduler = FeatureTaskScheduler()


def get_task_scheduler() -> FeatureTaskScheduler:
    """获取全局任务调度器"""
    return _task_scheduler


def submit_task(task_type: str,


                data: Any,
                priority: TaskPriority = TaskPriority.NORMAL,
                metadata: Optional[Dict[str, Any]] = None) -> str:
    """提交任务的便捷函数"""
    return _task_scheduler.submit_task(task_type, data, priority, metadata)
