#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征任务调度器

提供分布式特征计算的任务调度功能。
支持统一工作节点注册表，兼容特征工作节点和训练执行器。
"""

import logging
import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue

# 导入统一工作节点注册表（从分布式协调器层）
from src.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus
)


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
                logger.error(f"任务队列已满，无法提交新任务: {task_type}")
                raise ValueError("任务队列已满")

            self._tasks[task_id] = task
            self._task_queue.put((-priority.value, task_id))  # 负值用于优先级队列
            self._stats["total_tasks"] += 1
            self._stats["pending_tasks"] += 1

        # 从metadata中获取原始job_id
        original_job_id = metadata.get("job_id") if metadata else None
        logger.info(f"提交任务: {task_id}, 原始任务ID: {original_job_id}, 类型: {task_type}, 优先级: {priority.value}, 队列大小: {self._task_queue.qsize()}")
        return task_id

    def get_task(self, worker_id: str) -> Optional[FeatureTask]:
        """获取任务（供工作节点调用）"""
        with self._lock:
            if self._task_queue.empty():
                logger.debug(f"任务队列为空，工作节点 {worker_id} 无任务可获取")
                return None

            # 获取最高优先级的任务
            priority, task_id = self._task_queue.get()
            task = self._tasks.get(task_id)

            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                task.worker_id = worker_id

                self._stats["pending_tasks"] -= 1
                self._stats["running_tasks"] += 1

                logger.info(f"分配任务: {task_id} -> 工作节点: {worker_id}, 队列大小: {self._task_queue.qsize()}")
                return task
            else:
                # 如果任务不存在或状态不是PENDING，将任务放回队列
                if task_id in self._tasks:
                    task_status = task.status if task else "未知"
                    logger.warning(f"任务 {task_id} 状态不是PENDING ({task_status})，重新放回队列")
                    self._task_queue.put((priority, task_id))
                else:
                    logger.warning(f"任务 {task_id} 不存在，丢弃")

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

    def get_available_workers(self, worker_type: Optional[WorkerType] = None) -> List[str]:
        """
        获取可用工作节点
        
        Args:
            worker_type: 可选，指定工作节点类型（默认获取特征工作节点）
            
        Returns:
            可用工作节点ID列表
        """
        try:
            # 优先使用统一注册表获取可用工作节点
            registry = get_unified_worker_registry()
            
            # 如果没有指定类型，默认获取特征工作节点
            if worker_type is None:
                worker_type = WorkerType.FEATURE_WORKER
            
            available_workers = registry.get_available_workers(worker_type)
            
            logger.debug(f"从统一注册表获取可用工作节点 ({worker_type.value}): {len(available_workers)}")
            return available_workers
            
        except Exception as e:
            logger.debug(f"使用统一注册表获取可用工作节点失败: {e}")
            
            # 降级方案：使用工作节点管理器
            try:
                from .worker_manager import get_worker_manager
                worker_manager = get_worker_manager()
                workers = worker_manager.get_all_workers()
                current_time = datetime.now()
                available_workers = []
                
                for worker in workers:
                    # 检查心跳时间（超过30秒认为离线）
                    if (current_time - worker.last_heartbeat).seconds < 30:
                        available_workers.append(worker.worker_id)
                
                logger.debug(f"从工作节点管理器获取可用工作节点: {len(available_workers)}")
                return available_workers
            except Exception as e2:
                logger.debug(f"使用工作节点管理器获取可用工作节点失败: {e2}")
                
                # 最后降级方案：使用调度器内部的工作节点列表
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
        try:
            # 优先使用工作节点管理器获取统计信息
            from .worker_manager import get_worker_manager
            worker_manager = get_worker_manager()
            worker_stats = worker_manager.get_worker_stats()
            
            with self._lock:
                stats = self._stats.copy()
                stats.update({
                    "queue_size": self._task_queue.qsize(),
                    "active_workers": worker_stats.get("active_workers", len(self.get_available_workers())),
                    "total_workers": worker_stats.get("total_workers", len(self._workers)),
                    "pending_tasks": self._stats.get("pending_tasks", 0),
                    "running_tasks": self._stats.get("running_tasks", 0),
                    "completed_tasks": self._stats.get("completed_tasks", 0),
                    "failed_tasks": self._stats.get("failed_tasks", 0),
                    "total_tasks": self._stats.get("total_tasks", 0),
                    "is_running": self._running
                })
                logger.debug(f"调度器统计信息: active_workers={stats['active_workers']}, queue_size={stats['queue_size']}, pending_tasks={stats['pending_tasks']}")
                return stats
        except Exception as e:
            logger.debug(f"使用工作节点管理器获取统计信息失败: {e}")
            
            # 降级方案：使用调度器内部的统计信息
            with self._lock:
                stats = self._stats.copy()
                stats.update({
                    "queue_size": self._task_queue.qsize(),
                    "active_workers": len(self.get_available_workers()),
                    "total_workers": len(self._workers),
                    "pending_tasks": self._stats.get("pending_tasks", 0),
                    "running_tasks": self._stats.get("running_tasks", 0),
                    "completed_tasks": self._stats.get("completed_tasks", 0),
                    "failed_tasks": self._stats.get("failed_tasks", 0),
                    "total_tasks": self._stats.get("total_tasks", 0),
                    "is_running": self._running
                })
                return stats

    def get_health_status(self) -> Dict[str, Any]:
        """获取调度器健康状态"""
        try:
            with self._lock:
                health_status = {
                    "status": "healthy" if self._running else "unhealthy",
                    "queue_size": self._task_queue.qsize(),
                    "active_workers": len(self.get_available_workers()),
                    "total_workers": len(self._workers),
                    "pending_tasks": self._stats.get("pending_tasks", 0),
                    "running_tasks": self._stats.get("running_tasks", 0),
                    "is_running": self._running,
                    "last_updated": datetime.now().isoformat()
                }
                
                # 检查健康状态
                if not self._running:
                    health_status["status"] = "unhealthy"
                    health_status["reason"] = "调度器未运行"
                elif len(self.get_available_workers()) == 0:
                    health_status["status"] = "warning"
                    health_status["reason"] = "无可用工作节点"
                elif self._task_queue.qsize() > 50:
                    health_status["status"] = "warning"
                    health_status["reason"] = "队列积压过多任务"
                
                logger.debug(f"调度器健康状态: {health_status['status']}")
                return health_status
        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                "status": "unhealthy",
                "reason": str(e),
                "last_updated": datetime.now().isoformat()
            }

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
        """启动调度器，并加载持久化存储中的待处理任务"""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("任务调度器已启动")
        
        # 加载持久化存储中的待处理任务
        self._load_pending_tasks_from_persistence()

    def stop(self) -> None:
        """
        停止调度器
        """
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join()
        logger.info("任务调度器已停止")

    def _load_pending_tasks_from_persistence(self) -> None:
        """
        从持久化存储加载待处理任务
        """
        try:
            from src.gateway.web.feature_task_persistence import list_feature_tasks
            
            logger.info("正在从持久化存储加载待处理任务...")
            tasks = list_feature_tasks()
            
            # 筛选待处理任务
            pending_tasks = [t for t in tasks if t.get("status") in ["pending", "submitted"]]
            
            if not pending_tasks:
                logger.info("没有待处理任务")
                return
            
            logger.info(f"发现 {len(pending_tasks)} 个待处理任务，正在提交到调度器...")
            
            for task in pending_tasks:
                try:
                    task_id = task.get("task_id")
                    task_type = task.get("task_type", "technical")
                    config = task.get("config", {})
                    
                    # 提交任务到调度器
                    self.submit_task(
                        task_type=task_type,
                        data=config,
                        priority=TaskPriority.NORMAL,
                        metadata={"task_id": task_id, "original_task": task}
                    )
                    
                    logger.info(f"待处理任务已提交到调度器: {task_id}")
                    
                except Exception as e:
                    logger.error(f"提交待处理任务失败: {e}")
                    continue
            
            logger.info(f"成功提交 {len(pending_tasks)} 个待处理任务到调度器")
            
        except Exception as e:
            logger.error(f"加载待处理任务失败: {e}")

    def start_with_workers(self, worker_count: int = None) -> None:
        """
        启动调度器并自动创建工作节点

        Args:
            worker_count: 工作节点数量，如果为None则根据CPU核心数自动计算
        """
        # 启动调度器
        if not self._running:
            self.start()
            logger.info("调度器已启动")
        else:
            logger.info("调度器已经在运行")

        # 计算工作节点数量
        if worker_count is None:
            import os
            # 根据CPU核心数计算，但限制最大为8个，避免资源过度占用
            cpu_count = os.cpu_count() or 2
            worker_count = min(max(2, cpu_count), 8)
            logger.info(f"根据CPU核心数自动计算工作节点数量: {worker_count} (CPU核心: {cpu_count}, 限制最大8个)")

        # 启动工作节点
        logger.info(f"开始创建 {worker_count} 个工作节点")

        # 导入工作节点执行器
        try:
            from .worker_executor import start_worker
            from .worker_manager import get_worker_manager

            worker_manager = get_worker_manager()
            worker_manager.start_monitoring()
            logger.info("工作节点管理器监控已启动")

            # 跟踪创建的工作节点
            created_workers = 0
            failed_workers = 0
            
            # 创建工作节点
            for i in range(worker_count):
                try:
                    worker_id = f"worker_{i}_{int(time.time())}"
                    capabilities = {
                        "max_concurrent_tasks": 1,
                        "supported_task_types": ["技术指标", "统计特征", "情感特征", "自定义特征"],
                        "cpu_cores": 1,
                        "max_memory": 512  # MB
                    }

                    logger.info(f"准备创建工作节点 {i+1}/{worker_count}: {worker_id}")
                    
                    # 启动工作节点线程
                    worker_thread = threading.Thread(
                        target=start_worker,
                        args=(worker_id, self),
                        daemon=True
                    )
                    worker_thread.start()
                    logger.debug(f"工作节点 {worker_id} 线程已启动")

                    # 注册工作节点
                    success = worker_manager.register_worker(worker_id, capabilities)
                    if success:
                        created_workers += 1
                        logger.info(f"已启动工作节点 {created_workers}/{worker_count}: {worker_id}")
                    else:
                        failed_workers += 1
                        logger.warning(f"注册工作节点 {worker_id} 失败")
                    
                    # 短暂休眠，避免同时启动过多线程
                    time.sleep(0.1)
                    
                except Exception as e:
                    failed_workers += 1
                    logger.error(f"创建工作节点 {i+1} 失败: {e}")
                    # 继续创建其他工作节点
                    continue

            logger.info(f"工作节点创建完成，成功启动 {created_workers}/{worker_count} 个工作节点，失败 {failed_workers} 个")
            
            # 检查创建的工作节点数量
            if created_workers == 0:
                logger.warning("没有成功创建任何工作节点，任务可能无法执行")
            elif created_workers < worker_count:
                logger.warning(f"部分工作节点创建失败，只成功创建了 {created_workers} 个工作节点")
            else:
                logger.info("所有工作节点创建成功")

        except Exception as e:
            logger.error(f"启动工作节点失败: {e}")
            # 即使工作节点启动失败，调度器仍然运行

    def _scheduler_loop(self) -> None:
        """
        调度器主循环
        """
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
                    
                    # 检查是否是训练执行器（训练执行器由统一注册表管理，不在此报告）
                    if not worker_id.startswith("training_executor"):
                        logger.warning(f"工作节点离线: {worker_id}")
                    else:
                        # 训练执行器的离线状态由统一注册表管理，这里只静默更新状态
                        logger.debug(f"训练执行器状态更新为离线: {worker_id} (由统一注册表管理)")

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
