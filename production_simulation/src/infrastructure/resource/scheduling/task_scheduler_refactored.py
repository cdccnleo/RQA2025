"""
task_scheduler_refactored 模块

提供 task_scheduler_refactored 相关功能和接口。
"""

import logging

# 导入配置类
import queue
import threading
import time
import uuid

from ..config.config_classes import TaskConfig
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
"""
任务调度器重构版
将TaskScheduler大类拆分为多个职责单一的专用类

职责分离:
- TaskManager: 任务生命周期管理
- TaskQueueManager: 队列管理
- TaskWorkerManager: 工作线程管理
- TaskSchedulerCore: 调度核心逻辑
- TaskMonitor: 任务监控和统计
- TaskSchedulerFacade: 门面类 (向后兼容)
"""

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    BACKGROUND = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    CRITICAL = 5


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =============================================================================
# 异常类定义
# =============================================================================


class TaskNotFoundException(Exception):
    """任务未找到异常"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task with id '{task_id}' not found")


class TaskNotCompletedError(Exception):
    """任务未完成错误"""

    def __init__(self, task_id: str, status: TaskStatus):
        self.task_id = task_id
        self.status = status
        super().__init__(f"Task '{task_id}' is not completed, current status: {status.value}")


@dataclass
class Task:
    """任务数据类"""
    id: str
    name: str
    func: Callable
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout: int = 3600
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

    def __lt__(self, other):
        """用于优先级队列排序 (优先级数字越小，优先级越高)"""
        return self.priority.value < other.priority.value


class TaskManager:
    """
    任务管理器
    职责: 任务生命周期管理
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def add_task(self, task: Task) -> str:
        """添加任务"""
        with self._lock:
            self.tasks[task.id] = task
            return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        with self._lock:
            return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus,
                           result: Any = None, error: Any = None):
        """更新任务状态"""
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = status
                if status == TaskStatus.RUNNING:
                    task.started_at = time.time()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.completed_at = time.time()
                    if result is not None:
                        task.result = result
                    if error is not None:
                        task.error = error

    def remove_task(self, task_id: str):
        """移除任务"""
        with self._lock:
            self.tasks.pop(task_id, None)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        with self._lock:
            return list(self.tasks.values())


class TaskQueueManager:
    """
    任务队列管理器
    职责: 队列操作管理
    """

    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self._lock = threading.Lock()

    def put_task(self, task: Task):
        """将任务放入队列"""
        with self._lock:
            self.queue.put(task)

    def get_next_task(self) -> Optional[Task]:
        """获取下一个待执行任务"""
        with self._lock:
            try:
                return self.queue.get()
            except Exception as e:
                return None

    def get_pending_count(self) -> int:
        """获取待处理任务数量"""
        with self._lock:
            return self.queue.size()


class TaskWorkerManager:
    """
    任务工作线程管理器
    职责: 工作线程生命周期管理
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.workers: List[threading.Thread] = []
        self.running = False
        self._lock = threading.Lock()

    def start_workers(self, worker_loop_func: Callable):
        """启动工作线程"""
        with self._lock:
            if self.running:
                return

            self.running = True
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=worker_loop_func,
                    name=f"TaskWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)

    def stop_workers(self):
        """停止工作线程"""
        with self._lock:
            if not self.running:
                return

            self.running = False
            for worker in self.workers:
                worker.join(timeout=5.0)

            self.workers.clear()

    def is_running(self) -> bool:
        """检查是否正在运行"""
        with self._lock:
            return self.running


class TaskMonitor:
    """
    任务监控器
    职责: 任务监控和统计
    """

    def __init__(self):
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'avg_execution_time': 0.0,
            'uptime': 0.0
        }
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_task_start(self):
        """记录任务开始"""
        with self._lock:
            self.stats['total_tasks'] += 1

    def record_task_completion(self, execution_time: float):
        """记录任务完成"""
        with self._lock:
            self.stats['completed_tasks'] += 1
            # 更新平均执行时间
            total_completed = self.stats['completed_tasks']
            current_avg = self.stats['avg_execution_time']
            self.stats['avg_execution_time'] = (
                (current_avg * (total_completed - 1)) + execution_time
            ) / total_completed

    def record_task_failure(self):
        """记录任务失败"""
        with self._lock:
            self.stats['failed_tasks'] += 1

    def record_task_cancellation(self):
        """记录任务取消"""
        with self._lock:
            self.stats['cancelled_tasks'] += 1

    def update_uptime(self):
        """更新运行时间"""
        with self._lock:
            self.stats['uptime'] = time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self.stats.copy()


class TaskSchedulerCore:
    """
    任务调度核心
    职责: 调度核心逻辑
    """

    def __init__(self,
                 task_manager: TaskManager,
                 queue_manager: TaskQueueManager,
                 worker_manager: TaskWorkerManager,
                 monitor: TaskMonitor):
        self.task_manager = task_manager
        self.queue_manager = queue_manager
        self.worker_manager = worker_manager
        self.monitor = monitor

        # 调度相关配置
        self.task_timeout = 3600
        self.enable_timeout_monitoring = True

    def submit_task(self, name: str, func: Callable, priority: TaskPriority,
                    timeout: int = 3600, *args, **kwargs) -> str:
        """提交任务到调度器"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            func=func,
            priority=priority,
            timeout=timeout,
            args=args,
            kwargs=kwargs
        )

        self.task_manager.add_task(task)
        self.queue_manager.put_task(task)
        self.monitor.record_task_start()

        logger.info(f"任务已提交: {task_id} ({name})")
        return task_id

    def submit_task_with_config(self, config: TaskConfig) -> str:
        """使用配置对象提交任务"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=config.task_type,
            func=lambda: None,  # 占位符，实际执行逻辑在config中
            priority=config.priority,
            timeout=config.timeout
        )

        self.task_manager.add_task(task)
        self.queue_manager.put_task(task)
        self.monitor.record_task_start()

        logger.info(f"任务已提交: {task_id} (类型: {config.task_type})")
        return task_id

    def execute_task(self, task: Task):
        """执行任务"""
        try:
            self.task_manager.update_task_status(task.id, TaskStatus.RUNNING)
            start_time = time.time()

            # 执行任务
            result = task.func(*task.args, **task.kwargs)

            execution_time = time.time() - start_time
            self.task_manager.update_task_status(task.id, TaskStatus.COMPLETED, result=result)
            self.monitor.record_task_completion(execution_time)

            logger.info(f"任务执行成功: {task.id} ({task.name}) in {execution_time:.2f}s")

        except Exception as e:
            self.task_manager.update_task_status(task.id, TaskStatus.FAILED, error=str(e))
            self.monitor.record_task_failure()
            logger.error(f"任务执行失败: {task.id} ({task.name}): {e}")

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.task_manager.get_task(task_id)
        if task and task.status == TaskStatus.PENDING:
            self.task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            self.monitor.record_task_cancellation()
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self.task_manager.get_task(task_id)
        return task.status if task else None

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        task = self.task_manager.get_task(task_id)
        return task.result if task else None


class TaskSchedulerFacade:
    """
    任务调度器门面类
    职责: 统一接口，向后兼容
    """

    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        # 初始化各个专用组件
        self.task_manager = TaskManager()
        self.queue_manager = TaskQueueManager(max_size=queue_size)
        self.worker_manager = TaskWorkerManager(max_workers=max_workers)
        self.monitor = TaskMonitor()

        # 初始化调度核心
        self.scheduler_core = TaskSchedulerCore(
            self.task_manager,
            self.queue_manager,
            self.worker_manager,
            self.monitor
        )

        # 兼容性属性
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.tasks = self.task_manager.tasks
        self.workers = self.worker_manager.workers
        self.running = False

        # 统计兼容性
        self.stats = self.monitor.stats

    def start(self):
        """启动调度器"""
        self.worker_manager.start_workers(self._worker_loop)
        self.running = True
        logger.info(f"任务调度器已启动 (workers: {self.max_workers})")

    def stop(self):
        """停止调度器"""
        self.worker_manager.stop_workers()
        self.running = False
        logger.info("任务调度器已停止")

    def _worker_loop(self):
        """工作线程循环"""
        while self.worker_manager.is_running():
            task = self.queue_manager.get_next_task()
            if task:
                self.scheduler_core.execute_task(task)
            else:
                time.sleep(0.1)  # 避免忙等待

    def submit_task(self, name, func=None, priority=None, *args, **kwargs):
        """提交任务 - 兼容旧接口"""
        if isinstance(name, Task):
            # 如果第一个参数是Task对象，直接使用
            task = name
            return self.scheduler_core.submit_task(
                task.name, task.func, task.priority, task.timeout
            )
        else:
            # 旧接口：name, func, priority
            if priority is None:
                priority = TaskPriority.NORMAL
            return self.scheduler_core.submit_task(name, func, priority, *args, **kwargs)

    def submit_task_with_config(self, config: TaskConfig) -> str:
        """使用配置对象提交任务 - 新接口"""
        return self.scheduler_core.submit_task_with_config(config)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.scheduler_core.cancel_task(task_id)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        return self.scheduler_core.get_task_status(task_id)

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self.scheduler_core.get_task_result(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self.monitor.update_uptime()
        return self.monitor.get_stats()

    # 其他兼容性方法
    def submit_task_object(self, name, func=None, priority=None, *args, **kwargs):
        """兼容性方法"""
        return self.submit_task(name, func, priority, *args, **kwargs)

    def get_next_task(self):
        """兼容性方法"""
        return self.queue_manager.get_next_task()

    def pending_tasks(self):
        """兼容性方法"""
        return self.queue_manager.get_pending_count()

    def shutdown(self):
        """兼容性方法"""
        self.stop()

    def set_task_executor(self, executor):
        """兼容性方法"""
        # 暂时不支持


# 向后兼容的别名
TaskScheduler = TaskSchedulerFacade
