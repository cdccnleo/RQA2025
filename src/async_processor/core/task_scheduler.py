"""
Task Scheduler Module
任务调度器模块

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- task_models.py: 任务数据模型
- task_scheduler.py: 任务调度器(本文件)

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import threading
import time
import queue

from .task_models import TaskPriority, TaskStatus, ScheduledTask

logger = logging.getLogger(__name__)


# 继续保留TaskScheduler类
class TaskScheduler:

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
                 max_retries: int = 3,
                 name: Optional[str] = None):
        """
        Initialize a scheduled task
        初始化调度任务

        Args:
            task_id: Unique task identifier
                   唯一任务标识符
            func: Function to execute
                 要执行的函数
            args: Positional arguments for the function
                 函数的位置参数
            kwargs: Keyword arguments for the function
                   函数的关键字参数
            priority: Task priority level
                     任务优先级
            scheduled_time: Time to execute the task (None for immediate)
                          执行任务的时间（None表示立即执行）
            timeout: Maximum execution time in seconds
                    最大执行时间（秒）
            retry_count: Current retry count
                        当前重试次数
            max_retries: Maximum number of retries
                        最大重试次数
        """
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.scheduled_time = scheduled_time or datetime.now()
        self.timeout = timeout
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.name = name or f"task_{task_id}"
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[Exception] = None

    def __lt__(self, other):
        """Comparison for priority queue"""
        if self.scheduled_time != other.scheduled_time:
            return self.scheduled_time < other.scheduled_time
        return self.priority.value > other.priority.value  # Higher priority first

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary
        将任务转换为字典

        Returns:
            dict: Task data as dictionary
                  任务数据字典
        """
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'priority': self.priority.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'has_error': self.error is not None,
            'error_message': str(self.error) if self.error else None
        }


class TaskScheduler:

    """
    Task Scheduler for Asynchronous Operations
    异步操作任务调度器

    Manages and executes scheduled tasks with priority and timing control
    管理和执行具有优先级和时间控制的调度任务
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_workers: int = 4, queue_size: int = 1000):
        """
        Initialize the task scheduler
        初始化任务调度器

        Args:
            config: Configuration dictionary
                   配置字典
            max_workers: Maximum number of worker threads
                        最大工作线程数
            queue_size: Maximum size of the task queue
                       任务队列的最大大小
        """
        # Handle config parameter (for backward compatibility)
        if config is not None:
            self.config = config
            self.max_workers = config.get('max_workers', max_workers)
            self.queue_size = config.get('queue_size', queue_size)
        else:
            self.config = {
                'max_workers': max_workers,
                'queue_size': queue_size,
                'task_timeout': 30.0,
                'enable_priority_queue': True,
                'enable_scheduling': True
            }
            self.max_workers = max_workers
            self.queue_size = queue_size

        self.task_queue = queue.PriorityQueue(maxsize=self.queue_size)
        self.active_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}  # For test compatibility
        self.running_tasks: Dict[str, ScheduledTask] = {}    # For test compatibility
        self.workers: List[threading.Thread] = []
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.task_counter = 0

        # Synchronization
        self.lock = threading.Lock()

        logger.info(f"Task scheduler initialized with {self.max_workers} workers")

    def start(self) -> bool:
        """
        Start the task scheduler
        启动任务调度器

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Task scheduler is already running")
            return False

        try:
            self.is_running = True

            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i + 1}")
                worker.daemon = True
                worker.start()
                self.workers.append(worker)

            # Start scheduler thread
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop, name="TaskScheduler")
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()

            logger.info(f"Task scheduler started with {self.max_workers} workers")
            return True

        except Exception as e:
            logger.error(f"Failed to start task scheduler: {str(e)}")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """
        Stop the task scheduler
        停止任务调度器

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Task scheduler is not running")
            return False

        try:
            self.is_running = False

            # Wait for workers to finish
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            # Wait for scheduler to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)

            logger.info("Task scheduler stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop task scheduler: {str(e)}")
            return False

    def schedule_task(self,
                      task_or_func,
                      args: Optional[tuple] = None,
                      kwargs: Optional[Dict[str, Any]] = None,
                      priority: TaskPriority = TaskPriority.NORMAL,
                      delay_seconds: float = 0,
                      timeout: Optional[float] = None,
                      max_retries: int = 3) -> Union[bool, Optional[str]]:
        """
        Schedule a task for execution
        调度任务以执行

        Args:
            task_or_func: ScheduledTask object or function to execute
                         ScheduledTask对象或要执行的函数
            args: Positional arguments (if task_or_func is function)
                 位置参数（如果task_or_func是函数）
            kwargs: Keyword arguments (if task_or_func is function)
                   关键字参数（如果task_or_func是函数）
            priority: Task priority (if task_or_func is function)
                     任务优先级（如果task_or_func是函数）
            delay_seconds: Delay before execution (seconds)
                          执行前的延迟（秒）
            timeout: Task timeout (seconds)
                    任务超时时间（秒）
            max_retries: Maximum retry attempts
                        最大重试次数

        Returns:
            bool or str: True if ScheduledTask scheduled successfully,
                        Task ID if function scheduled successfully,
                        False/None if scheduling failed
                        如果ScheduledTask调度成功返回True，
                        如果函数调度成功返回任务ID，
                        调度失败返回False/None
        """
        # Check if task_or_func is a ScheduledTask object
        if isinstance(task_or_func, ScheduledTask):
            return self.schedule_task_obj(task_or_func)

        # Validate input
        if task_or_func is None:
            return False

        # Handle function scheduling
        try:
            with self.lock:
                self.task_counter += 1
                task_id = f"task_{self.task_counter}"
                scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)

                task = ScheduledTask(
                    task_id=task_id,
                    func=task_or_func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority,
                    scheduled_time=scheduled_time,
                    timeout=timeout,
                    max_retries=max_retries
                )

                # Store in appropriate collection based on scheduling time
                if delay_seconds > 0:
                    # Delayed task - store in scheduled_tasks
                    self.scheduled_tasks[task_id] = task
                    self.task_queue.put((task.priority.value, task))
                else:
                    # Immediate task - store in active_tasks and queue
                    self.active_tasks[task_id] = task
                    self.task_queue.put((task.priority.value, task))

                logger.info(f"Task {task_id} scheduled for {scheduled_time}")
                return task_id

        except queue.Full:
            logger.warning("Task queue is full, cannot schedule new task")
            return None
        except Exception as e:
            logger.error(f"Failed to schedule task: {str(e)}")
            return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task
        取消已调度的任务

        Args:
            task_id: ID of the task to cancel
                     要取消的任务ID

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Task {task_id} cancelled")
                    return True

            # Check scheduled tasks
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Task {task_id} cancelled")
                    return True

            return False

    def unschedule_task(self, task_id: str) -> bool:
        """
        Remove a task from the scheduled tasks
        从调度任务中移除任务

        Args:
            task_id: ID of the task to unschedule
                     要取消调度的任务ID

        Returns:
            bool: True if unscheduled successfully, False otherwise
                  取消调度成功返回True，否则返回False
        """
        with self.lock:
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
                # Remove from task_queue if present (this is more complex with heapq)
                # For simplicity, we'll just remove from scheduled_tasks
                logger.info(f"Task {task_id} unscheduled")
                return True
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task
        获取任务的状态

        Args:
            task_id: Task ID
                     任务ID

        Returns:
            dict: Task status information or None if not found
                 任务状态信息，如果未找到则返回None
        """
        with self.lock:
            task = None
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
            elif task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
            elif task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]

            if task:
                return {
                    'task_id': task_id,
                    'status': task.status.value if hasattr(task.status, 'value') else str(task.status),
                    'created_at': task.created_at.isoformat() if hasattr(task, 'created_at') else None,
                    'started_at': getattr(task, 'started_at', None),
                    'completed_at': getattr(task, 'completed_at', None),
                    'priority': getattr(task, 'priority', None),
                    'progress': getattr(task, 'progress', 0.0)
                }
            else:
                return None

    def submit_task_for_execution(self, task) -> bool:
        """
        Submit a task for immediate execution
        提交任务以立即执行

        Args:
            task: Task to submit
                 要提交的任务

        Returns:
            bool: True if submitted successfully, False otherwise
                  提交成功返回True，否则返回False
        """
        try:
            with self.lock:
                if task.task_id in self.active_tasks:
                    logger.warning(f"Task {task.task_id} is already active")
                    return False

                # Add to active tasks
                self.active_tasks[task.task_id] = task

                # Add to priority queue for execution
                self.task_queue.put((task.priority.value, task))

                logger.info(f"Task {task.task_id} submitted for execution")
                return True

        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {str(e)}")
            return False

    def submit_bulk_tasks(self, tasks: list) -> int:
        """
        批量提交任务

        Args:
            tasks: 任务列表

        Returns:
            int: 成功提交的任务数量
        """
        success_count = 0
        with self.lock:
            for task in tasks:
                try:
                    if task.task_id in self.scheduled_tasks or task.task_id in self.active_tasks:
                        logger.warning(f"Task {task.task_id} is already scheduled/active, skipping")
                        continue

                    # Add to scheduled tasks (for bulk submission, we schedule rather than execute immediately)
                    self.scheduled_tasks[task.task_id] = task

                    success_count += 1
                    logger.info(f"Task {task.task_id} submitted for scheduling")

                except Exception as e:
                    logger.error(f"Failed to submit task {task.task_id}: {str(e)}")

        logger.info(f"Bulk submission completed: {success_count}/{len(tasks)} tasks submitted")
        return success_count

    def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task
        重试失败的任务

        Args:
            task_id: ID of the task to retry
                     要重试的任务ID

        Returns:
            bool: True if retry was successful, False otherwise
                  重试成功返回True，否则返回False
        """
        try:
            with self.lock:
                # Check if task exists in any state
                task = None
                if task_id in self.scheduled_tasks:
                    task = self.scheduled_tasks[task_id]
                elif task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                else:
                    # Check completed tasks
                    for completed_task in self.completed_tasks:
                        if hasattr(completed_task, 'task_id') and completed_task.task_id == task_id:
                            task = completed_task
                            break

                if task is None:
                    logger.warning(f"Task {task_id} not found for retry")
                    return False

                # Check if task can be retried
                if task.retry_count >= task.max_retries:
                    logger.warning(f"Task {task_id} has exceeded maximum retries ({task.max_retries})")
                    return False

                # Increment retry count
                task.retry_count += 1

                # Reset task status for retry
                task.status = TaskStatus.PENDING

                # Reschedule the task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

                # Add back to scheduled tasks
                self.scheduled_tasks[task_id] = task

                # Reset scheduled time to now for immediate retry
                task.scheduled_time = datetime.now()

                logger.info(f"Task {task_id} scheduled for retry (attempt {task.retry_count}/{task.max_retries})")
                return True

        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {str(e)}")
            return False

    def restart(self) -> bool:
        """
        Restart the task scheduler
        重启任务调度器

        Returns:
            bool: True if restart was successful, False otherwise
                  重启成功返回True，否则返回False
        """
        try:
            with self.lock:
                # Clear all tasks
                self.scheduled_tasks.clear()
                self.active_tasks.clear()
                self.task_counter = 0

                # Reset running state and restart
                self.is_running = True

                logger.info("Task scheduler restarted successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to restart task scheduler: {str(e)}")
            return False

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get scheduler status information
        获取调度器状态信息

        Returns:
            dict: Status information including task counts, running state, etc.
                  状态信息，包括任务数量、运行状态等
        """
        with self.lock:
            # Count tasks by status
            queued_tasks = sum(1 for task in self.scheduled_tasks.values() if task.status == TaskStatus.PENDING)
            running_tasks = sum(1 for task in self.active_tasks.values() if task.status == TaskStatus.RUNNING)
            completed_tasks = len(self.completed_tasks)

            return {
                'is_running': self.is_running,
                'queued_tasks': queued_tasks,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'total_scheduled_tasks': len(self.scheduled_tasks),
                'total_active_tasks': len(self.active_tasks),
                'total_completed_tasks': len(self.completed_tasks),
                'task_counter': self.task_counter
            }

    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """
        Get scheduler performance metrics
        获取调度器性能指标

        Returns:
            dict: Performance metrics including execution times, failure rates, etc.
                  性能指标，包括执行时间、失败率等
        """
        with self.lock:
            total_tasks = len(self.scheduled_tasks) + len(self.active_tasks) + len(self.completed_tasks)
            completed_tasks = len(self.completed_tasks)
            failed_tasks = sum(1 for task in self.completed_tasks.values() if hasattr(task, 'status') and task.status == TaskStatus.FAILED)

            # Calculate average execution time
            execution_times = []
            for task in self.completed_tasks.values():
                if hasattr(task, 'execution_time') and task.execution_time is not None:
                    execution_times.append(task.execution_time)

            average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'average_execution_time': average_execution_time
            }

    def schedule_task_obj(self, task: ScheduledTask) -> bool:
        """
        Schedule a task object for execution
        调度任务对象以执行

        Args:
            task: ScheduledTask object to schedule
                 要调度的ScheduledTask对象

        Returns:
            bool: True if scheduled successfully, False otherwise
                  调度成功返回True，否则返回False
        """
        try:
            with self.lock:
                if task.task_id in self.scheduled_tasks:
                    logger.info(f"Task {task.task_id} already scheduled, updating")
                    # Update existing task
                    self.scheduled_tasks[task.task_id] = task
                    # Re-queue the task (remove and re-add to priority queue)
                    # Note: This is simplified - in practice, we'd need to remove from heap first
                    self.task_queue.put((task.priority.value, task))
                    return True

                # Add to scheduled tasks for test compatibility
                self.scheduled_tasks[task.task_id] = task

                # Add to priority queue
                self.task_queue.put((task.priority.value, task))

                logger.info(f"Task {task.task_id} scheduled successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to schedule task {task.task_id}: {str(e)}")
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics
        获取队列统计信息

        Returns:
            dict: Queue statistics including queued, running, completed tasks
                  队列统计信息，包括排队中、运行中、已完成的任务
        """
        with self.lock:
            return {
                'queued': self.task_queue.qsize(),
                'running': len(self.running_tasks),
                'scheduled': len(self.scheduled_tasks),
                'completed': len(self.completed_tasks),
                'total_active': len(self.active_tasks),
                'is_running': self.is_running,
                'max_workers': self.max_workers,
                'queue_size': self.queue_size
            }

    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop
        主要的调度器循环
        """
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                # Check for tasks that are ready to execute
                now = datetime.now()

                # This is a simplified implementation
                # In a real implementation, you'd use a more sophisticated
                # timing mechanism, possibly with a separate priority queue
                # for scheduled tasks

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                time.sleep(1)

        logger.info("Scheduler loop stopped")

    def _worker_loop(self) -> None:
        """
        Worker thread loop
        工作线程循环
        """
        thread_name = threading.current_thread().name
        logger.info(f"Worker {thread_name} started")

        while self.is_running:
            try:
                # Get task from queue (returns (priority, task))
                _, task = self.task_queue.get(timeout=1)

                if task.status == TaskStatus.CANCELLED:
                    self.task_queue.task_done()
                    continue

                # Execute task
                self._execute_task(task)
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {thread_name} error: {str(e)}")

        logger.info(f"Worker {thread_name} stopped")

    def _execute_task(self, task: ScheduledTask) -> None:
        """
        Execute a scheduled task
        执行调度的任务

        Args:
            task: Task to execute
                  要执行的任务
        """
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

            logger.info(f"Executing task {task.task_id}")

            # Execute the task function
            if task.timeout:
                # Execute with timeout
                result = self._execute_with_timeout(task.func, task.args, task.kwargs, task.timeout)
            else:
                result = task.func(*task.args, **task.kwargs)

            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            task.error = e
            task.completed_at = datetime.now()

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                # Put back in queue for retry
                try:
                    self.task_queue.put((task.priority.value, task))
                    logger.info(
                        f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                    return
                except queue.Full:
                    pass

            # Max retries reached or queue full
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.task_id} failed permanently: {str(e)}")

        finally:
            # Move to completed tasks
            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: Dict[str, Any], timeout: float) -> Any:
        """
        Execute a function with timeout
        使用超时执行函数

        Args:
            func: Function to execute
                 要执行的函数
            args: Positional arguments
                 位置参数
            kwargs: Keyword arguments
                   关键字参数
            timeout: Timeout in seconds
                    超时时间（秒）

        Returns:
            Function result
            函数结果

        Raises:
            TimeoutError: If execution exceeds timeout
                         如果执行超过超时时间
        """
        import signal

        def timeout_handler(signum, frame):

            raise TimeoutError(f"Function execution exceeded {timeout} seconds")

        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics
        获取调度器统计信息

        Returns:
            dict: Scheduler statistics
                  调度器统计信息
        """
        with self.lock:
            return {
                'is_running': self.is_running,
                'max_workers': self.max_workers,
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'total_tasks_scheduled': self.task_counter
            }


# Global task scheduler instance
# 全局任务调度器实例
task_scheduler = TaskScheduler()

__all__ = [
    'TaskPriority',
    'TaskStatus',
    'ScheduledTask',
    'TaskScheduler',
    'task_scheduler'
]
