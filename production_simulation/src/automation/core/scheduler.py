"""
Scheduler Module
调度器模块

This module provides scheduling capabilities for automation tasks
此模块为自动化任务提供调度能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
from croniter import croniter

logger = logging.getLogger(__name__)


class ScheduleType(Enum):

    """Schedule type enumeration"""
    INTERVAL = "interval"      # Fixed interval execution
    CRON = "cron"             # Cron expression scheduling
    ONCE = "once"             # Execute once at specific time
    DAILY = "daily"           # Daily execution
    WEEKLY = "weekly"         # Weekly execution
    MONTHLY = "monthly"       # Monthly execution


class TaskStatus(Enum):

    """Scheduled task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ScheduledTask:

    """
    Scheduled Task Class
    调度任务类

    Represents a task scheduled for execution
    表示计划执行的任务
    """

    def __init__(self,


                 task_id: str,
                 name: str,
                 function: Callable,
                 schedule_type: ScheduleType,
                 schedule_config: Dict[str, Any],
                 args: Optional[List[Any]] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 max_retries: int = 3,
                 timeout: Optional[float] = None,
                 enabled: bool = True):
        """
        Initialize scheduled task
        初始化调度任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Human - readable task name
                 人类可读的任务名称
            function: Function to execute
                     要执行的函数
            schedule_type: Type of schedule
                          调度类型
            schedule_config: Schedule configuration
                           调度配置
            args: Positional arguments for function
                 函数的位置参数
            kwargs: Keyword arguments for function
                   函数的关键字参数
            max_retries: Maximum number of retries on failure
                        失败时的最大重试次数
            timeout: Task execution timeout (seconds)
                    任务执行超时时间（秒）
            enabled: Whether the task is enabled
                    任务是否启用
        """
        self.task_id = task_id
        self.name = name
        self.function = function
        self.schedule_type = schedule_type
        self.schedule_config = schedule_config
        self.args = args or []
        self.kwargs = kwargs or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self.enabled = enabled

        # Runtime state
        self.status = TaskStatus.PENDING
        self.next_run: Optional[datetime] = None
        self.last_run: Optional[datetime] = None
        self.created_at = datetime.now()

        # Execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.average_execution_time = 0.0
        self.last_execution_time = 0.0

        # Retry information
        self.current_retry = 0
        self.last_error: Optional[Exception] = None

        # Calculate next run time
        self._calculate_next_run()

    def _calculate_next_run(self) -> None:
        """
        Calculate the next run time based on schedule configuration
        根据调度配置计算下次运行时间
        """
        now = datetime.now()

        try:
            if self.schedule_type == ScheduleType.INTERVAL:
                # Interval scheduling
                interval_seconds = self.schedule_config.get('interval_seconds', 3600)
                if self.last_run:
                    self.next_run = self.last_run + timedelta(seconds=interval_seconds)
                else:
                    self.next_run = now + timedelta(seconds=interval_seconds)

            elif self.schedule_type == ScheduleType.CRON:
                # Cron expression scheduling
                cron_expression = self.schedule_config.get('cron_expression', '0 * * * *')
                cron = croniter(cron_expression, now)
                self.next_run = cron.get_next(datetime)

            elif self.schedule_type == ScheduleType.ONCE:
                # One - time execution
                run_time_str = self.schedule_config.get('run_time')
                if run_time_str:
                    self.next_run = datetime.fromisoformat(run_time_str)
                else:
                    self.next_run = None

            elif self.schedule_type == ScheduleType.DAILY:
                # Daily execution
                run_time = self.schedule_config.get('run_time', '09:00:00')
                hour, minute, second = map(int, run_time.split(':'))
                next_run = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                self.next_run = next_run

            elif self.schedule_type == ScheduleType.WEEKLY:
                # Weekly execution
                run_time = self.schedule_config.get('run_time', '09:00:00')
                weekday = self.schedule_config.get('weekday', 0)  # 0=Monday, 6=Sunday
                hour, minute, second = map(int, run_time.split(':'))

                next_run = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
                days_ahead = (weekday - now.weekday()) % 7
                if days_ahead == 0 and next_run <= now:
                    days_ahead = 7
                next_run += timedelta(days=days_ahead)
                self.next_run = next_run

            elif self.schedule_type == ScheduleType.MONTHLY:
                # Monthly execution
                run_time = self.schedule_config.get('run_time', '09:00:00')
                day = self.schedule_config.get('day', 1)
                hour, minute, second = map(int, run_time.split(':'))

                next_run = now.replace(day=day, hour=hour, minute=minute,
                                       second=second, microsecond=0)
                if next_run <= now:
                    # Move to next month
                    if now.month == 12:
                        next_run = next_run.replace(year=now.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=now.month + 1)
                self.next_run = next_run

        except Exception as e:
            logger.error(f"Failed to calculate next run time for task {self.task_id}: {str(e)}")
            self.next_run = None

    def should_run(self) -> bool:
        """
        Check if the task should run now
        检查任务现在是否应该运行

        Returns:
            bool: True if task should run, False otherwise
                  如果任务应该运行则返回True，否则返回False
        """
        if not self.enabled or self.next_run is None:
            return False

        return datetime.now() >= self.next_run

    def execute(self) -> Dict[str, Any]:
        """
        Execute the scheduled task
        执行调度任务

        Returns:
            dict: Execution result
                  执行结果
        """
        self.status = TaskStatus.RUNNING
        self.execution_count += 1
        start_time = time.time()

        result = {
            'task_id': self.task_id,
            'task_name': self.name,
            'executed_at': datetime.now(),
            'success': False,
            'execution_time': 0.0,
            'retry_attempt': self.current_retry
        }

        try:
            # Execute the function
            if self.timeout:
                # Execute with timeout
                import signal

                def timeout_handler(signum, frame):

                    raise TimeoutError(f"Task execution exceeded {self.timeout} seconds")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

                try:
                    function_result = self.function(*self.args, **self.kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                function_result = self.function(*self.args, **self.kwargs)

            # Success
            execution_time = time.time() - start_time
            result.update({
                'success': True,
                'result': function_result,
                'execution_time': execution_time
            })

            self.success_count += 1
            self.current_retry = 0
            self.last_error = None

            # Update average execution time
            total_executions = self.success_count + self.failure_count
            self.average_execution_time = (
                (self.average_execution_time * (total_executions - 1)) + execution_time
            ) / total_executions

            self.last_execution_time = execution_time

        except Exception as e:
            execution_time = time.time() - start_time
            result.update({
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            })

            self.failure_count += 1
            self.last_error = e

            # Handle retries
            if self.current_retry < self.max_retries:
                self.current_retry += 1
                result['will_retry'] = True
                result['retry_attempt'] = self.current_retry
                logger.warning(
                    f"Task {self.task_id} failed, will retry (attempt {self.current_retry}/{self.max_retries})")
            else:
                result['max_retries_exceeded'] = True
                logger.error(
                    f"Task {self.task_id} failed permanently after {self.max_retries} retries")

        # Update status and schedule next run
        if result['success']:
            self.status = TaskStatus.COMPLETED
        elif self.current_retry >= self.max_retries:
            self.status = TaskStatus.FAILED
        else:
            self.status = TaskStatus.PENDING  # Will retry

        self.last_run = result['executed_at']

        # Calculate next run time
        if result['success'] or self.current_retry >= self.max_retries:
            self._calculate_next_run()

        return result

    def cancel(self) -> bool:
        """
        Cancel the scheduled task
        取消调度任务

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        if self.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            self.status = TaskStatus.CANCELLED
            self.enabled = False
            logger.info(f"Task {self.task_id} cancelled")
            return True
        return False

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task information
        获取任务信息

        Returns:
            dict: Task information
                  任务信息
        """
        return {
            'task_id': self.task_id,
            'name': self.name,
            'schedule_type': self.schedule_type.value,
            'schedule_config': self.schedule_config,
            'status': self.status.value,
            'enabled': self.enabled,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / max(self.execution_count, 1) * 100,
            'average_execution_time': self.average_execution_time,
            'current_retry': self.current_retry,
            'max_retries': self.max_retries,
            'last_error': str(self.last_error) if self.last_error else None
        }


class TaskScheduler:

    """
    Task Scheduler Class
    任务调度器类

    Manages and executes scheduled tasks
    管理和执行调度任务
    """

    def __init__(self, scheduler_name: str = "default_task_scheduler"):
        """
        Initialize task scheduler
        初始化任务调度器

        Args:
            scheduler_name: Name of the scheduler
                          调度器的名称
        """
        self.scheduler_name = scheduler_name
        self.tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Scheduler settings
        self.check_interval = 30.0  # seconds
        self.max_concurrent_tasks = 10
        self.active_tasks = 0

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'executed_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0
        }

        logger.info(f"Task scheduler {scheduler_name} initialized")

    def add_task(self, task: ScheduledTask) -> None:
        """
        Add a task to the scheduler
        将任务添加到调度器中

        Args:
            task: Task to add
                 要添加的任务
        """
        self.tasks[task.task_id] = task
        self.stats['total_tasks'] = len(self.tasks)
        logger.info(f"Added task: {task.name} ({task.task_id})")

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the scheduler
        从调度器中移除任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                task.cancel()
            del self.tasks[task_id]
            self.stats['total_tasks'] = len(self.tasks)
            logger.info(f"Removed task: {task_id}")
            return True
        return False

    def start_scheduler(self) -> bool:
        """
        Start the task scheduler
        启动任务调度器

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return False

        try:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            logger.info("Task scheduler started")
            return True
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            self.is_running = False
            return False

    def stop_scheduler(self) -> bool:
        """
        Stop the task scheduler
        停止任务调度器

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return False

        try:
            self.is_running = False
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
            logger.info("Task scheduler stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {str(e)}")
            return False

    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop
        主要的调度器循环
        """
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                # Find tasks that should run
                tasks_to_run = []
                for task in self.tasks.values():
                    if (task.enabled
                        and task.should_run()
                            and self.active_tasks < self.max_concurrent_tasks):
                        tasks_to_run.append(task)

                # Execute tasks
                for task in tasks_to_run:
                    self._execute_task_async(task)

                # Sleep before next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                time.sleep(self.check_interval)

        logger.info("Scheduler loop stopped")

    def _execute_task_async(self, task: ScheduledTask) -> None:
        """
        Execute a task asynchronously
        异步执行任务

        Args:
            task: Task to execute
                 要执行的任务
        """

        def task_wrapper():

            self.active_tasks += 1
            try:
                result = task.execute()

                # Update statistics
                self.stats['executed_tasks'] += 1
                if result['success']:
                    self.stats['successful_tasks'] += 1
                else:
                    self.stats['failed_tasks'] += 1

                # Log result
                if result['success']:
                    logger.info(
                        f"Task {task.task_id} executed successfully in {result['execution_time']:.2f}s")
                else:
                    logger.error(
                        f"Task {task.task_id} execution failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Task execution error for {task.task_id}: {str(e)}")
                self.stats['failed_tasks'] += 1
            finally:
                self.active_tasks -= 1

        # Start task in a separate thread
        execution_thread = threading.Thread(target=task_wrapper, daemon=True)
        execution_thread.start()

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get scheduler status
        获取调度器状态

        Returns:
            dict: Scheduler status information
                  调度器状态信息
        """
        pending_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PENDING)
        running_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)
        completed_tasks = sum(1 for task in self.tasks.values()
                              if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)

        return {
            'scheduler_name': self.scheduler_name,
            'is_running': self.is_running,
            'total_tasks': len(self.tasks),
            'pending_tasks': pending_tasks,
            'running_tasks': running_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'active_executions': self.active_tasks,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'stats': self.stats
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task
        获取特定任务的状态

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            dict: Task status or None if not found
                  任务状态，如果未找到则返回None
        """
        if task_id in self.tasks:
            return self.tasks[task_id].get_task_info()
        return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific task
        取消特定任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        if task_id in self.tasks:
            return self.tasks[task_id].cancel()
        return False

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all tasks with optional status filter
        列出所有任务，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of task information
                  任务信息列表
        """
        tasks = []
        for task in self.tasks.values():
            task_info = task.get_task_info()
            if status_filter is None or task_info['status'] == status_filter:
                tasks.append(task_info)
        return tasks

    def create_interval_task(self,


                             task_id: str,
                             name: str,
                             function: Callable,
                             interval_seconds: int,
                             **kwargs) -> str:
        """
        Create an interval - based scheduled task
        创建基于间隔的调度任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Task name
                 任务名称
            function: Function to execute
                     要执行的函数
            interval_seconds: Interval between executions
                             执行间隔（秒）
            **kwargs: Additional task parameters
                     其他任务参数

        Returns:
            str: Created task ID
                 创建的任务ID
        """
        schedule_config = {'interval_seconds': interval_seconds}
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config=schedule_config,
            **kwargs
        )

        self.add_task(task)
        return task_id

    def create_cron_task(self,


                         task_id: str,
                         name: str,
                         function: Callable,
                         cron_expression: str,
                         **kwargs) -> str:
        """
        Create a cron - based scheduled task
        创建基于cron的调度任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Task name
                 任务名称
            function: Function to execute
                     要执行的函数
            cron_expression: Cron expression for scheduling
                           用于调度的cron表达式
            **kwargs: Additional task parameters
                     其他任务参数

        Returns:
            str: Created task ID
                 创建的任务ID
        """
        schedule_config = {'cron_expression': cron_expression}
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=ScheduleType.CRON,
            schedule_config=schedule_config,
            **kwargs
        )

        self.add_task(task)
        return task_id

    def create_daily_task(self,


                          task_id: str,
                          name: str,
                          function: Callable,
                          run_time: str = "09:00:00",
                          **kwargs) -> str:
        """
        Create a daily scheduled task
        创建每日调度任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Task name
                 任务名称
            function: Function to execute
                     要执行的函数
            run_time: Time to run daily (HH:MM:SS)
                     每日运行时间（HH:MM:SS）
            **kwargs: Additional task parameters
                     其他任务参数

        Returns:
            str: Created task ID
                 创建的任务ID
        """
        schedule_config = {'run_time': run_time}
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=ScheduleType.DAILY,
            schedule_config=schedule_config,
            **kwargs
        )

        self.add_task(task)
        return task_id


# Global task scheduler instance
# 全局任务调度器实例
scheduler = TaskScheduler()

__all__ = [
    'ScheduleType',
    'TaskStatus',
    'ScheduledTask',
    'TaskScheduler',
    'scheduler'
]
