"""
Executor Manager Module
执行器管理器模块

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- executor_models.py: 执行器模型和指标
- managed_executor.py: 托管执行器
- executor_manager.py: 执行器管理器(本文件)

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import concurrent.futures

from .executor_models import ExecutorType, ExecutorStatus, ExecutorMetrics
from .managed_executor import ManagedExecutor

logger = logging.getLogger(__name__)


# 继续保留ExecutorManager类

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


class ManagedExecutor:

    """
    Managed Executor Class
    托管执行器类

    Wraps different types of executors with monitoring and management capabilities
    使用监控和管理能力包装不同类型的执行器
    """

    def __init__(self,


                 executor_type: ExecutorType,
                 max_workers: Optional[int] = None,
                 executor_id: Optional[str] = None):
        """
        Initialize managed executor
        初始化托管执行器

        Args:
            executor_type: Type of executor to create
                          要创建的执行器类型
            max_workers: Maximum number of workers (for pool executors)
                        最大工作线程数（对于池执行器）
            executor_id: Unique identifier for this executor
                        此执行器的唯一标识符
        """
        self.executor_type = executor_type
        self.executor_id = executor_id or f"{executor_type.value}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"
        self.max_workers = max_workers
        self.status = ExecutorStatus.INITIALIZING
        self.executor: Optional[concurrent.futures.Executor] = None
        self.metrics = ExecutorMetrics()
        self.futures: Dict[str, concurrent.futures.Future] = {}

        # Create the actual executor
        self._create_executor()

    def _create_executor(self):
        """Create the underlying executor"""
        try:
            if self.executor_type == ExecutorType.THREAD_POOL:
                workers = self.max_workers or min(32, os.cpu_count() * 2)
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                self.max_workers = workers
            elif self.executor_type == ExecutorType.PROCESS_POOL:
                workers = self.max_workers or min(8, os.cpu_count())
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
                self.max_workers = workers
            elif self.executor_type == ExecutorType.SINGLE_THREAD:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                self.max_workers = 1

            self.status = ExecutorStatus.RUNNING
            logger.info(
                f"Managed executor {self.executor_id} created with {self.max_workers} workers")

        except Exception as e:
            self.status = ExecutorStatus.ERROR
            logger.error(f"Failed to create executor {self.executor_id}: {str(e)}")

    def submit_task(self,


                    func: Callable,
                    *args,
                    task_id: Optional[str] = None,
                    **kwargs) -> Optional[str]:
        """
        Submit a task for execution
        提交任务以执行

        Args:
            func: Function to execute
                 要执行的函数
            *args: Positional arguments
                  位置参数
            task_id: Optional task identifier
                    可选的任务标识符
            **kwargs: Keyword arguments
                     关键字参数

        Returns:
            str: Task ID if submitted successfully, None otherwise
                 如果提交成功则返回任务ID，否则返回None
        """
        if not self.executor or self.status != ExecutorStatus.RUNNING:
            logger.warning(f"Executor {self.executor_id} is not available")
            return None

        try:
            task_id = task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Submit the task
            future = self.executor.submit(func, *args, **kwargs)
            self.futures[task_id] = future

            # Record metrics
            self.metrics.record_task_start()

            # Add callback to track completion
            future.add_done_callback(lambda f: self._task_completed_callback(task_id, f))

            logger.debug(f"Task {task_id} submitted to executor {self.executor_id}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to submit task to executor {self.executor_id}: {str(e)}")
            return None

    def _task_completed_callback(self, task_id: str, future: concurrent.futures.Future):
        """
        Callback for task completion
        任务完成的回调

        Args:
            task_id: Task identifier
                    任务标识符
            future: Future object
                   Future对象
        """
        try:
            # Calculate execution time (simplified)
            execution_time = 0.001  # Placeholder

            if future.exception():
                self.metrics.record_task_failure()
                logger.error(f"Task {task_id} failed: {future.exception()}")
            else:
                self.metrics.record_task_completion(execution_time)
                logger.debug(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Error in task completion callback for {task_id}: {str(e)}")

        finally:
            # Clean up
            if task_id in self.futures:
                del self.futures[task_id]

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        取消正在运行的任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        if task_id in self.futures:
            future = self.futures[task_id]
            cancelled = future.cancel()
            if cancelled:
                self.metrics.record_task_failure()
                logger.info(f"Task {task_id} cancelled")
            return cancelled

        return False

    def get_task_result(self, task_id: str, timeout: float = 1.0) -> Any:
        """
        Get the result of a completed task
        获取已完成任务的结果

        Args:
            task_id: Task identifier
                    任务标识符
            timeout: Timeout for result retrieval
                    结果检索超时时间

        Returns:
            Task result
            任务结果

        Raises:
            TimeoutError: If result is not available within timeout
                         如果在超时时间内结果不可用
            Exception: If task failed
                      如果任务失败
        """
        if task_id not in self.futures:
            raise ValueError(f"Task {task_id} not found")

        future = self.futures[task_id]
        return future.result(timeout=timeout)

    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor
        关闭执行器

        Args:
            wait: Whether to wait for running tasks to complete
                 是否等待正在运行的任务完成
        """
        if self.executor and self.status == ExecutorStatus.RUNNING:
            self.status = ExecutorStatus.STOPPING
            self.executor.shutdown(wait=wait)
            self.status = ExecutorStatus.STOPPED
            logger.info(f"Executor {self.executor_id} shutdown")

    def get_status(self) -> Dict[str, Any]:
        """
        Get executor status and metrics
        获取执行器状态和指标

        Returns:
            dict: Executor status information
                  执行器状态信息
        """
        return {
            'executor_id': self.executor_id,
            'executor_type': self.executor_type.value,
            'status': self.status.value,
            'max_workers': self.max_workers,
            'active_tasks': len(self.futures),
            'metrics': self.metrics.to_dict()
        }


class ExecutorManager:

    """
    Executor Manager Class
    执行器管理器类

    Manages multiple executors and provides load balancing and optimization
    管理多个执行器并提供负载均衡和优化
    """

    def __init__(self):

        self.executors: Dict[str, ManagedExecutor] = {}
        self.executor_counter = 0
        self.is_running = False

        # Auto - scaling settings
        self.enable_auto_scaling = True
        self.min_executors = 1
        self.max_executors = 10
        self.scale_up_threshold = 0.8  # Scale up when 80% capacity reached
        self.scale_down_threshold = 0.2  # Scale down when 20% capacity reached

        logger.info("Executor manager initialized")

    def create_executor(self,
                        executor_type_or_id,
                        executor_type: Optional[ExecutorType] = None,
                        max_workers: Optional[int] = None) -> Optional[str]:
        """
        Create a new managed executor
        创建新的托管执行器

        Args:
            executor_type_or_id: Type of executor to create or custom executor ID
                               要创建的执行器类型或自定义执行器ID
            executor_type: Type of executor to create (if first param is ID)
                         要创建的执行器类型（如果第一个参数是ID）
            max_workers: Maximum number of workers
                        最大工作线程数

        Returns:
            bool: True if created successfully, False otherwise
                  创建成功返回True，否则返回False
        """
        try:
            # Handle different input formats
            if isinstance(executor_type_or_id, dict):
                # Config dict format
                config = executor_type_or_id
                actual_executor_type = config.get('executor_type', 'thread_pool')
                if actual_executor_type == 'thread_pool':
                    actual_executor_type = ExecutorType.THREAD_POOL
                elif actual_executor_type == 'process_pool':
                    actual_executor_type = ExecutorType.PROCESS_POOL
                max_workers = config.get('max_workers', max_workers)
                thread_name_prefix = config.get('thread_name_prefix')
                self.executor_counter += 1
                executor_id = f"executor_{self.executor_counter}"
            elif executor_type is None:
                # First param is executor_type
                actual_executor_type = executor_type_or_id
                self.executor_counter += 1
                executor_id = f"executor_{self.executor_counter}"
            else:
                # First param is executor_id, second is executor_type
                executor_id = executor_type_or_id
                actual_executor_type = executor_type

            if executor_id in self.executors:
                logger.info(f"Executor {executor_id} already exists, recreating")
                # Close existing executor if running
                try:
                    self.executors[executor_id].shutdown()
                except:
                    pass
                # Create new executor

            executor = ManagedExecutor(actual_executor_type, max_workers, executor_id)
            self.executors[executor_id] = executor

            logger.info(f"Created executor {executor_id} of type {actual_executor_type}")
            return executor_id
        except Exception as e:
            logger.error(f"Failed to create executor: {str(e)}")
            return None

    def get_executor(self, executor_id: str) -> Optional[ManagedExecutor]:
        """
        Get an executor by ID
        通过ID获取执行器

        Args:
            executor_id: Executor identifier
                        执行器标识符

        Returns:
            ManagedExecutor: Executor instance or None if not found
                            执行器实例，如果未找到则返回None
        """
        return self.executors.get(executor_id)

    def remove_executor(self, executor_id: str) -> bool:
        """
        Remove an executor by ID
        通过ID移除执行器

        Args:
            executor_id: Executor identifier
                        执行器标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if executor_id in self.executors:
            executor = self.executors[executor_id]
            # Shutdown the executor
            if hasattr(executor, 'shutdown'):
                executor.shutdown(wait=True)
            del self.executors[executor_id]
            logger.info(f"Removed executor {executor_id}")
            return True
        return False

    def scale_executor(self, executor_id: str, new_max_workers: int) -> bool:
        """
        Scale an executor by changing its max workers
        通过更改最大工作线程数来扩展执行器

        Args:
            executor_id: Executor identifier
                        执行器标识符
            new_max_workers: New maximum number of workers
                           新的最大工作线程数

        Returns:
            bool: True if scaled successfully, False otherwise
                  扩展成功返回True，否则返回False
        """
        if executor_id not in self.executors:
            return False

        managed_executor = self.executors[executor_id]
        executor_type = managed_executor.executor_type

        # Shutdown existing executor
        if hasattr(managed_executor.executor, 'shutdown'):
            managed_executor.executor.shutdown(wait=True)

        # Create new executor with new max_workers
        if executor_type == ExecutorType.THREAD_POOL:
            from concurrent.futures import ThreadPoolExecutor
            new_executor = ThreadPoolExecutor(max_workers=new_max_workers)
        elif executor_type == ExecutorType.PROCESS_POOL:
            from concurrent.futures import ProcessPoolExecutor
            new_executor = ProcessPoolExecutor(max_workers=new_max_workers)
        else:
            return False

        # Update managed executor
        managed_executor.executor = new_executor
        managed_executor.max_workers = new_max_workers

        logger.info(f"Scaled executor {executor_id} to {new_max_workers} workers")
        return True

    def get_manager_status(self) -> Dict[str, Any]:
        """
        Get the status of the executor manager
        获取执行器管理器的状态

        Returns:
            dict: Status information including executor count, active tasks, etc.
                  状态信息，包括执行器数量、活跃任务等
        """
        # Calculate active tasks across all executors
        total_active_tasks = sum(
            getattr(e, '_threads', 0) if hasattr(e, '_threads') else 0
            for e in self.executors.values()
        )

        return {
            'total_executors': len(self.executors),
            'active_executors': len([e for e in self.executors.values() if hasattr(e, 'is_running') and e.is_running]),
            'total_active_tasks': total_active_tasks,
            'executor_types': list(set(e.executor_type.value for e in self.executors.values())),
            'total_max_workers': sum(e.max_workers for e in self.executors.values() if hasattr(e, 'max_workers')),
            'resource_usage': {
                'cpu_percent': 50.0,  # Placeholder
                'memory_percent': 60.0  # Placeholder
            }
        }

    def get_manager_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the executor manager
        获取执行器管理器的指标

        Returns:
            dict: Metrics information including performance stats
                  指标信息，包括性能统计
        """
        # Calculate metrics (placeholders for actual implementation)
        total_tasks_submitted = sum(
            getattr(e, '_tasks_submitted', 0) if hasattr(e, '_tasks_submitted') else 0
            for e in self.executors.values()
        )
        total_tasks_completed = sum(
            getattr(e, '_tasks_completed', 0) if hasattr(e, '_tasks_completed') else 0
            for e in self.executors.values()
        )

        return {
            'executor_count': len(self.executors),
            'active_executor_count': len([e for e in self.executors.values() if hasattr(e, 'is_running') and e.is_running]),
            'total_workers': sum(e.max_workers for e in self.executors.values() if hasattr(e, 'max_workers')),
            'executor_utilization': 0.0,  # Placeholder for utilization calculation
            'total_tasks_submitted': total_tasks_submitted,
            'total_tasks_completed': total_tasks_completed,
            'average_execution_time': 0.1,  # Placeholder
            'resource_efficiency': 0.8  # Placeholder
        }

    def submit_task(self, executor_id: str, func: Callable, *args, **kwargs) -> Optional[concurrent.futures.Future]:
        """
        Submit a task to a specific executor
        将任务提交到指定的执行器

        Args:
            executor_id: Target executor ID
                        目标执行器ID
            func: Function to execute
                 要执行的函数
            *args: Positional arguments
                  位置参数
            **kwargs: Keyword arguments
                     关键字参数

        Returns:
            Future: Future object if submitted successfully, None otherwise
                   如果提交成功则返回Future对象，否则返回None
        """
        executor = self.get_executor(executor_id)
        if executor:
            try:
                # Submit directly to the executor and return the future
                return executor.executor.submit(func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to submit task to executor {executor_id}: {str(e)}")
                return None
        return None

    def submit_task_to_best_executor(self,


                                     func: Callable,
                                     *args,
                                     **kwargs) -> Optional[tuple]:
        """
        Submit task to the best available executor
        将任务提交到最佳可用的执行器

        Args:
            func: Function to execute
                 要执行的函数
            *args: Positional arguments
                  位置参数
            **kwargs: Keyword arguments
                     关键字参数

        Returns:
            tuple: (executor_id, task_id) if submitted successfully, None otherwise
                  如果提交成功则返回(执行器ID, 任务ID)，否则返回None
        """
        # Find the executor with lowest load
        best_executor = None
        min_load = float('inf')

        for executor in self.executors.values():
            if executor.status == ExecutorStatus.RUNNING:
                load = executor.metrics.current_active_threads / executor.max_workers
                if load < min_load:
                    min_load = load
                    best_executor = executor

        if best_executor:
            task_id = best_executor.submit_task(func, *args, **kwargs)
            if task_id:
                return (best_executor.executor_id, task_id)

        return None

    def get_system_load(self) -> Dict[str, Any]:
        """
        Get overall system load
        获取整体系统负载

        Returns:
            dict: System load information
                  系统负载信息
        """
        total_executors = len(self.executors)
        running_executors = sum(1 for e in self.executors.values()
                                if e.status == ExecutorStatus.RUNNING)
        total_active_tasks = sum(e.metrics.current_active_threads
                                 for e in self.executors.values())
        total_capacity = sum(e.max_workers for e in self.executors.values())

        return {
            'total_executors': total_executors,
            'running_executors': running_executors,
            'total_active_tasks': total_active_tasks,
            'total_capacity': total_capacity,
            'system_load': total_active_tasks / max(total_capacity, 1),
            'executors': {eid: e.get_status() for eid, e in self.executors.items()}
        }

    def auto_scale_executors(self):
        """
        Automatically scale executors based on load
        根据负载自动扩展执行器

        Returns:
            bool: True if scaling occurred, False otherwise
                  如果发生了扩展则返回True，否则返回False
        """
        if not self.enable_auto_scaling:
            return False

        system_load = self.get_system_load()
        load_ratio = system_load['system_load']

        scaled = False

        # Scale up
        if load_ratio > self.scale_up_threshold and len(self.executors) < self.max_executors:
            # Create a new thread pool executor
            self.create_executor(ExecutorType.THREAD_POOL)
            scaled = True
            logger.info("Auto - scaled up: created new executor due to high load")

        # Scale down
        elif load_ratio < self.scale_down_threshold and len(self.executors) > self.min_executors:
            # Find an idle executor to remove
            idle_executor_id = None
            for eid, executor in self.executors.items():
                if (executor.status == ExecutorStatus.RUNNING
                        and executor.metrics.current_active_threads == 0):
                    idle_executor_id = eid
                    break

            if idle_executor_id:
                executor = self.executors[idle_executor_id]
                executor.shutdown()
                del self.executors[idle_executor_id]
                scaled = True
                logger.info(f"Auto - scaled down: removed idle executor {idle_executor_id}")

        return scaled

    def shutdown_all(self, wait: bool = True):
        """
        Shutdown all executors
        关闭所有执行器

        Args:
            wait: Whether to wait for running tasks
                 是否等待正在运行的任务
        """
        logger.info("Shutting down all executors")

        for executor in self.executors.values():
            executor.shutdown(wait=wait)

        self.executors.clear()
        logger.info("All executors shutdown")

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics
        获取管理器统计信息

        Returns:
            dict: Manager statistics
                  管理器统计信息
        """
        return {
            'total_executors': len(self.executors),
            'system_load': self.get_system_load(),
            'enable_auto_scaling': self.enable_auto_scaling,
            'min_executors': self.min_executors,
            'max_executors': self.max_executors,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold
        }

    def get_executor_status(self, executor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific executor
        获取特定执行器的状态

        Args:
            executor_id: Executor ID
                        执行器ID

        Returns:
            dict: Executor status information or None if not found
                 执行器状态信息，如果未找到则返回None
        """
        executor = self.executors.get(executor_id)
        if executor:
            return {
                'executor_id': executor_id,
                'status': executor.status.value,
                'type': executor.executor_type.value,
                'max_workers': executor.max_workers,
                'active_tasks': executor.metrics.current_active_threads,
                'active': executor.metrics.current_active_threads > 0,
                'is_alive': executor.executor is not None and not executor.executor._shutdown
            }
        return None

    def get_executor_metrics(self, executor_id: str) -> Optional[ExecutorMetrics]:
        """
        Get the metrics of a specific executor
        获取特定执行器的指标

        Args:
            executor_id: Executor ID
                        执行器ID

        Returns:
            ExecutorMetrics: Executor metrics object or None if not found
                           执行器指标对象，如果未找到则返回None
        """
        executor = self.executors.get(executor_id)
        if executor:
            return executor.metrics
        return None


    def shutdown_executor(self, executor_id: str) -> bool:
        """
        Shutdown a specific executor
        关闭特定执行器

        Args:
            executor_id: Executor ID to shutdown
                         要关闭的执行器ID

        Returns:
            bool: True if shutdown successfully, False otherwise
                 关闭成功返回True，否则返回False
        """
        try:
            executor = self.executors.get(executor_id)
            if executor:
                executor.shutdown()
                del self.executors[executor_id]
                logger.info(f"Executor {executor_id} shutdown successfully")
                return True
            else:
                logger.warning(f"Executor {executor_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error shutting down executor {executor_id}: {e}")
            return False

    def submit_task(self, task_config_or_func, *args, **kwargs) -> Optional[str]:
        """
        Submit a task to an available executor
        向可用执行器提交任务

        Args:
            task_config_or_func: Task config dict or function to execute
                                任务配置字典或要执行的函数
            *args: Positional arguments for the function
                   函数的位置参数
            **kwargs: Keyword arguments for the function
                     函数的关键字参数

        Returns:
            str: Task ID if submitted successfully, None otherwise
                 如果提交成功则返回任务ID，否则返回None
        """
        try:
            # Handle both task_config dict and direct function call
            if isinstance(task_config_or_func, dict):
                # Task config format
                func = task_config_or_func.get('function')
                task_args = task_config_or_func.get('args', [])
                task_kwargs = task_config_or_func.get('kwargs', {})
                executor_id = task_config_or_func.get('executor_id')
            else:
                # Direct function call
                func = task_config_or_func
                task_args = args
                task_kwargs = kwargs
                executor_id = None

            if not func:
                logger.error("No function provided for task submission")
                return None

            # Find an available executor
            available_executor = None
            if executor_id:
                # Use specific executor
                available_executor = self.executors.get(executor_id)
            else:
                # Find any available executor
                for executor in self.executors.values():
                    if executor.status == ExecutorStatus.RUNNING:
                        available_executor = executor
                        break

            if available_executor:
                # Submit task to executor
                future = available_executor.executor.submit(func, *task_args, **task_kwargs)
                task_id = f"task_{len(available_executor.active_tasks)}"
                available_executor.active_tasks[task_id] = future
                logger.info(f"Task {task_id} submitted to executor {available_executor.executor_id}")
                return task_id
            else:
                logger.warning("No available executor found")
                return None
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return None


# Global executor manager instance
# 全局执行器管理器实例
executor_manager = ExecutorManager()

__all__ = [
    'ExecutorType',
    'ExecutorStatus',
    'ExecutorMetrics',
    'ManagedExecutor',
    'ExecutorManager',
    'executor_manager'
]
