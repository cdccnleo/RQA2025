#!/usr/bin/env python3
"""
异步任务管理器

实现非阻塞风险评估和处理，支持任务队列、优先级调度、并发控制和状态监控
"""

import asyncio
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from queue import PriorityQueue, Queue
import uuid
import json

logger = logging.getLogger(__name__)


class TaskPriority(Enum):

    """任务优先级"""
    CRITICAL = 0    # 关键任务（立即执行）
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级
    BACKGROUND = 4  # 后台任务


class TaskStatus(Enum):

    """任务状态"""
    PENDING = "pending"         # 等待中
    RUNNING = "running"         # 运行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消
    TIMEOUT = "timeout"         # 超时


class TaskType(Enum):

    """任务类型"""
    RISK_CALCULATION = "risk_calculation"         # 风险计算
    PORTFOLIO_ANALYSIS = "portfolio_analysis"     # 组合分析
    MARKET_IMPACT = "market_impact"               # 市场冲击分析
    COMPLIANCE_CHECK = "compliance_check"         # 合规检查
    PREDICTION = "prediction"                     # 预测任务
    BATCH_PROCESSING = "batch_processing"         # 批量处理
    REPORT_GENERATION = "report_generation"       # 报告生成


@dataclass(order=True)
class Task:

    """异步任务"""
    priority: int
    task_id: str = field(compare=False)
    task_type: TaskType = field(compare=False)
    name: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: dict = field(compare=False, default_factory=dict)
    callback: Optional[Callable] = field(compare=False, default=None)
    error_callback: Optional[Callable] = field(compare=False, default=None)
    timeout: Optional[float] = field(compare=False, default=None)
    max_retries: int = field(compare=False, default=3)
    retry_delay: float = field(compare=False, default=1.0)

    # 任务状态
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    created_at: datetime = field(compare=False, default_factory=datetime.now)
    started_at: Optional[datetime] = field(compare=False, default=None)
    completed_at: Optional[datetime] = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: Optional[Exception] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)

    # 任务元数据
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    def __post_init__(self):
        # 确保优先级是整数
        if isinstance(self.priority, TaskPriority):
            self.priority = self.priority.value


@dataclass
class TaskResult:

    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    execution_time: Optional[float] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:

    """异步任务管理器"""

    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):

        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        # 任务队列（优先级队列）
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)

        # 任务存储
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RiskTask")

        # 异步事件循环
        self.event_loop = None
        self.loop_thread = None

        # 控制标志
        self.running = False
        self.shutdown_event = threading.Event()

        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'avg_execution_time': 0.0,
            'queue_size': 0,
            'active_workers': 0
        }

        logger.info(f"异步任务管理器初始化完成，最大工作线程数: {self.max_workers}")

    def start(self):
        """启动任务管理器"""
        if self.running:
            logger.warning("任务管理器已在运行中")
            return

        self.running = True
        self.shutdown_event.clear()

        # 启动事件循环线程
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        # 启动任务处理线程
        for i in range(self.max_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker_thread.start()

        logger.info(f"任务管理器已启动，{self.max_workers}个工作线程")

    def stop(self, timeout: float = 5.0):
        """停止任务管理器"""
        if not self.running:
            return

        logger.info("正在停止任务管理器...")
        self.running = False
        self.shutdown_event.set()

        # 取消所有运行中的任务
        for task_id, future in self.running_tasks.items():
            if not future.done():
                future.cancel()
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED

        # 关闭线程池
        self.executor.shutdown(wait=True, timeout=timeout)

        # 等待事件循环线程结束
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=timeout)

        # 等待所有工作线程结束
        # 注意：工作线程是守护线程，在主程序退出时会自动结束
        # 但在测试环境中需要手动等待

        logger.info("任务管理器已停止")

    def submit_task(self, task_type: TaskType, name: str, func: Callable,
                    args: tuple = (), kwargs: dict = {},
                    priority: TaskPriority = TaskPriority.NORMAL,
                    callback: Optional[Callable] = None,
                    error_callback: Optional[Callable] = None,
                    timeout: Optional[float] = None,
                    max_retries: int = 3) -> str:
        """
        提交异步任务

        Args:
            task_type: 任务类型
            name: 任务名称
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 任务优先级
            callback: 成功回调函数
            error_callback: 错误回调函数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数

        Returns:
            任务ID
        """
        if not self.running:
            raise RuntimeError("任务管理器未启动")

        if self.task_queue.full():
            raise RuntimeError("任务队列已满")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务
        task = Task(
            priority=priority,
            task_id=task_id,
            task_type=task_type,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback,
            timeout=timeout,
            max_retries=max_retries
        )

        # 存储任务
        self.tasks[task_id] = task

        # 添加到队列
        self.task_queue.put(task)

        # 更新统计
        self.stats['total_tasks'] += 1
        self.stats['queue_size'] = self.task_queue.qsize()

        logger.info(f"任务已提交 {task_id} ({name})")
        return task_id

    async def submit_task_async(self, task_type: TaskType, name: str, func: Callable,
                                args: tuple = (), kwargs: dict = {},
                                priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        异步提交任务

        Args:
            task_type: 任务类型
            name: 任务名称
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 任务优先级

        Returns:
            任务ID
        """
        return self.submit_task(task_type, name, func, args, kwargs, priority)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        return task.status if task else None

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self.completed_tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.stats['cancelled_tasks'] += 1
            logger.info(f"任务已取消 {task_id}")
            return True
        elif task.status == TaskStatus.RUNNING:
            # 取消运行中的任务
            future = self.running_tasks.get(task_id)
            if future and not future.done():
                future.cancel()
            task.status = TaskStatus.CANCELLED
            self.stats['cancelled_tasks'] += 1
            logger.info(f"运行中任务已取消: {task_id}")
            return True

        return False


    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        等待任务完成

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            任务结果
        """
        start_time = time.time()

        while True:
            result = self.get_task_result(task_id)
            if result:
                return result

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.1)  # 短暂等待

    async def wait_for_task_async(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        异步等待任务完成

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            任务结果
        """
        if not self.event_loop:
            return self.wait_for_task(task_id, timeout)

        return await asyncio.wait_for(
            self._wait_for_task_coro(task_id),
            timeout=timeout
        )

    async def _wait_for_task_coro(self, task_id: str):
        """等待任务完成的协程"""
        while True:
            result = self.get_task_result(task_id)
            if result:
                return result
            await asyncio.sleep(0.1)


    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'queue_size': self.task_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_workers': len(self.running_tasks),
            'max_workers': self.max_workers,
            'running': self.running
        }


    def get_task_stats(self) -> Dict[str, Any]:
        """获取任务统计"""
        return self.stats.copy()

    def get_running_tasks(self) -> List[str]:
        """获取运行中的任务ID列表"""
        return list(self.running_tasks.keys())

    def get_pending_tasks(self) -> List[str]:
        """获取等待中的任务ID列表"""
        pending = []
        # 注意：PriorityQueue不支持直接遍历，这里返回已知任务中的等待任务
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                pending.append(task_id)
        return pending

    def _run_event_loop(self):
        """运行异步事件循环"""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)

            logger.info("异步事件循环已启动")

            # 运行事件循环直到关闭
            self.event_loop.run_until_complete(self._event_loop_main())

        except Exception as e:
            logger.error(f"事件循环异常: {e}")
        finally:
            if self.event_loop:
                self.event_loop.close()

    async def _event_loop_main(self):
        """事件循环主函数"""
        while self.running:
            try:
                # 处理异步任务
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    def _worker_loop(self):
        """工作线程循环"""
        logger.info(f"工作线程 {threading.current_thread().name} 已启动")

        while self.running and not self.shutdown_event.is_set():
            try:
                # 从队列获取任务，使用更短的超时
                try:
                    task = self.task_queue.get(timeout=0.1)
                except:
                    # 队列为空，短暂等待后继续
                    continue

                if not self.running:
                    break

                # 执行任务
                self._execute_task(task)

            except Exception as e:
                logger.error(f"工作线程异常: {e}")

        logger.info(f"工作线程 {threading.current_thread().name} 已停止")

    def _execute_task(self, task: Task):
        """执行任务"""
        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = None  # 标记为运行中

            logger.info(f"开始执行任务 {task.task_id} ({task.name})")

            # 在线程池中执行任务
            future = self.executor.submit(self._run_task_function, task)
            self.running_tasks[task.task_id] = future

            # 等待任务完成或超时
            try:
                if task.timeout:
                    result = future.result(timeout=task.timeout)
                else:
                    result = future.result()
            except TimeoutError:
                logger.warning(f"任务超时: {task.task_id}")
                task.status = TaskStatus.TIMEOUT
                task.error = TimeoutError(f"任务执行超时 ({task.timeout}s)")
                self.stats['failed_tasks'] += 1
                return
            except Exception as e:
                logger.error(f"任务执行失败: {task.task_id}, {e}")
                self._handle_task_error(task, e)
                return

            # 任务成功完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            execution_time = (task.completed_at - task.started_at).total_seconds()

            # 创建任务结果
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                created_at=task.created_at,
                completed_at=task.completed_at,
                metadata=task.metadata
            )

            # 存储结果
            self.completed_tasks[task.task_id] = task_result

            # 更新统计
            self.stats['completed_tasks'] += 1
            self._update_avg_execution_time(execution_time)

            logger.info(f"任务完成: {task.task_id}, 执行时间: {execution_time:.2f}s")

            # 执行成功回调
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        # 异步回调
                        if self.event_loop:
                            asyncio.run_coroutine_threadsafe(
                                task.callback(task_result),
                                self.event_loop
                            )
                    else:
                        # 同步回调
                        task.callback(task_result)
                except Exception as e:
                    logger.error(f"任务回调执行失败: {e}")

        except Exception as e:
            logger.error(f"任务执行异常: {e}")
            self._handle_task_error(task, e)

        finally:
            # 清理运行中任务记录
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

    def _run_task_function(self, task: Task) -> Any:
        """运行任务函数"""
        try:
            # 处理重试逻辑
            while task.retry_count < task.max_retries:
                try:
                    result = task.func(*task.args, **task.kwargs)
                    return result
                except Exception as e:
                    task.retry_count += 1
                    if task.retry_count < task.max_retries:
                        logger.warning(
                            f"任务重试 {task.retry_count}/{task.max_retries}: {task.task_id}, {e}")
                        time.sleep(task.retry_delay * task.retry_count)  # 递增延迟
                    else:
                        raise e

            # 超出最大重试次数
            raise Exception(f"任务执行失败，已重试 {task.max_retries} 次")

        except Exception as e:
            logger.error(f"任务函数执行失败: {e}")
            raise


    def _handle_task_error(self, task: Task, error: Exception):
        """处理任务错误"""
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.now()

        # 创建错误结果
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=error,
            created_at=task.created_at,
            completed_at=task.completed_at,
            metadata=task.metadata
        )

        # 存储结果
        self.completed_tasks[task.task_id] = task_result

        # 更新统计
        self.stats['failed_tasks'] += 1

        # 执行错误回调
        if task.error_callback:
            try:
                if asyncio.iscoroutinefunction(task.error_callback):
                    if self.event_loop:
                        asyncio.run_coroutine_threadsafe(
                            task.error_callback(task_result),
                            self.event_loop
                        )
                else:
                    task.error_callback(task_result)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")

    def _update_avg_execution_time(self, execution_time: float):
        """更新平均执行时间"""
        total_completed = self.stats['completed_tasks']
        current_avg = self.stats['avg_execution_time']

        # 增量更新平均值
        self.stats['avg_execution_time'] = (
            (current_avg * (total_completed - 1)) + execution_time
        ) / total_completed

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):

        """上下文管理器出口"""
        self.stop()


# 全局任务管理器实例
_global_task_manager: Optional[AsyncTaskManager] = None


def get_global_task_manager() -> AsyncTaskManager:
    """获取全局任务管理器实例"""
    global _global_task_manager

    if _global_task_manager is None:
        _global_task_manager = AsyncTaskManager()

    return _global_task_manager


def submit_async_task(task_type: TaskType, name: str, func: Callable,
                      args: tuple = (), kwargs: dict = {},
                      priority: TaskPriority = TaskPriority.NORMAL,
                      **task_kwargs) -> str:
    """
    提交异步任务到全局管理器

    Args:
        task_type: 任务类型
        name: 任务名称
        func: 要执行的函数
        args: 位置参数
        kwargs: 关键字参数
        priority: 任务优先级
        **task_kwargs: 其他任务参数

    Returns:
        任务ID
    """
    manager = get_global_task_manager()
    return manager.submit_task(
        task_type=task_type,
        name=name,
        func=func,
        args=args,
        kwargs=kwargs,
        priority=priority,
        **task_kwargs
    )


def get_task_status(task_id: str) -> Optional[TaskStatus]:
    """获取任务状态"""
    manager = get_global_task_manager()
    return manager.get_task_status(task_id)


def get_task_result(task_id: str) -> Optional[TaskResult]:
    """获取任务结果"""
    manager = get_global_task_manager()
    return manager.get_task_result(task_id)


def cancel_task(task_id: str) -> bool:
    """取消任务"""
    manager = get_global_task_manager()
    return manager.cancel_task(task_id)


async def submit_async_task_async(task_type: TaskType, name: str, func: Callable,
                                  args: tuple = (), kwargs: dict = {},
                                  priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """异步提交任务"""
    manager = get_global_task_manager()
    return await manager.submit_task_async(task_type, name, func, args, kwargs, priority)


async def wait_for_task_async(task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
    """异步等待任务完成"""
    manager = get_global_task_manager()
    return await manager.wait_for_task_async(task_id, timeout)
