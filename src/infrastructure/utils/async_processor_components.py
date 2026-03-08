#!/usr/bin/env python3
"""
异步处理队列组件

提供异步任务处理能力，提升系统并发性能和响应性。

作者: 系统架构师
创建时间: 2025-01-28
版本: 2.0.0
"""

import threading
import queue
import time
from typing import Callable, Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime
import logging

from src.core.constants import DEFAULT_BATCH_SIZE, MAX_RECORDS

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    submitted_at: float
    priority: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = time.time()
        if self.completed_at is None:
            self.completed_at = time.time()


@dataclass
class ProcessorStats:
    """处理器统计信息"""
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    queue_size: int = 0
    active_workers: int = 0
    avg_processing_time: float = 0.0
    uptime_seconds: float = 0.0
    last_activity: Optional[float] = None


class AsyncProcessor:
    """异步处理器"""

    def __init__(self, max_workers: int = 10, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # 统计信息
        self.stats = ProcessorStats()
        self.start_time = time.time()
        self._task_id_counter = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """启动异步处理器"""
        if self.is_running:
            logger.warning("异步处理器已在运行中")
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.start_time = time.time()

        logger.info(f"异步处理器已启动，工作者数量: {self.max_workers}，队列大小: {self.queue_size}")

    def stop(self) -> None:
        """停止异步处理器"""
        if not self.is_running:
            logger.warning("异步处理器未在运行")
            return

        logger.info("正在停止异步处理器...")
        self.is_running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                logger.warning("工作线程未能在5秒内停止")

        self.executor.shutdown(wait=True)
        logger.info("异步处理器已停止")

    def submit_task(self, func: Callable, *args, priority: int = 1,
                   timeout: Optional[float] = None, **kwargs) -> Optional[str]:
        """提交任务到异步队列

        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 任务优先级（暂时未实现）
            timeout: 任务超时时间
            **kwargs: 关键字参数

        Returns:
            任务ID，如果提交失败则返回None
        """
        try:
            with self._lock:
                task_id = f"task_{self._task_id_counter}"
                self._task_id_counter += 1

            task_info = TaskInfo(
                task_id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                submitted_at=time.time(),
                priority=priority,
                timeout=timeout
            )

            self.task_queue.put(task_info, timeout=1)

            with self._lock:
                self.stats.total_submitted += 1
                self.stats.queue_size = self.task_queue.qsize()
                self.stats.last_activity = time.time()

            logger.debug(f"任务 {task_id} 已提交到队列")
            return task_id

        except queue.Full:
            logger.warning("任务队列已满，拒绝新任务")
            return None

    def _process_queue(self) -> None:
        """处理任务队列"""
        while self.is_running:
            try:
                # 获取任务
                task_info = self.task_queue.get(timeout=1)

                # 提交到线程池执行
                future = self.executor.submit(self._execute_task, task_info)
                future.add_done_callback(self._task_completed)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"队列处理错误: {e}")
                with self._lock:
                    self.stats.total_failed += 1

    def _execute_task(self, task_info: TaskInfo) -> TaskResult:
        """执行单个任务"""
        start_time = time.time()

        try:
            # 检查超时
            if task_info.timeout and (time.time() - task_info.submitted_at) > task_info.timeout:
                raise TimeoutError(f"任务 {task_info.task_id} 已超时")

            result = task_info.func(*task_info.args, **task_info.kwargs)
            processing_time = time.time() - start_time

            return TaskResult(
                task_id=task_info.task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                executed_at=start_time,
                completed_at=time.time()
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return TaskResult(
                task_id=task_info.task_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                executed_at=start_time,
                completed_at=time.time()
            )

    def _task_completed(self, future: Future) -> None:
        """任务完成回调"""
        try:
            task_result = future.result()

            with self._lock:
                if task_result.success:
                    self.stats.total_completed += 1
                    # 更新平均处理时间
                    if self.stats.total_completed == 1:
                        self.stats.avg_processing_time = task_result.processing_time
                    else:
                        total_completed = self.stats.total_completed
                        self.stats.avg_processing_time = (
                            (self.stats.avg_processing_time * (total_completed - 1)) +
                            task_result.processing_time
                        ) / total_completed
                else:
                    self.stats.total_failed += 1

                self.stats.last_activity = time.time()

        except Exception as e:
            logger.error(f"任务完成处理错误: {e}")
            with self._lock:
                self.stats.total_failed += 1

    def get_stats(self) -> ProcessorStats:
        """获取处理器统计信息"""
        with self._lock:
            self.stats.queue_size = self.task_queue.qsize()
            self.stats.active_workers = self.max_workers
            self.stats.uptime_seconds = time.time() - self.start_time

            return ProcessorStats(**self.stats.__dict__)


# 全局异步处理器实例
_async_processor = None


def get_async_processor():
    """获取全局异步处理器实例"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncProcessor()
    return _async_processor


def submit_async_task(func: Callable, *args, **kwargs):
    """提交异步任务的便捷函数"""
    processor = get_async_processor()
    if not processor.is_running:
        processor.start()
    return processor.submit_task(func, *args, **kwargs)
