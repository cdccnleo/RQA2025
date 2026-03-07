"""
托管执行器

单个执行器的托管实现。

从executor_manager.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import concurrent.futures
import time
from typing import Any, Callable, Optional
from datetime import datetime

from .executor_models import ExecutorType, ExecutorStatus, ExecutorMetrics

logger = logging.getLogger(__name__)


class ManagedExecutor:
    """
    Managed Executor Class
    托管执行器类

    Provides a managed wrapper for thread/process pool executors
    为线程池/进程池执行器提供托管包装
    """

    def __init__(self, executor_type: ExecutorType = ExecutorType.THREAD_POOL,
                 max_workers: int = 4, executor_id: Optional[str] = None):
        self.executor_type = executor_type
        self.max_workers = max_workers
        self.executor_id = executor_id or f"{executor_type.value}_{id(self)}"
        
        # 创建执行器
        if executor_type == ExecutorType.THREAD_POOL:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        elif executor_type == ExecutorType.PROCESS_POOL:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
        
        # 状态和指标
        self.status = ExecutorStatus.INITIALIZING
        self.metrics = ExecutorMetrics()
        self.created_at = datetime.now()
        
        # 完成初始化
        if self.executor:
            self.status = ExecutorStatus.RUNNING
            logger.info(f"托管执行器创建成功: {self.executor_id} ({executor_type.value}, {max_workers} workers)")
        else:
            self.status = ExecutorStatus.ERROR
            logger.error(f"托管执行器创建失败: {self.executor_id}")

    def submit(self, fn: Callable, *args, **kwargs) -> Optional[concurrent.futures.Future]:
        """Submit a task to executor"""
        if self.status != ExecutorStatus.RUNNING:
            logger.warning(f"执行器 {self.executor_id} 未运行，无法提交任务")
            return None

        try:
            self.metrics.record_task_start()
            future = self.executor.submit(fn, *args, **kwargs)
            
            # 添加回调来记录完成
            future.add_done_callback(self._task_done_callback)
            
            return future
            
        except Exception as e:
            logger.error(f"任务提交失败: {e}")
            self.metrics.record_task_failure()
            return None

    def _task_done_callback(self, future: concurrent.futures.Future):
        """Task completion callback"""
        try:
            if future.exception():
                self.metrics.record_task_failure()
                logger.warning(f"任务执行失败: {future.exception()}")
            else:
                execution_time = time.time() - self.metrics.last_updated.timestamp()
                self.metrics.record_task_completion(execution_time)
                
        except Exception as e:
            logger.error(f"任务回调异常: {e}")

    def shutdown(self, wait: bool = True):
        """Shutdown executor"""
        if self.status == ExecutorStatus.STOPPED:
            return

        self.status = ExecutorStatus.STOPPING
        
        if self.executor:
            self.executor.shutdown(wait=wait)
        
        self.status = ExecutorStatus.STOPPED
        logger.info(f"执行器已关闭: {self.executor_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics"""
        return self.metrics.to_dict()

    def get_status(self) -> ExecutorStatus:
        """Get executor status"""
        return self.status


__all__ = ['ManagedExecutor']

