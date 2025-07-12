"""
并行数据加载器实现
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd

from src.infrastructure.utils.exceptions import DataLoaderError
from src.data.interfaces import IDataModel
from src.data.base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class ParallelLoadingManager:
    """
    并行数据加载管理器，负责协调多个数据加载器的并行执行
    """
    def __init__(self, max_workers: int = 4):
        """
        初始化并行加载管理器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        self.results_cache = {}

        logger.info(f"ParallelLoadingManager initialized with {max_workers} workers")

    def submit_task(
        self,
        loader: BaseDataLoader,
        task_id: str,
        start_date: str,
        end_date: str,
        frequency: str,
        **kwargs
    ) -> None:
        """
        提交加载任务

        Args:
            loader: 数据加载器
            task_id: 任务ID
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数
        """
        if task_id in self.active_tasks:
            logger.warning(f"Task {task_id} is already running")
            return

        future = self.executor.submit(
            loader.load,
            start_date,
            end_date,
            frequency,
            **kwargs
        )
        self.active_tasks[task_id] = {
            'future': future,
            'start_time': datetime.now(),
            'params': {
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency,
                **kwargs
            }
        }
        logger.debug(f"Submitted task {task_id}")

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[IDataModel]:
        """
        获取任务结果

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            Optional[IDataModel]: 数据模型，如果任务不存在或失败则返回None

        Raises:
            TimeoutError: 如果等待超时
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found")
            return None

        try:
            future = self.active_tasks[task_id]['future']
            result = future.result(timeout=timeout)

            # 缓存结果
            self.results_cache[task_id] = {
                'result': result,
                'completion_time': datetime.now(),
                'params': self.active_tasks[task_id]['params']
            }

            # 清理活动任务
            del self.active_tasks[task_id]

            return result
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            return None

    def wait_all(self, timeout: Optional[float] = None) -> Dict[str, IDataModel]:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            Dict[str, IDataModel]: 任务ID到结果的映射
        """
        results = {}
        try:
            futures = {
                task_id: task['future']
                for task_id, task in self.active_tasks.items()
            }

            for task_id, future in futures.items():
                try:
                    result = future.result(timeout=timeout)
                    results[task_id] = result

                    # 更新缓存
                    self.results_cache[task_id] = {
                        'result': result,
                        'completion_time': datetime.now(),
                        'params': self.active_tasks[task_id]['params']
                    }
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {str(e)}")

            # 清理完成的任务
            self.active_tasks.clear()

            return results
        except Exception as e:
            logger.error(f"Error waiting for tasks: {str(e)}")
            return results

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功取消
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found")
            return False

        try:
            future = self.active_tasks[task_id]['future']
            cancelled = future.cancel()
            if cancelled:
                del self.active_tasks[task_id]
                logger.info(f"Task {task_id} cancelled")
            return cancelled
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务状态信息
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            future = task['future']
            return {
                'status': 'running' if not future.done() else 'completed',
                'start_time': task['start_time'].isoformat(),
                'params': task['params']
            }
        elif task_id in self.results_cache:
            cache = self.results_cache[task_id]
            return {
                'status': 'cached',
                'completion_time': cache['completion_time'].isoformat(),
                'params': cache['params']
            }
        else:
            return {'status': 'not_found'}

    def clear_cache(self, older_than: Optional[datetime] = None) -> int:
        """
        清理缓存

        Args:
            older_than: 清理此时间之前的缓存

        Returns:
            int: 清理的缓存条目数
        """
        if older_than is None:
            count = len(self.results_cache)
            self.results_cache.clear()
            return count

        keys_to_remove = [
            k for k, v in self.results_cache.items()
            if v['completion_time'] < older_than
        ]

        for k in keys_to_remove:
            del self.results_cache[k]

        return len(keys_to_remove)

    def shutdown(self, wait: bool = True) -> None:
        """
        关闭加载管理器

        Args:
            wait: 是否等待所有任务完成
        """
        try:
            self.executor.shutdown(wait=wait)
            logger.info("ParallelLoadingManager shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
