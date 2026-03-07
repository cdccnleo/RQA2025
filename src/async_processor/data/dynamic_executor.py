"""
RQA2025 动态执行器

提供动态任务执行和资源管理功能
"""

from typing import Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


class DynamicExecutor:

    """动态执行器，支持线程和进程执行"""

    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        """
        初始化动态执行器

        Args:
            max_workers: 最大工作线程 / 进程数
            use_processes: 是否使用进程池（否则使用线程池）
        """
        self.max_workers = max_workers
        self.use_processes = use_processes

        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            执行结果
        """
        try:
            future = self.executor.submit(func, *args, **kwargs)
            return future.result()
        except Exception as e:
            logger.error(f"执行失败: {e}")
            raise

    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """
        映射执行

        Args:
            func: 要执行的函数
            iterable: 可迭代对象

        Returns:
            结果列表
        """
        try:
            futures = [self.executor.submit(func, item) for item in iterable]
            return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"映射执行失败: {e}")
            raise

    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("动态执行器已关闭")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.shutdown()
