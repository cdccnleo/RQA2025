"""
增强版并行数据加载器
实现智能任务调度、批量处理和动态资源分配
"""

import logging

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import psutil
from datetime import datetime
from src.data.loader.base_loader import BaseDataLoader

logger = get_infrastructure_logger('enhanced_parallel_loader')


class TaskPriority(Enum):

    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LoadTask:

    """加载任务"""
    task_id: str
    loader: BaseDataLoader
    start_date: str
    end_date: str
    frequency: str
    priority: TaskPriority = TaskPriority.NORMAL
    kwargs: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.kwargs is None:
            self.kwargs = {}


class EnhancedParallelLoadingManager:

    """
    增强版并行数据加载管理器

    特性：
    - 智能任务调度和优先级管理
    - 动态资源分配和负载均衡
    - 批量任务处理和结果聚合
    - 实时性能监控和统计
    """

    def __init__(self,


                 max_workers: Optional[int] = None,
                 enable_auto_scaling: bool = True,
                 batch_size: int = 10,
                 max_queue_size: int = 1000):
        """
        初始化增强版并行加载管理器

        Args:
            max_workers: 最大工作线程数，None表示自动检测
            enable_auto_scaling: 是否启用自动扩缩容
            batch_size: 批量处理大小
            max_queue_size: 最大队列大小
        """
        # 智能计算最优线程数
        cpu_count = psutil.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        # 基于CPU和内存的智能线程数计算
        if max_workers is None:
            # 考虑CPU核心数、内存大小和I / O密集型任务特性
            optimal_workers = min(
                cpu_count * 4,  # I / O密集型任务可以更多线程
                int(memory_gb * 2),  # 每GB内存2个线程
                32  # 最大限制
            )
            self.max_workers = max(4, optimal_workers)  # 最少4个线程
        else:
            self.max_workers = max_workers

        self.enable_auto_scaling = enable_auto_scaling
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # 线程池和任务队列
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='EnhancedLoader_'
        )
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = {}

        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_time': 0.0,
            'current_load': 0.0,
            'queue_size': 0,
            'active_tasks': 0,
            'executor_info': {
                'max_workers': self.max_workers,
                'thread_name_prefix': 'EnhancedLoader_'
            }
        }

        # 监控线程
        self.monitor_thread = None
        self.shutdown_event = threading.Event()

        if self.enable_auto_scaling:
            self._start_monitoring()

        logger.info(f"EnhancedParallelLoadingManager initialized with {self.max_workers} workers")

    def _start_monitoring(self):
        """启动性能监控线程"""
        self.monitor_thread = threading.Thread(
            target=self._monitor_performance,
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitor_thread.start()

    def _monitor_performance(self):
        """性能监控线程"""
        while not self.shutdown_event.is_set():
            try:
                # 计算当前负载
                active_count = len(self.active_tasks)
                total_workers = self.executor._max_workers
                self.stats['current_load'] = active_count / \
                    total_workers if total_workers > 0 else 0
                self.stats['queue_size'] = len(self.task_queue)

                # 智能调整工作线程数
                self._adjust_workers()

                # 每5秒检查一次
                self.shutdown_event.wait(5)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                self.shutdown_event.wait(10)  # 出错时等待更长时间

    def submit_task(self, task: LoadTask) -> str:
        """
        提交单个任务

        Args:
            task: 加载任务

        Returns:
            任务ID
        """
        if len(self.task_queue) >= self.max_queue_size:
            raise RuntimeError("Task queue is full")

        self.task_queue.append(task)
        self.active_tasks[task.task_id] = task
        self.stats['total_tasks'] += 1
        self.stats['queue_size'] = len(self.task_queue)

        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id

    def submit_batch(self, tasks: List[LoadTask]) -> List[str]:
        """
        批量提交任务

        Args:
            tasks: 任务列表

        Returns:
            任务ID列表
        """
        task_ids = []
        for task in tasks:
            try:
                task_id = self.submit_task(task)
                task_ids.append(task_id)
            except RuntimeError as e:
                logger.warning(f"Failed to submit task {task.task_id}: {e}")

        return task_ids

    def process(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        执行任务队列中的所有任务

        Args:
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        if not self.task_queue:
            return {}

        start_time = time.time()
        results = {}
        futures = []

        # 按优先级排序任务
        sorted_tasks = sorted(self.task_queue, key=lambda x: x.priority.value, reverse=True)

        # 批量提交任务
        for task in sorted_tasks:
            future = self.executor.submit(self._execute_single_task, task)
            futures.append((task.task_id, future))
            self.active_tasks[task.task_id] = task
            self.stats['total_tasks'] += 1
            self.stats['active_tasks'] += 1

        # 收集结果
        for task_id, future in futures:
            try:
                result = future.result(timeout=timeout)
                results[task_id] = result
                self.stats['completed_tasks'] += 1

            except Exception as e:
                logger.error(f"Task {task_id} processing failed: {e}")
                results[task_id] = None
                self.stats['failed_tasks'] += 1

            finally:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                self.stats['active_tasks'] -= 1

        # 更新统计信息
        execution_time = time.time() - start_time
        if self.stats['completed_tasks'] > 0:
            self.stats['avg_task_time'] = execution_time / self.stats['completed_tasks']

        # 清空任务队列
        self.task_queue.clear()
        self.stats['queue_size'] = 0

        return results

    def process_single_task(self, task: LoadTask) -> Any:
        """
        执行单个任务

        Args:
            task: 加载任务

        Returns:
            执行结果
        """
        try:
            # 根据加载器类型调用相应的方法
            if hasattr(task.loader, 'load_data'):
                # 使用load_data方法（股票、指数、财务数据加载器）
                # 确保参数正确传递
                kwargs = task.kwargs.copy()

                # 对于股票数据加载器，symbol应该是第一个位置参数
                if 'symbol' in kwargs:
                    symbol = kwargs.pop('symbol')
                    result = task.loader.load_data(
                        symbol,
                        start_date=task.start_date,
                        end_date=task.end_date,
                        **kwargs
                    )
                else:
                    # 对于其他加载器，使用关键字参数
                    result = task.loader.load_data(
                        start_date=task.start_date,
                        end_date=task.end_date,
                        **kwargs
                    )
            else:
                # 使用通用的load方法
                result = task.loader.load(
                    task.start_date,
                    task.end_date,
                    task.frequency,
                    **task.kwargs
                )

            logger.debug(f"Task {task.task_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task {task.task_id} processing failed: {e}")
            raise

    def _adjust_workers(self):
        """智能调整工作线程数"""
        if not self.enable_auto_scaling:
            return

        current_load = self.stats['current_load']
        queue_size = self.stats['queue_size']
        avg_task_time = self.stats['avg_task_time']

        # 获取系统资源使用情况
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # 智能扩缩容策略
        current_workers = self.executor._max_workers

        # 扩容条件：负载高且系统资源充足
        if (current_load > 0.8 or queue_size > self.batch_size) and cpu_percent < 80 and memory_percent < 85:
            new_workers = min(current_workers * 2, 32)
            if new_workers > current_workers:
                self._resize_executor(new_workers)
                logger.info(f"扩容到 {new_workers} 个工作线程")

        # 缩容条件：负载低且资源使用率高
        elif current_load < 0.3 and (cpu_percent > 90 or memory_percent > 90):
            new_workers = max(current_workers // 2, 4)
            if new_workers < current_workers:
                self._resize_executor(new_workers)
                logger.info(f"缩容到 {new_workers} 个工作线程")

    def processor(self, new_workers: int):
        """动态调整线程池大小"""
        try:
            # 创建新的线程池
            new_executor = ThreadPoolExecutor(
                max_workers=new_workers,
                thread_name_prefix='EnhancedLoader_'
            )

            # 等待当前任务完成
            self.executor.shutdown(wait=True)

            # 替换线程池
            self.executor = new_executor
            self.stats['executor_info']['max_workers'] = new_workers

        except Exception as e:
            logger.error(f"调整线程池大小失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            **self.stats,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'executor_info': {
                'max_workers': self.executor._max_workers,
                'thread_name_prefix': self.executor._thread_name_prefix
            }
        }

    def shutdown(self, wait: bool = True):
        """关闭管理器"""
        self.executor.shutdown(wait=wait)
        logger.info("EnhancedParallelLoadingManager shutdown")


def create_enhanced_loader(config: Optional[Dict[str, Any]] = None) -> EnhancedParallelLoadingManager:
    """
    创建增强版并行加载器

    Args:
        config: 配置字典

    Returns:
        增强版并行加载管理器
    """
    if config is None:
        config = {}

    return EnhancedParallelLoadingManager(
        max_workers=config.get('max_workers'),
        enable_auto_scaling=config.get('enable_auto_scaling', True),
        batch_size=config.get('batch_size', 10),
        max_queue_size=config.get('max_queue_size', 1000)
    )
