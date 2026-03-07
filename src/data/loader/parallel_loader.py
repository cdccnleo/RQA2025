"""
优化并行加载器 - 性能优化版本
"""
# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from src.infrastructure.logging import get_infrastructure_logger
from typing import List, Dict, Any, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum
import threading


logger = get_infrastructure_logger('__name__')


class TaskStatus(Enum):

    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class LoadResult:

    """加载结果"""
    task_id: str
    status: TaskStatus
    data: Any = None
    error: str = None
    duration: float = 0.0
    retry_count: int = 0


class OptimizedParallelLoader:

    """
    优化的并行数据加载器
    支持任务调度、错误恢复、性能监控
    """

    def __init__(self, max_workers: int = 8, timeout: int = 30, max_retries: int = 3):
        """
        初始化并行加载器

        Args:
            max_workers: 最大工作线程数
            timeout: 任务超时时间（秒）
            max_retries: 最大重试次数
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 任务队列和状态管理
        self.task_queue = []
        self.task_status = {}
        self.task_results = {}

        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'timeout_tasks': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

        # 线程锁
        self._lock = threading.RLock()

        logger.info(f"OptimizedParallelLoader initialized with {max_workers} workers")

    def load(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        加载数据 - 兼容性接口

        Args:
            tasks: 任务列表，每个任务为包含func和kwargs的字典

        Returns:
            Dict[str, Any]: 任务结果字典
        """
        # 转换任务格式
        converted_tasks = []
        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            config = {
                'func': task.get('func'),
                'kwargs': task.get('kwargs', {})
            }
            converted_tasks.append((task_id, config))

        # 使用_batch_load执行
        results = self._batch_load(converted_tasks)

        # 转换结果格式
        converted_results = {}
        for task_id, load_result in results.items():
            if load_result.status == TaskStatus.COMPLETED:
                converted_results[task_id] = load_result.data
            else:
                converted_results[task_id] = None

        return converted_results

    def batch_load(self, tasks: List[Tuple[str, Dict]],


                   priority: bool = False) -> Dict[str, LoadResult]:
        """
        批量并行加载数据 - 公共接口

        Args:
            tasks: 任务列表，每个任务为 (task_id, config) 元组
            priority: 是否启用优先级调度

        Returns:
            Dict[str, LoadResult]: 任务结果字典
        """
        return self._batch_load(tasks, priority)

    def _batch_load(self, tasks: List[Tuple[str, Dict]],


                    priority: bool = False) -> Dict[str, LoadResult]:
        """
        批量并行加载数据

        Args:
            tasks: 任务列表，每个任务为 (task_id, config) 元组
            priority: 是否启用优先级调度

        Returns:
            Dict[str, LoadResult]: 任务结果字典
        """
        with self._lock:
            # 重置结果
            self.task_results.clear()
            self.task_status.clear()

            # 准备任务
            self.task_queue = tasks.copy()
            if priority:
                self.task_queue.sort(key=lambda x: x[1].get('priority', 0), reverse=True)

            # 提交任务
            futures = {}
            for task_id, config in self.task_queue:
                future = self.executor.submit(self._load_single_with_retry, task_id, config)
                futures[future] = task_id
                self.task_status[task_id] = TaskStatus.PENDING

            # 收集结果
            self._collect_results(futures)

            # 更新统计
            self._update_stats()

            return self.task_results

    def _load_single_with_retry(self, task_id: str, config: Dict) -> LoadResult:
        """
        单个数据加载任务（带重试）

        Args:
            task_id: 任务ID
            config: 任务配置

        Returns:
            LoadResult: 加载结果
        """
        start_time = time.time()
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # 更新任务状态
                self.task_status[task_id] = TaskStatus.RUNNING

                # 执行加载任务
                result = self._load_single(task_id, config)
                result.duration = time.time() - start_time
                result.retry_count = retry_count

                # 更新状态
                self.task_status[task_id] = TaskStatus.COMPLETED

                return result

            except TimeoutError:
                retry_count += 1
                logger.warning(f"Task {task_id} timeout, retry {retry_count}/{self.max_retries}")
                self.task_status[task_id] = TaskStatus.TIMEOUT

            except Exception as e:
                retry_count += 1
                logger.error(f"Task {task_id} failed, retry {retry_count}/{self.max_retries}: {e}")
                self.task_status[task_id] = TaskStatus.FAILED

                if retry_count > self.max_retries:
                    return LoadResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        duration=time.time() - start_time,
                        retry_count=retry_count
                    )

        return LoadResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="Max retries exceeded",
            duration=time.time() - start_time,
            retry_count=retry_count
        )

    def _load_single(self, task_id: str, config: Dict) -> LoadResult:
        """
        单个数据加载任务

        Args:
            task_id: 任务ID
            config: 任务配置

        Returns:
            LoadResult: 加载结果
        """
        start_time = time.time()

        try:
            # 模拟加载过程
            load_time = config.get('load_time', 0.1)  # 默认100ms
            time.sleep(load_time)

            # 模拟数据
            data = {
                "id": task_id,
                "timestamp": time.time(),
                "config": config,
                "load_time": load_time
            }

            return LoadResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                data=data,
                duration=time.time() - start_time
            )

        except Exception as e:
            return LoadResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time
            )

    def _collect_results(self, futures: Dict):
        """
        收集任务结果

        Args:
            futures: Future对象字典
        """
        try:
            for future in as_completed(futures, timeout=self.timeout):
                task_id = futures[future]
                try:
                    result = future.result(timeout=1)  # 1秒超时获取结果
                    self.task_results[task_id] = result

                    # 更新统计
                    with self._lock:
                        if result.status == TaskStatus.COMPLETED:
                            self.stats['completed_tasks'] += 1
                        elif result.status == TaskStatus.FAILED:
                            self.stats['failed_tasks'] += 1
                        elif result.status == TaskStatus.TIMEOUT:
                            self.stats['timeout_tasks'] += 1

                        self.stats['total_time'] += result.duration

                except TimeoutError:
                    logger.error(f"Task {task_id} result collection timeout")
                    self.task_results[task_id] = LoadResult(
                        task_id=task_id,
                        status=TaskStatus.TIMEOUT,
                        error="Result collection timeout"
                    )
                    with self._lock:
                        self.stats['timeout_tasks'] += 1

                except Exception as e:
                    logger.error(f"Task {task_id} result collection failed: {e}")
                    self.task_results[task_id] = LoadResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=f"Result collection failed: {str(e)}"
                    )
                    with self._lock:
                        self.stats['failed_tasks'] += 1

        except TimeoutError:
            logger.error("Batch load timeout")
            # 处理未完成的任务
            for future in futures:
                if not future.done():
                    task_id = futures[future]
                    self.task_results[task_id] = LoadResult(
                        task_id=task_id,
                        status=TaskStatus.TIMEOUT,
                        error="Batch load timeout"
                    )
                    with self._lock:
                        self.stats['timeout_tasks'] += 1

    def _update_stats(self):
        """更新性能统计"""
        with self._lock:
            self.stats['total_tasks'] = len(self.task_queue)
            if self.stats['total_tasks'] > 0:
                self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_tasks']

    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            total_tasks = self.stats['total_tasks']
            if total_tasks == 0:
                success_rate = 0.0
            else:
                success_rate = (self.stats['completed_tasks'] / total_tasks) * 100

            return {
                **self.stats,
                'success_rate': f"{success_rate:.2f}%",
                'avg_time_ms': self.stats['avg_time'] * 1000,
                'active_workers': len([f for f in self.executor._threads if f.is_alive()])
            }

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskStatus]: 任务状态
        """
        return self.task_status.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否取消成功
        """
        # 这里可以实现任务取消逻辑
        logger.info(f"Cancelling task {task_id}")
        return True

    def shutdown(self, wait: bool = True):
        """
        关闭并行加载器

        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)
        logger.info("OptimizedParallelLoader shutdown")


# 向后兼容性别名
ParallelDataLoader = OptimizedParallelLoader
