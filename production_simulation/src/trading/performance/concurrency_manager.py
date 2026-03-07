import logging
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
RQA2025 交易层并发处理优化模块

实现异步处理和并发控制机制，提升高频交易并发性能
"""

import asyncio
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from dataclasses import dataclass
from datetime import datetime
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# 导入统一基础设施集成层
try:
    from src.core.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


class TaskPriority(Enum):

    """任务优先级枚举"""
    CRITICAL = 0      # 关键任务（如紧急风控）
    HIGH = 1         # 高优先级（如市价单）
    NORMAL = 2       # 普通优先级（如限价单）
    LOW = 3          # 低优先级（如数据更新）


class ConcurrencyMode(Enum):

    """并发模式枚举"""
    ASYNC = "async"              # 异步模式
    THREAD = "thread"           # 线程模式
    PROCESS = "process"         # 进程模式
    HYBRID = "hybrid"           # 混合模式


@dataclass
class Task:

    """任务对象"""
    task_id: str
    priority: TaskPriority
    function: Callable
    args: tuple
    kwargs: dict
    created_at: datetime
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    def __lt__(self, other):
        """优先级比较（用于优先级队列）"""
        return self.priority.value < other.priority.value


@dataclass
class ConcurrencyStats:

    """并发统计信息"""
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    queued_tasks: int
    average_execution_time: float
    peak_concurrent_tasks: int
    thread_pool_utilization: float
    queue_length: int
    rejected_tasks: int


class AsyncTaskManager:

    """异步任务管理器"""

    def __init__(self, max_concurrent: int = 1000, mode: ConcurrencyMode = ConcurrencyMode.ASYNC):

        self.max_concurrent = max_concurrent
        self.mode = mode

        # 异步组件
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 任务统计
        self.stats = ConcurrencyStats(
            active_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            queued_tasks=0,
            average_execution_time=0.0,
            peak_concurrent_tasks=0,
            thread_pool_utilization=0.0,
            queue_length=0,
            rejected_tasks=0
        )

        # 执行时间记录
        self.execution_times: List[float] = []

        # 控制标志
        self.running = False

        logger.info(f"异步任务管理器初始化完成，最大并发数: {max_concurrent}")

    async def submit_task(self, coro: Awaitable, priority: TaskPriority = TaskPriority.NORMAL) -> Any:
        """提交异步任务"""
        async with self.semaphore:
            self.stats.active_tasks += 1
            self.stats.peak_concurrent_tasks = max(
                self.stats.peak_concurrent_tasks,
                self.stats.active_tasks
            )

            start_time = time.time()
            try:
                result = await coro
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                # 更新平均执行时间
                if len(self.execution_times) > 1000:
                    self.execution_times = self.execution_times[-1000:]
                self.stats.average_execution_time = sum(
                    self.execution_times) / len(self.execution_times)

                self.stats.completed_tasks += 1
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                self.stats.failed_tasks += 1
                logger.error(f"异步任务执行失败: {e}")
                raise e

            finally:
                self.stats.active_tasks -= 1

    async def submit_callable(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs) -> Any:
        """提交可调用对象作为异步任务"""
        async def wrapper():
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)

        return await self.submit_task(wrapper(), priority)

    def get_stats(self) -> ConcurrencyStats:
        """获取统计信息"""
        return self.stats


class ThreadPoolManager:

    """线程池管理器"""

    def __init__(self, max_workers: int = 50):

        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="trading_thread")

        # 统计信息
        self.stats = ConcurrencyStats(
            active_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            queued_tasks=0,
            average_execution_time=0.0,
            peak_concurrent_tasks=0,
            thread_pool_utilization=0.0,
            queue_length=0,
            rejected_tasks=0
        )

        # 任务队列
        self.task_queue = queue.PriorityQueue()

        logger.info(f"线程池管理器初始化完成，最大工作线程数: {max_workers}")

    def submit_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,


                    **kwargs) -> concurrent.futures.Future:
        """提交任务到线程池"""
        future = self.executor.submit(func, *args, **kwargs)

        # 添加完成回调
        future.add_done_callback(self._task_completion_callback)

        self.stats.queued_tasks += 1
        return future

    def _task_completion_callback(self, future: concurrent.futures.Future):
        """任务完成回调"""
        self.stats.queued_tasks -= 1

        if future.exception():
            self.stats.failed_tasks += 1
        else:
            self.stats.completed_tasks += 1

    def get_stats(self) -> ConcurrencyStats:
        """获取统计信息"""
        # 更新线程池利用率
        active_threads = len([t for t in self.executor._threads if t.is_alive()])
        self.stats.thread_pool_utilization = active_threads / self.max_workers
        self.stats.queue_length = self.task_queue.qsize()

        return self.stats

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)
        logger.info("线程池管理器已关闭")


class PriorityTaskScheduler:

    """优先级任务调度器"""

    def __init__(self, max_concurrent: int = 100):

        self.max_concurrent = max_concurrent
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = set()
        self.task_lock = threading.RLock()

        # 统计信息
        self.stats = ConcurrencyStats(
            active_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            queued_tasks=0,
            average_execution_time=0.0,
            peak_concurrent_tasks=0,
            thread_pool_utilization=0.0,
            queue_length=0,
            rejected_tasks=0
        )

        # 工作线程
        self.workers: List[threading.Thread] = []
        self.running = False

        logger.info(f"优先级任务调度器初始化完成，最大并发数: {max_concurrent}")

    def start(self):
        """启动调度器"""
        if self.running:
            return

        self.running = True

        # 启动工作线程
        for i in range(self.max_concurrent):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"priority_worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"优先级任务调度器已启动，工作线程数: {self.max_concurrent}")

    def stop(self):
        """停止调度器"""
        if not self.running:
            return

        self.running = False

        # 等待工作线程结束
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5)

        self.workers.clear()
        logger.info("优先级任务调度器已停止")

    def submit_task(self, task: Task) -> bool:
        """提交优先级任务"""
        if not self.running:
            logger.warning("调度器未运行，无法提交任务")
            return False

        try:
            self.task_queue.put(task, timeout=1.0)  # 1秒超时
            self.stats.queued_tasks += 1
            return True
        except queue.Full:
            self.stats.rejected_tasks += 1
            logger.warning("任务队列已满，任务被拒绝")
            return False

    def submit_function(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,


                        timeout: Optional[float] = None, **kwargs) -> bool:
        """提交函数作为优先级任务"""
        task = Task(
            task_id=f"task_{int(time.time() * 1000000)}",
            priority=priority,
            function=func,
            args=args,
            kwargs=kwargs,
            created_at=datetime.now(),
            timeout=timeout
        )

        return self.submit_task(task)

    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 获取任务（带超时）
                task = self.task_queue.get(timeout=0.1)

                with self.task_lock:
                    self.active_tasks.add(task.task_id)
                    self.stats.active_tasks += 1
                    self.stats.queued_tasks -= 1
                    self.stats.peak_concurrent_tasks = max(
                        self.stats.peak_concurrent_tasks,
                        self.stats.active_tasks
                    )

                # 执行任务
                self._execute_task(task)

                # 标记任务完成
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"工作线程异常: {e}")

    def _execute_task(self, task: Task):
        """执行任务"""
        start_time = time.time()

        try:
            # 执行任务函数
            task.function(*task.args, **task.kwargs)
            execution_time = time.time() - start_time

            # 更新统计
            self.stats.completed_tasks += 1
            self.stats.average_execution_time = (
                (self.stats.average_execution_time * (self.stats.completed_tasks - 1) + execution_time)
                / self.stats.completed_tasks
            )

            logger.debug(f"任务 {task.task_id} 执行完成，耗时: {execution_time:.4f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats.failed_tasks += 1

            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(
                    f"任务 {task.task_id} 执行失败，重试 {task.retry_count}/{task.max_retries}: {e}")
                # 重新提交任务
                self.submit_task(task)
            else:
                logger.error(f"任务 {task.task_id} 执行失败，已达到最大重试次数: {e}")

        finally:
            with self.task_lock:
                self.active_tasks.discard(task.task_id)
                self.stats.active_tasks -= 1

    def get_stats(self) -> ConcurrencyStats:
        """获取统计信息"""
        with self.task_lock:
            self.stats.queue_length = self.task_queue.qsize()
            return self.stats


class TradingConcurrencyManager:

    """交易并发管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化交易并发管理器"""
        self.config = config or {}

        # 基础设施集成
        self._infrastructure_adapter = None
        self._monitoring = None

        # 初始化基础设施集成
        self._init_infrastructure_integration()

        # 从配置中获取参数
        self._load_config()

        # 并发组件
        self.async_manager: Optional[AsyncTaskManager] = None
        self.thread_pool_manager: Optional[ThreadPoolManager] = None
        self.priority_scheduler: Optional[PriorityTaskScheduler] = None

        # 初始化并发组件
        self._init_concurrency_components()

        logger.info("交易并发管理器初始化完成")

    def _init_infrastructure_integration(self):
        """初始化基础设施集成"""
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            logger.warning("统一基础设施集成层不可用，并发管理器使用降级模式")
            return

        try:
            self._infrastructure_adapter = get_trading_layer_adapter()

            if self._infrastructure_adapter:
                services = self._infrastructure_adapter.get_infrastructure_services()
                self._monitoring = services.get('monitoring')

                logger.info("交易并发管理器成功连接统一基础设施集成层")
        except Exception as e:
            logger.error(f"基础设施集成初始化失败: {e}")

    def _load_config(self):
        """从配置中获取参数"""
        try:
            if self._infrastructure_adapter:
                services = self._infrastructure_adapter.get_infrastructure_services()
                config_manager = services.get('config_manager')

                if config_manager:
                    self.max_async_concurrent = config_manager.get(
                        'trading.concurrency.max_async', 1000)
                    self.max_thread_workers = config_manager.get(
                        'trading.concurrency.max_threads', 50)
                    self.max_priority_concurrent = config_manager.get(
                        'trading.concurrency.max_priority', 100)
                    self.concurrency_mode = ConcurrencyMode(
                        config_manager.get('trading.concurrency.mode', ConcurrencyMode.HYBRID.value)
                    )
                else:
                    self._set_default_config()
            else:
                self._set_default_config()

        except Exception as e:
            logger.warning(f"配置加载失败，使用默认值: {e}")
            self._set_default_config()

    def _set_default_config(self):
        """设置默认配置"""
        self.max_async_concurrent = 1000
        self.max_thread_workers = 50
        self.max_priority_concurrent = 100
        self.concurrency_mode = ConcurrencyMode.HYBRID

    def _init_concurrency_components(self):
        """初始化并发组件"""
        # 根据模式初始化相应的组件
        if self.concurrency_mode in [ConcurrencyMode.ASYNC, ConcurrencyMode.HYBRID]:
            self.async_manager = AsyncTaskManager(self.max_async_concurrent, self.concurrency_mode)

        if self.concurrency_mode in [ConcurrencyMode.THREAD, ConcurrencyMode.HYBRID]:
            self.thread_pool_manager = ThreadPoolManager(self.max_thread_workers)

        if self.concurrency_mode in [ConcurrencyMode.HYBRID]:
            self.priority_scheduler = PriorityTaskScheduler(self.max_priority_concurrent)
            self.priority_scheduler.start()

    async def submit_async_task(self, coro: Awaitable, priority: TaskPriority = TaskPriority.NORMAL) -> Any:
        """提交异步任务"""
        if self.async_manager:
            return await self.async_manager.submit_task(coro, priority)
        else:
            logger.warning("异步管理器不可用")
            raise RuntimeError("异步管理器不可用")

    def submit_thread_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs) -> concurrent.futures.Future:
        """提交线程任务"""
        if self.thread_pool_manager:
            return self.thread_pool_manager.submit_task(func, *args, priority=priority, **kwargs)
        else:
            logger.warning("线程池管理器不可用")
            raise RuntimeError("线程池管理器不可用")

    def submit_priority_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,


                             timeout: Optional[float] = None, **kwargs) -> bool:
        """提交优先级任务"""
        if self.priority_scheduler:
            return self.priority_scheduler.submit_function(func, *args, priority=priority, timeout=timeout, **kwargs)
        else:
            logger.warning("优先级调度器不可用")
            raise RuntimeError("优先级调度器不可用")

    def submit_trading_task(self, func: Callable, *args, task_type: str = "normal", **kwargs) -> Union[concurrent.futures.Future, bool, Any]:
        """智能提交交易任务（根据任务类型自动选择最佳并发模式）"""
        # 根据任务类型确定优先级
        priority_map = {
            "emergency": TaskPriority.CRITICAL,
            "market_order": TaskPriority.HIGH,
            "limit_order": TaskPriority.NORMAL,
            "data_update": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL
        }

        priority = priority_map.get(task_type, TaskPriority.NORMAL)

        # 基础设施集成：记录监控指标
        if self._monitoring:
            try:
                self._monitoring.record_metric(
                    'concurrency_task_submitted',
                    1,
                    {
                        'task_type': task_type,
                        'priority': priority.value,
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                logger.warning(f"记录并发任务指标失败: {e}")

        # 根据并发模式选择最佳策略
        if self.concurrency_mode == ConcurrencyMode.ASYNC and asyncio.iscoroutinefunction(func):
            # 异步模式
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return loop.create_task(self.submit_async_task(func(*args, **kwargs), priority))
                else:
                    # 如果没有运行中的事件循环，返回协程对象
                    return func(*args, **kwargs)
            except RuntimeError:
                # 没有事件循环，返回协程对象
                return func(*args, **kwargs)

        elif self.concurrency_mode == ConcurrencyMode.THREAD:
            # 线程模式
            return self.submit_thread_task(func, *args, priority=priority, **kwargs)

        elif self.concurrency_mode == ConcurrencyMode.HYBRID:
            # 混合模式：根据优先级选择
            if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                # 关键和高优先级任务使用异步或优先级调度器
                if asyncio.iscoroutinefunction(func):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            return loop.create_task(self.submit_async_task(func(*args, **kwargs), priority))
                        else:
                            return func(*args, **kwargs)
                    except RuntimeError:
                        return func(*args, **kwargs)
                else:
                    success = self.submit_priority_task(func, *args, priority=priority, **kwargs)
                    return success  # 返回布尔值表示是否成功提交
            else:
                # 普通和低优先级任务使用线程池
                return self.submit_thread_task(func, *args, priority=priority, **kwargs)

        else:
            # 默认使用线程模式
            return self.submit_thread_task(func, *args, priority=priority, **kwargs)

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """获取并发统计信息"""
        stats = {
            'mode': self.concurrency_mode.value,
            'async_stats': None,
            'thread_stats': None,
            'priority_stats': None
        }

        if self.async_manager:
            stats['async_stats'] = self.async_manager.get_stats()

        if self.thread_pool_manager:
            stats['thread_stats'] = self.thread_pool_manager.get_stats()

        if self.priority_scheduler:
            stats['priority_stats'] = self.priority_scheduler.get_stats()

        return stats

    def optimize_concurrency(self):
        """优化并发配置"""
        stats = self.get_concurrency_stats()

        # 根据统计信息调整配置
        if stats['async_stats']:
            async_stats = stats['async_stats']
            # 如果异步任务经常达到并发上限，增加并发数
        if async_stats.active_tasks >= self.max_async_concurrent * 0.9:
            logger.info("异步任务经常达到并发上限，考虑增加异步并发数")

        if stats['thread_stats']:
            thread_stats = stats['thread_stats']
            # 如果线程池利用率过高，增加线程数
        if thread_stats.thread_pool_utilization > 0.8:
            logger.info("线程池利用率过高，考虑增加线程数")

        logger.info("并发配置优化完成")

    def shutdown(self):
        """关闭并发管理器"""
        logger.info("正在关闭交易并发管理器...")

        if self.priority_scheduler:
            self.priority_scheduler.stop()

        if self.thread_pool_manager:
            self.thread_pool_manager.shutdown()

        logger.info("交易并发管理器已关闭")

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            'component': 'TradingConcurrencyManager',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'mode': self.concurrency_mode.value,
            'warnings': [],
            'critical_issues': []
        }

        stats = self.get_concurrency_stats()

        # 检查各个组件状态
        if self.async_manager and stats['async_stats']:
            async_stats = stats['async_stats']
        if async_stats.rejected_tasks > 10:
            health_info['warnings'].append("异步任务拒绝数量过多")

        if self.thread_pool_manager and stats['thread_stats']:
            thread_stats = stats['thread_stats']
        if thread_stats.failed_tasks > thread_stats.completed_tasks * 0.1:
            health_info['warnings'].append("线程任务失败率过高")

        if self.priority_scheduler and stats['priority_stats']:
            priority_stats = stats['priority_stats']
        if priority_stats.queue_length > self.max_priority_concurrent * 2:
            health_info['critical_issues'].append("优先级任务队列过长")

        # 总体状态评估
        if health_info['critical_issues']:
            health_info['status'] = 'critical'
        elif health_info['warnings']:
            health_info['status'] = 'warning'

        # 添加基础设施集成状态
        health_info['infrastructure_integration'] = INFRASTRUCTURE_INTEGRATION_AVAILABLE

        return health_info


# 全局并发管理器实例
_concurrency_manager = None
_concurrency_manager_lock = threading.Lock()


def get_concurrency_manager() -> TradingConcurrencyManager:
    """获取全局并发管理器实例"""
    global _concurrency_manager

    if _concurrency_manager is None:
        with _concurrency_manager_lock:
            if _concurrency_manager is None:
                _concurrency_manager = TradingConcurrencyManager()

    return _concurrency_manager


# 便捷函数
async def submit_async_trading_task(coro: Awaitable, priority: TaskPriority = TaskPriority.NORMAL):
    """提交异步交易任务"""
    manager = get_concurrency_manager()
    return await manager.submit_async_task(coro, priority)


def submit_thread_trading_task(func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs):
    """提交线程交易任务"""
    manager = get_concurrency_manager()
    return manager.submit_thread_task(func, *args, priority=priority, **kwargs)


def submit_priority_trading_task(func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs):
    """提交优先级交易任务"""
    manager = get_concurrency_manager()
    return manager.submit_priority_trading_task(func, *args, priority=priority, **kwargs)


def submit_smart_trading_task(func: Callable, *args, task_type: str = "normal", **kwargs):
    """智能提交交易任务"""
    manager = get_concurrency_manager()
    return manager.submit_trading_task(func, *args, task_type=task_type, **kwargs)
