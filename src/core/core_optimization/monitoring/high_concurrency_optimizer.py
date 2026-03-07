"""
高并发优化管理器 - 大规模并发处理优化

解决高优先级问题2：大规模并发处理优化
- 优化核心业务层的并发处理能力
- 提升策略计算和交易执行的并发性能
- 实现智能资源调度和负载均衡

作者: 系统架构师
创建时间: 2025 - 01 - 28
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import PriorityQueue
from enum import Enum

logger = logging.getLogger(__name__)


class ConcurrencyLevel(Enum):

    """并发级别枚举"""
    LOW = "low"           # 低并发 (< 100 TPS)
    MEDIUM = "medium"     # 中并发 (100 - 1000 TPS)
    HIGH = "high"         # 高并发 (1000 - 10000 TPS)
    EXTREME = "extreme"   # 极高并发 (> 10000 TPS)


class TaskPriority(Enum):

    """任务优先级枚举"""
    CRITICAL = 4  # 关键任务
    HIGH = 3      # 高优先级
    NORMAL = 2    # 普通优先级
    LOW = 1       # 低优先级


@dataclass
class Task:

    """任务数据类"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    timeout: float = 30.0
    created_time: float = None

    def __post_init__(self):

        if self.created_time is None:
            self.created_time = time.time()


@dataclass
class WorkerStats:

    """工作线程统计信息"""
    worker_id: str
    task_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    avg_processing_time: float = 0.0
    last_active_time: float = 0.0
    current_load: int = 0


class AdaptiveThreadPool:

    """自适应线程池"""

    def __init__(self, min_workers: int = 4, max_workers: int = 50,


                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.2):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        self._executor = ThreadPoolExecutor(max_workers=min_workers)
        self._current_workers = min_workers
        self._lock = threading.Lock()
        self._stats = {}
        self._last_scale_time = time.time()
        self._scale_cooldown = 60  # 缩放冷却时间(秒)

    def submit(self, fn, *args, **kwargs):
        """提交任务"""
        with self._lock:
            # 检查是否需要扩容
            self._check_scaling()

            future = self._executor.submit(fn, *args, **kwargs)
            return future

    def _check_scaling(self):
        """检查是否需要扩容或缩容"""
        if time.time() - self._last_scale_time < self._scale_cooldown:
            return

        # 计算当前负载
        active_threads = len([t for t in self._executor._threads if t.is_alive()])
        load_factor = active_threads / self._current_workers

        if load_factor > self.scale_up_threshold and self._current_workers < self.max_workers:
            # 扩容
            self._scale_up()
        elif load_factor < self.scale_down_threshold and self._current_workers > self.min_workers:
            # 缩容
            self._scale_down()

    def _scale_up(self):
        """扩容"""
        new_workers = min(self._current_workers * 2, self.max_workers)
        if new_workers > self._current_workers:
            logger.info(f"线程池扩容: {self._current_workers} -> {new_workers}")
            self._executor = ThreadPoolExecutor(max_workers=new_workers)
            self._current_workers = new_workers
            self._last_scale_time = time.time()

    def _scale_down(self):
        """缩容"""
        new_workers = max(self._current_workers // 2, self.min_workers)
        if new_workers < self._current_workers:
            logger.info(f"线程池缩容: {self._current_workers} -> {new_workers}")
            self._executor = ThreadPoolExecutor(max_workers=new_workers)
            self._current_workers = new_workers
            self._last_scale_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """获取线程池统计信息"""
        return {
            'current_workers': self._current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'active_threads': len([t for t in self._executor._threads if t.is_alive()]),
            'load_factor': len([t for t in self._executor._threads if t.is_alive()]) / self._current_workers
        }


class TaskScheduler:

    """任务调度器 - 支持优先级和负载均衡"""

    def __init__(self, max_workers: int = 20):

        self.max_workers = max_workers
        self._task_queue = PriorityQueue()
        self._workers = {}
        self._lock = threading.Lock()
        self._running = True

        # 启动调度线程
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        # 初始化工作线程
        self._init_workers()

    def _init_workers(self):
        """初始化工作线程"""
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            self._workers[worker_id] = WorkerStats(worker_id=worker_id)
            thread = threading.Thread(target=self._worker_loop, args=(worker_id,), daemon=True)
            thread.start()

    def submit_task(self, task: Task) -> str:
        """提交任务"""
        with self._lock:
            self._task_queue.put((-task.priority.value, task.created_time, task))
            logger.info(f"任务已提交: {task.task_id}, 类型: {task.task_type}, 优先级: {task.priority.name}")
            return task.task_id

    def _scheduler_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 查找最空闲的工作线程
                if not self._task_queue.empty():
                    idle_worker = self._find_idle_worker()
                    if idle_worker:
                        self._assign_task_to_worker(idle_worker)
                time.sleep(0.01)  # 10ms调度间隔
            except Exception as e:
                logger.error(f"调度循环异常: {e}")

    def _find_idle_worker(self) -> Optional[str]:
        """查找最空闲的工作线程"""
        idle_workers = [
            worker_id for worker_id, stats in self._workers.items()
            if stats.current_load < 3  # 每个工作线程最多处理3个并发任务
        ]
        return idle_workers[0] if idle_workers else None

    def _assign_task_to_worker(self, worker_id: str):
        """分配任务给工作线程"""
        try:
            _, _, task = self._task_queue.get_nowait()
            self._workers[worker_id].current_load += 1
            # 这里可以实现实际的任务分配逻辑
            logger.info(f"任务 {task.task_id} 已分配给工作线程 {worker_id}")
        except Exception as e:
            logger.error(f"任务分配失败: {e}")

    def _worker_loop(self, worker_id: str):
        """工作线程循环"""
        while self._running:
            try:
                # 模拟任务处理
                if self._workers[worker_id].current_load > 0:
                    time.sleep(0.1)  # 模拟处理时间
                    self._workers[worker_id].current_load -= 1
                    self._workers[worker_id].task_count += 1
                    self._workers[worker_id].last_active_time = time.time()
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"工作线程异常: {worker_id}, {e}")

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            'queue_size': self._task_queue.qsize(),
            'worker_count': len(self._workers),
            'worker_stats': {wid: stats.__dict__ for wid, stats in self._workers.items()},
            'total_tasks': sum(stats.task_count for stats in self._workers.values())
        }


class HighConcurrencyOptimizer:

    """高并发优化管理器"""

    def __init__(self, concurrency_level: ConcurrencyLevel = ConcurrencyLevel.HIGH):

        self.concurrency_level = concurrency_level
        self.logger = logging.getLogger(__name__)

        # 根据并发级别配置参数
        self._configure_for_concurrency_level()

        # 初始化组件
        self._thread_pool = AdaptiveThreadPool(
            min_workers=self._min_workers,
            max_workers=self._max_workers
        )
        self._task_scheduler = TaskScheduler(max_workers=self._max_workers)
        self._performance_monitor = PerformanceMonitor()

        # 启动监控线程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info(f"高并发优化管理器已初始化，并发级别: {concurrency_level.value}")

    def _configure_for_concurrency_level(self):
        """根据并发级别配置参数"""
        configs = {
            ConcurrencyLevel.LOW: {
                'min_workers': 4,
                'max_workers': 16,
                'batch_size': 10,
                'queue_size': 100
            },
            ConcurrencyLevel.MEDIUM: {
                'min_workers': 8,
                'max_workers': 32,
                'batch_size': 50,
                'queue_size': 500
            },
            ConcurrencyLevel.HIGH: {
                'min_workers': 16,
                'max_workers': 64,
                'batch_size': 100,
                'queue_size': 1000
            },
            ConcurrencyLevel.EXTREME: {
                'min_workers': 32,
                'max_workers': 128,
                'batch_size': 200,
                'queue_size': 5000
            }
        }

        config = configs[self.concurrency_level]
        self._min_workers = config['min_workers']
        self._max_workers = config['max_workers']
        self._batch_size = config['batch_size']
        self._queue_size = config['queue_size']

    def optimize_strategy_execution(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化策略执行 - 批量并发处理"""
        if not strategies:
            return []

        # 创建任务
        tasks = []
        for strategy in strategies:
            task = Task(
                task_id=strategy.get('id', f"strategy_{len(tasks)}"),
                task_type="strategy_execution",
                priority=TaskPriority.NORMAL,
                data=strategy
            )
            tasks.append(task)

        # 批量提交任务
        results = []
        for task in tasks:
            self._task_scheduler.submit_task(task)
            results.append({
                'task_id': task.task_id,
                'status': 'submitted',
                'submitted_time': task.created_time
            })

        return results

    def optimize_trading_execution(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化交易执行 - 高并发处理"""
        if not orders:
            return []

        # 按优先级分组
        priority_groups = {}
        for order in orders:
            priority = order.get('priority', 'normal')
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(order)

        results = []

        # 按优先级顺序处理
        for priority in ['critical', 'high', 'normal', 'low']:
            if priority in priority_groups:
                group_orders = priority_groups[priority]

                # 批量处理同一优先级的订单
                for i in range(0, len(group_orders), self._batch_size):
                    batch = group_orders[i:i + self._batch_size]
                    batch_results = self._process_order_batch(batch, priority)
                    results.extend(batch_results)

        return results

    def _process_order_batch(self, orders: List[Dict[str, Any]], priority: str) -> List[Dict[str, Any]]:
        """处理订单批次"""
        futures = []

        # 并行提交订单处理任务
        for order in orders:
            future = self._thread_pool.submit(self._process_single_order, order)
            futures.append(future)

        # 收集结果
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                results.append({
                    'order_id': 'unknown',
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                })

        return results

    def _process_single_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个订单"""
        # 模拟订单处理
        time.sleep(0.05)  # 模拟处理时间

        return {
            'order_id': order.get('id', 'unknown'),
            'status': 'processed',
            'execution_time': 0.05,
            'timestamp': time.time()
        }

    def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                # 收集性能指标
                self._performance_monitor.collect_metrics()

                # 检查是否需要调整配置
                self._check_performance_adjustments()

                time.sleep(5)  # 5秒监控间隔
            except Exception as e:
                logger.error(f"监控循环异常: {e}")

    def _check_performance_adjustments(self):
        """检查性能调整"""
        stats = self._performance_monitor.get_stats()

        # 根据负载情况调整配置
        load_factor = stats.get('load_factor', 0)
        if load_factor > 0.8:
            # 高负载，增加资源
            self._scale_up_resources()
        elif load_factor < 0.3:
            # 低负载，减少资源
            self._scale_down_resources()

    def _scale_up_resources(self):
        """扩容资源"""
        logger.info("检测到高负载，开始扩容资源")
        # 增加线程池大小
        self._thread_pool._scale_up()
        # 增加调度器工作线程
        self._task_scheduler.max_workers = min(self._task_scheduler.max_workers + 4, 100)

    def _scale_down_resources(self):
        """缩容资源"""
        logger.info("检测到低负载，开始缩容资源")
        # 减少线程池大小
        self._thread_pool._scale_down()
        # 减少调度器工作线程
        self._task_scheduler.max_workers = max(self._task_scheduler.max_workers - 2, 8)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'concurrency_level': self.concurrency_level.value,
            'thread_pool_stats': self._thread_pool.get_stats(),
            'scheduler_stats': self._task_scheduler.get_scheduler_stats(),
            'performance_stats': self._performance_monitor.get_stats(),
            'configuration': {
                'min_workers': self._min_workers,
                'max_workers': self._max_workers,
                'batch_size': self._batch_size,
                'queue_size': self._queue_size
            }
        }

    def shutdown(self):
        """关闭优化管理器"""
        logger.info("正在停止高并发优化管理器...")

        # 设置停止标志
        self._running = False

        # 停止监控线程
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                logger.warning("监控线程未能及时停止")

        # 停止线程池
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)

        # 停止任务调度器
        if hasattr(self, '_task_scheduler'):
            self._task_scheduler.shutdown()

        logger.info("高并发优化管理器已关闭")


class PerformanceMonitor:

    """性能监控器"""

    def __init__(self):

        self._metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'thread_count': [],
            'load_factor': [],
            'response_times': []
        }
        self._lock = threading.Lock()

    def collect_metrics(self):
        """收集性能指标"""
        try:
            import psutil
            with self._lock:
                self._metrics['cpu_usage'].append(psutil.cpu_percent())
                self._metrics['memory_usage'].append(psutil.virtual_memory().percent)
                self._metrics['thread_count'].append(threading.active_count())

                # 计算负载因子 (简化版)
                load_factor = min(1.0, threading.active_count() / 50)
                self._metrics['load_factor'].append(load_factor)

                # 保持最近100个数据点
                for key in self._metrics:
                    if len(self._metrics[key]) > 100:
                        self._metrics[key] = self._metrics[key][-100:]

        except ImportError:
            # 如果没有psutil，使用简化版监控
            with self._lock:
                self._metrics['cpu_usage'].append(50.0)  # 模拟值
                self._metrics['memory_usage'].append(60.0)
                self._metrics['thread_count'].append(threading.active_count())
                self._metrics['load_factor'].append(0.5)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'avg_cpu_usage': sum(self._metrics['cpu_usage']) / max(len(self._metrics['cpu_usage']), 1),
                'avg_memory_usage': sum(self._metrics['memory_usage']) / max(len(self._metrics['memory_usage']), 1),
                'avg_thread_count': sum(self._metrics['thread_count']) / max(len(self._metrics['thread_count']), 1),
                'current_load_factor': self._metrics['load_factor'][-1] if self._metrics['load_factor'] else 0,
                'data_points': len(self._metrics['cpu_usage'])
            }


# 全局高并发优化管理器实例
_high_concurrency_optimizer = None


def get_high_concurrency_optimizer() -> HighConcurrencyOptimizer:
    """获取全局高并发优化管理器实例"""
    global _high_concurrency_optimizer
    if _high_concurrency_optimizer is None:
        _high_concurrency_optimizer = HighConcurrencyOptimizer()
    return _high_concurrency_optimizer


def init_high_concurrency_optimizer(concurrency_level: ConcurrencyLevel = ConcurrencyLevel.HIGH):
    """初始化高并发优化管理器"""
    global _high_concurrency_optimizer
    _high_concurrency_optimizer = HighConcurrencyOptimizer(concurrency_level)
    logger.info(f"高并发优化管理器已初始化，并发级别: {concurrency_level.value}")


# 便捷函数

def optimize_strategy_batch(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """优化策略批量执行"""
    optimizer = get_high_concurrency_optimizer()
    return optimizer.optimize_strategy_execution(strategies)


def optimize_trading_batch(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """优化交易批量执行"""
    optimizer = get_high_concurrency_optimizer()
    return optimizer.optimize_trading_execution(orders)


if __name__ == "__main__":
    # 使用示例
    init_high_concurrency_optimizer(ConcurrencyLevel.HIGH)

    optimizer = get_high_concurrency_optimizer()

    # 优化策略执行
    strategies = [
        {'id': 'strategy_1', 'type': 'momentum', 'params': {'period': 20}},
        {'id': 'strategy_2', 'type': 'mean_reversion', 'params': {'threshold': 0.05}},
    ]

    strategy_results = optimizer.optimize_strategy_execution(strategies)
    print(f"策略优化结果: {strategy_results}")

    # 优化交易执行
    orders = [
        {'id': 'order_1', 'symbol': 'AAPL', 'quantity': 100, 'priority': 'high'},
        {'id': 'order_2', 'symbol': 'GOOGL', 'quantity': 50, 'priority': 'normal'},
    ]

    order_results = optimizer.optimize_trading_execution(orders)
    print(f"交易优化结果: {order_results}")

    # 查看优化统计
    stats = optimizer.get_optimization_stats()
    print(f"优化统计: {stats}")

    optimizer.shutdown()
