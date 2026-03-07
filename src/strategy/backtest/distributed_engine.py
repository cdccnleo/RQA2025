#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
分布式回测引擎

基于设计文档实现的分布式回测系统，支持大规模策略并行回测。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from dataclasses import dataclass, field
import multiprocessing as mp
import queue
import threading
import time
import json
import pickle
from pathlib import Path
import uuid
import psutil

logger = logging.getLogger(__name__)


@dataclass
class BacktestTask:

    """回测任务数据结构"""
    task_id: str
    strategy_config: Dict
    data_config: Dict
    backtest_config: Dict
    priority: int = 0
    timeout: int = 3600
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None

    def __lt__(self, other):
        """支持PriorityQueue比较"""
        if self.priority != other.priority:
            return self.priority < other.priority
        # 如果优先级相同，按创建时间排序
        return self.created_at < other.created_at


@dataclass
class BacktestResult:

    """回测结果数据结构"""
    task_id: str
    strategy_name: str
    performance_metrics: Dict[str, float]
    trade_history: List[Dict]
    portfolio_values: List[float]
    benchmark_values: List[float]
    execution_time: float
    memory_usage: float
    created_at: datetime = field(default_factory=datetime.now)


class TaskScheduler:

    """智能任务调度器 - 优化版本"""

    def __init__(self, max_workers: int = None):

        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.task_queue = queue.PriorityQueue()
        self.worker_pool = []
        self.resource_monitor = ResourceMonitor()
        self.running = False
        self.lock = threading.Lock()

        # 负载均衡优化
        self.worker_loads = {}  # 记录每个worker的负载
        self.task_history = {}  # 任务执行历史
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'average_wait_time': 0.0
        }

        # 初始化worker池
        self._init_worker_pool()

    def _init_worker_pool(self):
        """初始化worker池"""
        for i in range(self.max_workers):
            worker = DistributedBacktestWorker(f"worker_{i}", {})
            self.worker_pool.append(worker)
            self.worker_loads[f"worker_{i}"] = {
                'current_tasks': 0,
                'total_tasks': 0,
                'average_execution_time': 0.0,
                'last_available_time': time.time()
            }

    def submit_task(self, task: BacktestTask) -> str:
        """提交任务"""
        with self.lock:
            self.task_queue.put((task.priority, task))
            logger.info(f"Task {task.task_id} submitted with priority {task.priority}")
            return task.task_id

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        # 这里应该从数据库或缓存中获取任务状态
        # 简化实现，返回基本信息
        return {
            "task_id": task_id,
            "status": "pending",  # 实际应该从存储中获取
            "created_at": datetime.now().isoformat()
        }

    def start(self):
        """启动调度器"""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")

    def stop(self):
        """停止调度器"""
        self.running = False
        if hasattr(self, 'scheduler_thread'):
            self.scheduler_thread.join()
        logger.info("Task scheduler stopped")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                if not self.task_queue.empty():
                    priority, task = self.task_queue.get(timeout=1)
                    available_worker = self._find_available_worker(task)

                    if available_worker:
                        self._assign_task(task, available_worker)
                    else:
                        # 如果没有可用worker，重新放回队列
                        self.task_queue.put((priority, task))
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)

    def _find_available_worker(self, task: BacktestTask) -> Optional['DistributedBacktestWorker']:
        """智能查找可用worker - 负载均衡优化"""
        available_workers = []

        for worker in self.worker_pool:
            if worker.is_available():
                worker_id = worker.worker_id
                load_info = self.worker_loads.get(worker_id, {})

                # 计算worker的负载分数（越低越好）
                current_load = load_info.get('current_tasks', 0)
                avg_exec_time = load_info.get('average_execution_time', 0.0)

                # 负载分数 = 当前任务数 * 0.6 + 平均执行时间 * 0.4
                load_score = current_load * 0.6 + avg_exec_time * 0.4

                available_workers.append((worker, load_score))

        if not available_workers:
            return None

        # 选择负载最低的worker
        best_worker = min(available_workers, key=lambda x: x[1])[0]

        # 更新worker负载信息
        worker_id = best_worker.worker_id
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id]['current_tasks'] += 1
            self.worker_loads[worker_id]['total_tasks'] += 1

        return best_worker

    def _assign_task(self, task: BacktestTask, worker: 'DistributedBacktestWorker'):
        """分配任务给worker"""
        worker.execute_task_async(task)


class DistributedBacktestWorker:

    """分布式回测Worker"""

    def __init__(self, worker_id: str, config: Dict):

        self.worker_id = worker_id
        self.config = config
        self.task_executor = TaskExecutor()
        self.data_cache = DataCache()
        self.resource_monitor = ResourceMonitor()
        self.current_task = None
        self.is_busy = False
        self.lock = threading.Lock()

    def is_available(self) -> bool:
        """检查worker是否可用"""
        with self.lock:
            return not self.is_busy and self.resource_monitor.check_resources()

    def execute_task_async(self, task: BacktestTask):
        """异步执行任务"""
        with self.lock:
            if self.is_busy:
                return False

            self.is_busy = True
            self.current_task = task

        # 在新线程中执行任务
        thread = threading.Thread(target=self._execute_task, args=(task,))
        thread.start()
        return True

    def _execute_task(self, task: BacktestTask):
        """执行任务 - 优化版本"""
        start_time = time.time()
        execution_time = 0.0

        try:
            logger.info(f"Worker {self.worker_id} starting task {task.task_id}")

            # 更新任务状态
            task.status = "running"

            # 执行回测
            result = self.task_executor.run_backtest(task)

            # 更新任务结果
            task.result = result
            task.status = "completed"

            execution_time = time.time() - start_time
            logger.info(
                f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Worker {self.worker_id} failed task {task.task_id} after {execution_time:.2f}s: {e}")
            task.error = str(e)
            task.status = "failed"

        finally:
            with self.lock:
                self.is_busy = False
                self.current_task = None

            # 更新性能统计
            self._update_performance_stats(execution_time, task.status == "completed")

    def _update_performance_stats(self, execution_time: float, success: bool):
        """更新性能统计"""
        # 这里可以添加更详细的性能统计
        if success:
            logger.debug(
                f"Worker {self.worker_id} task completed successfully in {execution_time:.2f}s")
        else:
            logger.debug(f"Worker {self.worker_id} task failed after {execution_time:.2f}s")


class TaskExecutor:

    """任务执行器"""

    def __init__(self):

        self.backtest_engine = None  # 将在需要时初始化

    def run_backtest(self, task: BacktestTask) -> BacktestResult:
        """执行回测任务"""
        start_time = time.time()

        try:
            # 1. 准备数据
            data = self._prepare_data(task.data_config)

            # 2. 加载策略
            strategy = self._load_strategy(task.strategy_config)

            # 3. 执行回测
            if self.backtest_engine is None:
                from .engine import OptimizedBacktestEngine
                self.backtest_engine = OptimizedBacktestEngine()

            # 将task.backtest_config转换为BacktestConfig对象
            from .engine import BacktestConfig
            backtest_config = BacktestConfig(
                initial_capital=task.backtest_config.get('initial_capital', 1000000.0),
                commission_rate=task.backtest_config.get('commission', 0.0003),
                slippage_rate=task.backtest_config.get('slippage', 0.0001),
                benchmark=task.backtest_config.get('benchmark'),
                risk_free_rate=task.backtest_config.get('risk_free_rate', 0.03),
                max_workers=task.backtest_config.get('max_workers'),
                enable_cache=task.backtest_config.get('enable_cache', True),
                cache_dir=task.backtest_config.get('cache_dir', "cache / backtest_results"),
                memory_limit_gb=task.backtest_config.get('memory_limit_gb', 4.0),
                enable_parallel=task.backtest_config.get('enable_parallel', True)
            )

            result = self.backtest_engine.run_backtest(
                strategy, data, backtest_config
            )

            # 4. 构建结果
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            backtest_result = BacktestResult(
                task_id=task.task_id,
                strategy_name=task.strategy_config.get('name', 'unknown'),
                performance_metrics=result.get('metrics', {}),
                trade_history=result.get('trades', []),
                portfolio_values=result.get('portfolio_values', []),
                benchmark_values=result.get('benchmark_values', []),
                execution_time=execution_time,
                memory_usage=memory_usage
            )

            return backtest_result

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            raise

    def _prepare_data(self, data_config: Dict) -> Dict[str, pd.DataFrame]:
        """准备数据"""
        # 这里应该根据data_config从数据源加载数据
        # 简化实现，返回模拟数据
        symbols = data_config.get('symbols', ['000001.SZ'])
        start_date = data_config.get('start_date', '2023 - 01 - 01')
        end_date = data_config.get('end_date', '2023 - 12 - 31')

        data = {}
        for symbol in symbols:
            # 生成模拟数据
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data[symbol] = pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum() + 100,
                'high': np.random.randn(len(dates)).cumsum() + 102,
                'low': np.random.randn(len(dates)).cumsum() + 98,
                'close': np.random.randn(len(dates)).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        return data

    def _load_strategy(self, strategy_config: Dict) -> Callable:
        """加载策略"""
        # 这里应该根据strategy_config加载策略函数
        # 简化实现，返回一个简单的策略

        def simple_strategy(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
            """简单策略：买入所有股票"""
            signals = {}
            for symbol in data.keys():
                signals[symbol] = 1.0  # 买入信号
            return signals

        return simple_strategy


class DataCache:

    """数据缓存管理器"""

    def __init__(self, cache_dir: str = "cache / distributed_data"):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache {key}: {e}")
        return None

    def set(self, key: str, data: Any):
        """设置缓存数据"""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache {key}: {e}")


class ResourceMonitor:

    """资源监控器"""

    def __init__(self, memory_limit_gb: float = 4.0, cpu_limit_percent: float = 80.0):

        self.memory_limit_gb = memory_limit_gb
        self.cpu_limit_percent = cpu_limit_percent

    def check_resources(self) -> bool:
        """检查资源是否可用"""
        try:
            # 检查内存使用
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 90:  # 内存使用超过90%
                return False

            # 检查CPU使用
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > self.cpu_limit_percent:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return True  # 出错时默认可用

    def get_resource_stats(self) -> Dict[str, float]:
        """获取资源统计"""
        try:
            memory = psutil.virtual_memory()
            return {
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'cpu_usage_percent': psutil.cpu_percent(interval=0.1)
            }
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}


class DistributedBacktestEngine:

    """分布式回测引擎主类"""

    def __init__(self, config: Dict = None):

        self.config = config or {}
        self.scheduler = TaskScheduler(
            max_workers=self.config.get('max_workers', None)
        )
        self.result_store = ResultStore()
        self.monitor = SystemMonitor()

        # 启动调度器
        self.scheduler.start()

    def submit_backtest(self, strategy_config: Dict, data_config: Dict,


                        backtest_config: Dict, priority: int = 0) -> str:
        """提交回测任务"""
        task_id = str(uuid.uuid4())

        task = BacktestTask(
            task_id=task_id,
            strategy_config=strategy_config,
            data_config=data_config,
            backtest_config=backtest_config,
            priority=priority
        )

        self.scheduler.submit_task(task)
        return task_id

    def get_task_status(self, task_id: str) -> Dict:
        """获取任务状态"""
        return self.scheduler.get_task_status(task_id)

    def get_task_result(self, task_id: str) -> Optional[BacktestResult]:
        """获取任务结果"""
        return self.result_store.get_result(task_id)

    def get_system_stats(self) -> Dict:
        """获取系统统计"""
        return self.monitor.get_stats()

    def shutdown(self):
        """关闭分布式引擎"""
        self.scheduler.stop()
        logger.info("Distributed backtest engine shutdown")


class ResultStore:

    """结果存储管理器"""

    def __init__(self, store_dir: str = "cache / distributed_results"):

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, task_id: str, result: BacktestResult):
        """保存结果"""
        result_file = self.store_dir / f"{task_id}.json"
        try:
            # 将结果转换为可序列化的格式
            result_dict = {
                'task_id': result.task_id,
                'strategy_name': result.strategy_name,
                'performance_metrics': result.performance_metrics,
                'trade_history': result.trade_history,
                'portfolio_values': result.portfolio_values,
                'benchmark_values': result.benchmark_values,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'created_at': result.created_at.isoformat()
            }

            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving result for task {task_id}: {e}")

    def get_result(self, task_id: str) -> Optional[BacktestResult]:
        """获取结果"""
        result_file = self.store_dir / f"{task_id}.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result_dict = json.load(f)

                return BacktestResult(
                    task_id=result_dict['task_id'],
                    strategy_name=result_dict['strategy_name'],
                    performance_metrics=result_dict['performance_metrics'],
                    trade_history=result_dict['trade_history'],
                    portfolio_values=result_dict['portfolio_values'],
                    benchmark_values=result_dict['benchmark_values'],
                    execution_time=result_dict['execution_time'],
                    memory_usage=result_dict['memory_usage'],
                    created_at=datetime.fromisoformat(result_dict['created_at'])
                )

            except Exception as e:
                logger.error(f"Error loading result for task {task_id}: {e}")

        return None


class SystemMonitor:

    """系统监控器"""

    def __init__(self):

        self.start_time = datetime.now()

    def get_stats(self) -> Dict:
        """获取系统统计"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
