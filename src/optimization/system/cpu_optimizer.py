"""
CPU Optimization Module
CPU优化模块

This module provides CPU optimization capabilities for quantitative trading systems
此模块为量化交易系统提供CPU优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import threading
import time
import psutil
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class CPUOptimizer:

    """
    CPU Optimizer Class
    CPU优化器类

    Provides CPU utilization optimization and workload distribution
    提供CPU利用率优化和工作负载分配
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize CPU optimizer
        初始化CPU优化器

        Args:
            max_workers: Maximum number of workers (None for auto - detection)
                        最大工作线程数（None表示自动检测）
        """
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() * 2)
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.cpu_monitoring = True
        self.monitoring_thread: Optional[threading.Thread] = None
        self.cpu_stats = []
        self.workload_queue = []
        self.is_running = False

        # CPU affinity settings
        self.cpu_affinity_enabled = True
        self.reserved_cores = 0  # Cores reserved for system processes

        logger.info(f"CPU optimizer initialized with {self.max_workers} max workers")

    def start_cpu_optimization(self) -> bool:
        """
        Start CPU optimization and monitoring
        开始CPU优化和监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("CPU optimizer already running")
            return False

        try:
            self.is_running = True

            # Create thread pools
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            self.process_pool = ProcessPoolExecutor(max_workers=min(8, multiprocessing.cpu_count()))

            # Start monitoring thread
            if self.cpu_monitoring:
                self.monitoring_thread = threading.Thread(
                    target=self._cpu_monitoring_loop, daemon=True)
                self.monitoring_thread.start()

            # Set CPU affinity if enabled
            if self.cpu_affinity_enabled:
                self._set_cpu_affinity()

            logger.info("CPU optimization started")
            return True

        except Exception as e:
            logger.error(f"Failed to start CPU optimization: {str(e)}")
            self.is_running = False
            return False

    def stop_cpu_optimization(self) -> bool:
        """
        Stop CPU optimization and monitoring
        停止CPU优化和监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("CPU optimizer not running")
            return False

        try:
            self.is_running = False

            # Shutdown thread pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            if self.process_pool:
                self.process_pool.shutdown(wait=True)

            # Wait for monitoring thread
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)

            logger.info("CPU optimization stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop CPU optimization: {str(e)}")
            return False

    def submit_cpu_task(self,

                        func: Callable,
                        *args,
                        priority: str = "normal",
                        use_process_pool: bool = False,
                        **kwargs) -> Any:
        """
        Submit a task for CPU execution
        提交任务以进行CPU执行

        Args:
            func: Function to execute
                 要执行的函数
            *args: Positional arguments
                  位置参数
            priority: Task priority ("low", "normal", "high")
                     任务优先级
            use_process_pool: Whether to use process pool instead of thread pool
                             是否使用进程池而不是线程池
            **kwargs: Keyword arguments
                     关键字参数

        Returns:
            Future object for the submitted task
            已提交任务的Future对象
        """
        if not self.is_running:
            raise RuntimeError("CPU optimizer is not running")

        # Choose appropriate pool
        pool = self.process_pool if use_process_pool else self.thread_pool

        if priority == "high":
            # Submit immediately for high priority
            return pool.submit(func, *args, **kwargs)
        else:
            # Add to workload queue for scheduling
            task_info = {
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'priority': priority,
                'submitted_at': datetime.now(),
                'pool': pool
            }
            self.workload_queue.append(task_info)

            # Process workload queue
            return self._process_workload_queue()

    def get_cpu_stats(self) -> Dict[str, Any]:
        """
        Get current CPU statistics
        获取当前CPU统计信息

        Returns:
            dict: CPU statistics
                  CPU统计信息
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_freq = psutil.cpu_freq(percpu=True) if psutil.cpu_freq() else None
            cpu_times = psutil.cpu_times_percent(interval=0.1)

            stats = {
                'timestamp': datetime.now(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_percent_overall': sum(cpu_percent) / len(cpu_percent),
                'cpu_percent_per_core': cpu_percent,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'cpu_freq_min': cpu_freq.min if cpu_freq else None,
                'cpu_freq_max': cpu_freq.max if cpu_freq else None,
                'cpu_times_user': cpu_times.user,
                'cpu_times_system': cpu_times.system,
                'cpu_times_idle': cpu_times.idle,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }

            # Store stats for trend analysis
            self.cpu_stats.append(stats)
            if len(self.cpu_stats) > 100:  # Keep last 100 measurements
                self.cpu_stats = self.cpu_stats[-100:]

            return stats

        except Exception as e:
            logger.error(f"Failed to get CPU stats: {str(e)}")
            return {'error': str(e)}

    def optimize_workload_distribution(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize workload distribution across CPU cores
        优化跨CPU核心的工作负载分配

        Args:
            tasks: List of tasks with their requirements
                  具有需求的任务列表

        Returns:
            dict: Optimized workload distribution
                  优化的工作负载分配
        """
        try:
            n_cores = psutil.cpu_count()
            available_cores = max(1, n_cores - self.reserved_cores)

            # Analyze task requirements
            task_weights = []
            for task in tasks:
                weight = task.get('cpu_weight', 1.0)  # Estimated CPU requirement
                task_weights.append(weight)

            # Simple load balancing algorithm
            total_weight = sum(task_weights)
            avg_weight_per_core = total_weight / available_cores

            distribution = {
                'total_cores': n_cores,
                'available_cores': available_cores,
                'total_tasks': len(tasks),
                'avg_weight_per_core': avg_weight_per_core,
                'task_distribution': {},
                'load_balance_score': 0.0
            }

            # Distribute tasks evenly
            core_loads = [0.0] * available_cores
            task_assignments = [[] for _ in range(available_cores)]

            for i, (task, weight) in enumerate(zip(tasks, task_weights)):
                # Find least loaded core
                min_load_idx = core_loads.index(min(core_loads))
                core_loads[min_load_idx] += weight
                task_assignments[min_load_idx].append(i)

            # Calculate load balance score (lower is better)
            load_variance = np.var(core_loads)
            distribution['load_balance_score'] = load_variance
            distribution['core_loads'] = core_loads
            distribution['task_assignments'] = task_assignments

            return distribution

        except Exception as e:
            logger.error(f"Failed to optimize workload distribution: {str(e)}")
            return {'error': str(e)}

    def set_cpu_affinity(self, core_ids: List[int]) -> bool:
        """
        Set CPU affinity for the current process
        为当前进程设置CPU亲和性

        Args:
            core_ids: List of CPU core IDs to use
                     要使用的CPU核心ID列表

        Returns:
            bool: True if set successfully, False otherwise
                  设置成功返回True，否则返回False
        """
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, set(core_ids))
                logger.info(f"CPU affinity set to cores: {core_ids}")
                return True
            else:
                logger.warning("CPU affinity not supported on this platform")
                return False
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {str(e)}")
            return False

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get CPU optimization recommendations
        获取CPU优化建议

        Returns:
            dict: Optimization recommendations
                  优化建议
        """
        try:
            current_stats = self.get_cpu_stats()

            recommendations = {
                'timestamp': datetime.now(),
                'current_load': current_stats.get('cpu_percent_overall', 0),
                'recommendations': []
            }

            cpu_load = current_stats.get('cpu_percent_overall', 0)

            if cpu_load > 90:
                recommendations['recommendations'].append({
                    'type': 'critical',
                    'message': 'CPU usage is critically high. Consider workload redistribution.',
                    'action': 'reduce_concurrent_tasks'
                })
            elif cpu_load > 70:
                recommendations['recommendations'].append({
                    'type': 'warning',
                    'message': 'CPU usage is high. Consider optimizing task scheduling.',
                    'action': 'optimize_scheduling'
                })
            elif cpu_load < 30:
                recommendations['recommendations'].append({
                    'type': 'info',
                    'message': 'CPU utilization is low. Consider increasing parallelism.',
                    'action': 'increase_parallelism'
                })

            # Analyze trends
            if len(self.cpu_stats) >= 10:
                recent_loads = [s.get('cpu_percent_overall', 0) for s in self.cpu_stats[-10:]]
                avg_load = sum(recent_loads) / len(recent_loads)

                if avg_load > 80:
                    recommendations['recommendations'].append({
                        'type': 'trend_analysis',
                        'message': 'Sustained high CPU usage detected. Consider system scaling.',
                        'action': 'consider_scaling'
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {str(e)}")
            return {'error': str(e)}

    def _set_cpu_affinity(self) -> None:
        """
        Set optimal CPU affinity for the process
        为进程设置最佳CPU亲和性
        """
        try:
            n_cores = psutil.cpu_count()
            if n_cores and n_cores > self.reserved_cores:
                # Use all cores except reserved ones
                available_cores = list(range(self.reserved_cores, n_cores))
                self.set_cpu_affinity(available_cores)
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {str(e)}")

    def _cpu_monitoring_loop(self) -> None:
        """
        CPU monitoring loop
        CPU监控循环
        """
        logger.info("CPU monitoring loop started")

        while self.is_running:
            try:
                # Get CPU stats
                stats = self.get_cpu_stats()

                # Log warnings for high CPU usage
                cpu_load = stats.get('cpu_percent_overall', 0)
                if cpu_load > 85:
                    logger.warning(f"High CPU usage detected: {cpu_load:.1f}%")

                # Process workload queue
                self._process_workload_queue()

                # Sleep before next monitoring cycle
                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"CPU monitoring loop error: {str(e)}")
                time.sleep(5)

        logger.info("CPU monitoring loop stopped")

    def _process_workload_queue(self) -> Any:
        """
        Process pending tasks in the workload queue
        处理工作负载队列中的待处理任务

        Returns:
            Future object for the submitted task
            已提交任务的Future对象
        """
        if not self.workload_queue:
            return None

        # Sort by priority (high priority first)
        priority_order = {'high': 0, 'normal': 1, 'low': 2}
        self.workload_queue.sort(key=lambda x: priority_order.get(x['priority'], 1))

        # Submit highest priority task
        task_info = self.workload_queue.pop(0)
        future = task_info['pool'].submit(
            task_info['func'],
            *task_info['args'],
            **task_info['kwargs']
        )

        return future

    def get_optimizer_status(self) -> Dict[str, Any]:
        """
        Get CPU optimizer status
        获取CPU优化器状态

        Returns:
            dict: Optimizer status information
                  优化器状态信息
        """
        return {
            'is_running': self.is_running,
            'max_workers': self.max_workers,
            'cpu_monitoring': self.cpu_monitoring,
            'cpu_affinity_enabled': self.cpu_affinity_enabled,
            'reserved_cores': self.reserved_cores,
            'workload_queue_size': len(self.workload_queue),
            'cpu_stats_count': len(self.cpu_stats),
            'current_cpu_stats': self.get_cpu_stats(),
            'optimization_recommendations': self.get_optimization_recommendations()
        }


# Global CPU optimizer instance
# 全局CPU优化器实例
cpu_optimizer = CPUOptimizer()

__all__ = ['CPUOptimizer', 'cpu_optimizer']
