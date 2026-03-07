#!/usr/bin/env python3
"""
RQA2025 基础设施层并发处理优化器

提供智能的并发处理优化，包括线程池管理、异步任务调度和资源分配优化。
这是Phase 3高级功能实现的一部分。
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime
import psutil
import os

logger = logging.getLogger(__name__)


class ConcurrencyOptimizer:
    """
    并发处理优化器

    智能管理并发资源，根据系统负载动态调整线程池大小和任务调度策略。
    """

    def __init__(self, min_workers: int = 2, max_workers: int = 16,
                 target_cpu_percent: float = 70.0, monitor_interval: float = 5.0):
        """
        初始化并发优化器

        Args:
            min_workers: 最小工作线程数
            max_workers: 最大工作线程数
            target_cpu_percent: 目标CPU使用率
            monitor_interval: 监控间隔（秒）
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_percent = target_cpu_percent
        self.monitor_interval = monitor_interval

        # 线程池管理
        self.current_workers = min_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ConcurrencyOptimizer")

        # 监控和调整
        self.monitoring_active = False
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # 性能统计
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0,
            'current_queue_size': 0,
            'cpu_usage_history': [],
            'thread_adjustments': 0,
            'start_time': datetime.now()
        }

        # 任务优先级队列
        self.task_queues = {
            'high': [],
            'normal': [],
            'low': []
        }
        self.queue_lock = threading.RLock()

        logger.info(f"并发优化器初始化完成，线程范围: {min_workers}-{max_workers}")

    def submit_task(self, func: Callable, *args, priority: str = 'normal',
                   timeout: Optional[float] = None, **kwargs) -> Future:
        """
        提交任务执行

        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 任务优先级 ('high', 'normal', 'low')
            timeout: 超时时间
            **kwargs: 关键字参数

        Returns:
            Future: 任务Future对象
        """
        start_time = time.time()

        def wrapped_func():
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 更新统计
                self.stats['tasks_completed'] += 1
                self._update_avg_execution_time(execution_time)

                return result
            except Exception as e:
                self.stats['tasks_failed'] += 1
                logger.error(f"任务执行失败: {e}")
                raise

        # 提交到线程池
        future = self.executor.submit(wrapped_func)
        self.stats['tasks_submitted'] += 1

        # 设置超时
        if timeout:
            future.timeout = timeout

        logger.debug(f"提交任务: {func.__name__}, 优先级: {priority}")
        return future

    def submit_batch(self, tasks: List[tuple], priority: str = 'normal') -> List[Future]:
        """
        批量提交任务

        Args:
            tasks: 任务列表，每个元素是 (func, args, kwargs) 元组
            priority: 任务优先级

        Returns:
            List[Future]: Future对象列表
        """
        futures = []

        for task in tasks:
            if len(task) == 3:
                func, args, kwargs = task
            elif len(task) == 2:
                func, args = task
                kwargs = {}
            else:
                logger.error(f"无效任务格式: {task}")
                continue

            future = self.submit_task(func, *args, priority=priority, **kwargs)
            futures.append(future)

        logger.info(f"批量提交任务: {len(futures)} 个")
        return futures

    def wait_for_completion(self, futures: List[Future], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        等待任务完成

        Args:
            futures: Future对象列表
            timeout: 总超时时间

        Returns:
            Dict[str, Any]: 完成结果
        """
        results = []
        completed = 0
        failed = 0
        start_time = time.time()

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                except Exception as e:
                    results.append({'error': str(e)})
                    failed += 1
                    logger.error(f"任务执行异常: {e}")

        except Exception as e:
            logger.error(f"等待任务完成超时或异常: {e}")

        elapsed_time = time.time() - start_time

        return {
            'total': len(futures),
            'completed': completed,
            'failed': failed,
            'results': results,
            'elapsed_time': elapsed_time,
            'success_rate': completed / max(len(futures), 1)
        }

    def start_monitoring(self):
        """启动性能监控和自动调整"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_and_adjust,
            name="ConcurrencyMonitor",
            daemon=True
        )
        self.monitor_thread.start()

        logger.info("并发优化器监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self.stop_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("并发优化器监控已停止")

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """
        获取并发统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        current_time = datetime.now()
        uptime = (current_time - self.stats['start_time']).total_seconds()

        # 获取当前系统状态
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        return {
            'current_workers': self.current_workers,
            'tasks_submitted': self.stats['tasks_submitted'],
            'tasks_completed': self.stats['tasks_completed'],
            'tasks_failed': self.stats['tasks_failed'],
            'success_rate': self.stats['tasks_completed'] / max(self.stats['tasks_submitted'], 1),
            'avg_execution_time': self.stats['avg_execution_time'],
            'current_cpu_percent': cpu_percent,
            'current_memory_percent': memory_percent,
            'thread_adjustments': self.stats['thread_adjustments'],
            'uptime_seconds': uptime,
            'tasks_per_second': self.stats['tasks_completed'] / max(uptime, 1),
            'monitoring_active': self.monitoring_active
        }

    def _monitor_and_adjust(self):
        """监控系统性能并自动调整并发度"""
        logger.info("开始并发性能监控")

        while not self.stop_event.is_set():
            try:
                # 获取当前性能指标
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                # 记录历史数据
                self.stats['cpu_usage_history'].append(cpu_percent)
                if len(self.stats['cpu_usage_history']) > 10:  # 只保留最近10个数据点
                    self.stats['cpu_usage_history'].pop(0)

                # 计算平均CPU使用率
                avg_cpu = sum(self.stats['cpu_usage_history']) / len(self.stats['cpu_usage_history'])

                # 动态调整线程数
                self._adjust_thread_pool(avg_cpu, memory_percent)

                # 等待下次检查
                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"并发监控异常: {e}")
                time.sleep(self.monitor_interval)

        logger.info("并发性能监控已停止")

    def _adjust_thread_pool(self, avg_cpu: float, memory_percent: float):
        """
        根据性能指标调整线程池大小

        Args:
            avg_cpu: 平均CPU使用率
            memory_percent: 内存使用率
        """
        old_workers = self.current_workers
        new_workers = old_workers

        # CPU使用率调整逻辑
        if avg_cpu > self.target_cpu_percent + 10:
            # CPU使用率过高，减少线程数
            new_workers = max(self.min_workers, old_workers - 1)
        elif avg_cpu < self.target_cpu_percent - 10:
            # CPU使用率过低，增加线程数
            new_workers = min(self.max_workers, old_workers + 1)

        # 内存使用率保护机制
        if memory_percent > 85:
            # 内存使用率过高，减少线程数
            new_workers = max(self.min_workers, old_workers - 2)
            logger.warning(f"内存使用率过高: {memory_percent:.1f}%，减少线程数")
        elif memory_percent > 90:
            # 内存使用率严重过高，强制减少到最小
            new_workers = self.min_workers
            logger.error(f"内存使用率严重过高: {memory_percent:.1f}%，强制减少到最小线程数")
        if new_workers != old_workers:
            # 调整线程池大小
            self._resize_thread_pool(new_workers)
            self.stats['thread_adjustments'] += 1

            logger.info(f"调整线程池大小: {old_workers} -> {new_workers} "
                       f"(CPU: {avg_cpu:.1f}%, 内存: {memory_percent:.1f}%)")

    def _resize_thread_pool(self, new_size: int):
        """
        调整线程池大小

        Args:
            new_size: 新的线程池大小
        """
        # 注意：ThreadPoolExecutor不支持动态调整大小
        # 这里我们重新创建线程池
        try:
            # 关闭旧线程池
            self.executor.shutdown(wait=True, timeout=10)

            # 创建新线程池
            self.executor = ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix=f"ConcurrencyOptimizer-{new_size}"
            )

            self.current_workers = new_size
            logger.debug(f"线程池已调整为 {new_size} 个线程")

        except Exception as e:
            logger.error(f"调整线程池大小失败: {e}")

    def _update_avg_execution_time(self, execution_time: float):
        """
        更新平均执行时间

        Args:
            execution_time: 执行时间
        """
        # 使用移动平均计算
        alpha = 0.1  # 平滑因子
        self.stats['avg_execution_time'] = (
            alpha * execution_time +
            (1 - alpha) * self.stats['avg_execution_time']
        )

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取性能优化建议

        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        stats = self.get_concurrency_stats()
        recommendations = []

        # CPU使用率建议
        if stats['current_cpu_percent'] > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'title': 'CPU使用率过高',
                'description': f'当前CPU使用率 {stats["current_cpu_percent"]:.1f}%，建议优化并发配置',
                'actions': [
                    '减少并发线程数',
                    '检查CPU密集型任务',
                    '考虑任务分批处理'
                ]
            })

        # 内存使用率建议
        if stats['current_memory_percent'] > 80:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'title': '内存使用率过高',
                'description': f'当前内存使用率 {stats["current_memory_percent"]:.1f}%，建议优化内存管理',
                'actions': [
                    '减少并发任务数',
                    '检查内存泄漏',
                    '优化数据结构'
                ]
            })

        # 任务失败率建议
        if stats['success_rate'] < 0.95:
            recommendations.append({
                'type': 'error_handling',
                'priority': 'medium',
                'title': '任务失败率较高',
                'description': f'任务成功率 {stats["success_rate"]:.1%}，建议改进错误处理',
                'actions': [
                    '改进错误处理机制',
                    '添加任务重试机制',
                    '检查任务依赖关系'
                ]
            })

        # 线程调整频率建议
        if stats['thread_adjustments'] > 10:
            recommendations.append({
                'type': 'stability',
                'priority': 'low',
                'title': '线程调整过于频繁',
                'description': f'线程池调整了 {stats["thread_adjustments"]} 次，系统可能不稳定',
                'actions': [
                    '调整监控参数',
                    '增加调整冷却时间',
                    '优化资源分配策略'
                ]
            })

        return recommendations

    def shutdown(self, timeout: float = 10.0):
        """
        关闭并发优化器

        Args:
            timeout: 关闭超时时间
        """
        logger.info("正在关闭并发优化器...")

        # 停止监控
        self.stop_monitoring()

        # 关闭线程池
        self.executor.shutdown(wait=True, timeout=timeout)

        logger.info("并发优化器已关闭")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取并发优化器的健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            stats = self.get_concurrency_stats()

            issues = []

            # 检查任务失败率
            if stats['success_rate'] < 0.9:
                issues.append(f"任务成功率过低: {stats['success_rate']:.1%}")

            # 检查资源使用
            if stats['current_cpu_percent'] > 90:
                issues.append(f"CPU使用率严重过高: {stats['current_cpu_percent']:.1f}%")
            if stats['current_memory_percent'] > 90:
                issues.append(f"内存使用率严重过高: {stats['current_memory_percent']:.1f}%")
            # 检查线程池状态
            if not hasattr(self.executor, '_threads') or not self.executor._threads:
                issues.append("线程池未正常运行")

            # 检查监控状态
            if not self.monitoring_active:
                issues.append("性能监控未启用")

            return {
                'status': 'healthy' if not issues else 'warning',
                'stats': stats,
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局并发优化器实例
global_concurrency_optimizer = ConcurrencyOptimizer()
