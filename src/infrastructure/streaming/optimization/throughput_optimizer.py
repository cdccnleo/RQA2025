"""
Throughput Optimizer Module
吞吐量优化器模块

This module provides throughput optimization capabilities for streaming operations
此模块为流操作提供吞吐量优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Callable
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import statistics
import concurrent.futures

logger = logging.getLogger(__name__)


class ThroughputOptimizer:

    """
    Streaming Throughput Optimizer
    流吞吐量优化器

    Optimizes data processing throughput in streaming pipelines
    优化流管道中的数据处理吞吐量
    """

    def __init__(self, target_throughput: int = 1000,


                 monitoring_window: int = 60):
        """
        Initialize the throughput optimizer
        初始化吞吐量优化器

        Args:
            target_throughput: Target throughput (items per second)
                             目标吞吐量（每秒项目数）
            monitoring_window: Monitoring window in seconds
                             监控窗口（秒）
        """
        self.target_throughput = target_throughput
        self.monitoring_window = monitoring_window

        # Throughput monitoring
        self.throughput_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=1000)
        self.is_running = False
        self.monitoring_thread = None

        # Optimization parameters
        self.batch_size = 100
        self.worker_count = 4
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count)

        # Performance metrics
        self.metrics_window = deque(maxlen=monitoring_window)

        logger.info(f"Throughput optimizer initialized with target {target_throughput} items / sec")

    def start_optimization(self) -> bool:
        """
        Start throughput optimization monitoring
        开始吞吐量优化监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Throughput optimizer is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Throughput optimization monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start throughput optimization: {str(e)}")
            self.is_running = False
            return False

    def stop_optimization(self) -> bool:
        """
        Stop throughput optimization monitoring
        停止吞吐量优化监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Throughput optimizer is not running")
            return False

        try:
            self.is_running = False
            self.executor.shutdown(wait=True)
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Throughput optimization monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop throughput optimization: {str(e)}")
            return False

    def record_processing_time(self, processing_time: float) -> None:
        """
        Record processing time for throughput calculation
        记录处理时间以计算吞吐量

        Args:
            processing_time: Time taken to process an item (seconds)
                           处理单个项目的时间（秒）
        """
        self.processing_times.append(processing_time)

        # Calculate current throughput
        if len(self.processing_times) > 10:
            avg_processing_time = statistics.mean(self.processing_times)
            current_throughput = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

            timestamp = datetime.now()
            throughput_data = {
                'timestamp': timestamp,
                'throughput': current_throughput,
                'avg_processing_time': avg_processing_time,
                'sample_count': len(self.processing_times)
            }

            self.throughput_history.append(throughput_data)
            self.metrics_window.append(throughput_data)

    def get_current_throughput(self) -> Dict[str, Any]:
        """
        Get current throughput metrics
        获取当前吞吐量指标

        Returns:
            dict: Current throughput data
                  当前吞吐量数据
        """
        if not self.throughput_history:
            return {
                'throughput': 0,
                'avg_processing_time': 0,
                'target_throughput': self.target_throughput,
                'status': 'no_data'
            }

        latest = self.throughput_history[-1]
        status = 'optimal' if latest['throughput'] >= self.target_throughput * 0.9 else 'suboptimal'

        return {
            'throughput': latest['throughput'],
            'avg_processing_time': latest['avg_processing_time'],
            'target_throughput': self.target_throughput,
            'efficiency': (latest['throughput'] / self.target_throughput) * 100,
            'status': status,
            'sample_count': latest['sample_count']
        }

    def optimize_batch_processing(self, data_items: List[Any],


                                  processing_func: Callable) -> List[Any]:
        """
        Optimize batch processing for better throughput
        优化批量处理以提高吞吐量

        Args:
            data_items: List of data items to process
                       要处理的数据项列表
            processing_func: Function to process each item
                           处理每个项目的函数

        Returns:
            list: Processed results
                  处理后的结果
        """
        if not data_items:
            return []

        try:
            # Dynamic batch sizing based on current throughput
            current_metrics = self.get_current_throughput()
            efficiency = current_metrics.get('efficiency', 100)

            # Adjust batch size based on efficiency
            if efficiency < 80:
                self.batch_size = max(10, self.batch_size // 2)  # Reduce batch size
            elif efficiency > 120:
                self.batch_size = min(1000, self.batch_size * 2)  # Increase batch size

            # Split data into optimal batches
            batches = []
            for i in range(0, len(data_items), self.batch_size):
                batches.append(data_items[i:i + self.batch_size])

            # Process batches concurrently
            results = []
            futures = []

            for batch in batches:
                future = self.executor.submit(self._process_batch, batch, processing_func)
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result(timeout=30)
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Failed to optimize batch processing: {str(e)}")
            return []

    def _process_batch(self, batch: List[Any], processing_func: Callable) -> List[Any]:
        """
        Process a single batch of data
        处理单个数据批次

        Args:
            batch: Batch of data items
                  数据项批次
            processing_func: Processing function
                           处理函数

        Returns:
            list: Processed results
                  处理后的结果
        """
        results = []
        batch_start_time = time.time()

        for item in batch:
            try:
                item_start_time = time.time()
                result = processing_func(item)
                item_processing_time = time.time() - item_start_time

                results.append(result)
                self.record_processing_time(item_processing_time)

            except Exception as e:
                logger.error(f"Item processing failed: {str(e)}")
                results.append(None)

        batch_processing_time = time.time() - batch_start_time
        logger.debug(f"Processed batch of {len(batch)} items in {batch_processing_time:.3f}s")

        return results

    def optimize_worker_count(self) -> int:
        """
        Optimize the number of worker threads
        优化工作线程数量

        Returns:
            int: Optimal number of workers
                 最佳工作线程数量
        """
        try:
            current_metrics = self.get_current_throughput()
            efficiency = current_metrics.get('efficiency', 100)

            # Adjust worker count based on efficiency
            if efficiency < 70:
                optimal_workers = min(16, self.worker_count + 1)
            elif efficiency > 130:
                optimal_workers = max(1, self.worker_count - 1)
            else:
                optimal_workers = self.worker_count

            if optimal_workers != self.worker_count:
                logger.info(f"Optimizing worker count: {self.worker_count} -> {optimal_workers}")
                self.worker_count = optimal_workers

                # Recreate executor with new worker count
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count)

            return self.worker_count

        except Exception as e:
            logger.error(f"Failed to optimize worker count: {str(e)}")
            return self.worker_count

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop for throughput optimization
        吞吐量优化的主要监控循环
        """
        logger.info("Throughput optimization monitoring loop started")

        while self.is_running:
            try:
                # Get current metrics
                current_metrics = self.get_current_throughput()

                # Apply optimizations if needed
                if current_metrics.get('efficiency', 100) < 90:
                    self.optimize_worker_count()

                    # Log optimization actions
                    logger.info(
                        f"Throughput optimization applied: efficiency {current_metrics.get('efficiency', 0):.1f}%")

                # Clean old metrics
                self._cleanup_old_metrics()

                # Wait before next monitoring cycle
                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Throughput monitoring loop error: {str(e)}")
                time.sleep(10)

        logger.info("Throughput optimization monitoring loop stopped")

    def _cleanup_old_metrics(self) -> None:
        """
        Clean up old throughput metrics
        清理旧的吞吐量指标
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=5)

            # Remove old metrics
            while self.metrics_window and self.metrics_window[0]['timestamp'] < cutoff_time:
                self.metrics_window.popleft()

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {str(e)}")

    def get_throughput_stats(self) -> Dict[str, Any]:
        """
        Get throughput optimization statistics
        获取吞吐量优化统计信息

        Returns:
            dict: Throughput statistics
                  吞吐量统计信息
        """
        try:
            current_metrics = self.get_current_throughput()

            # Calculate historical statistics
            throughputs = [m['throughput'] for m in self.throughput_history]
            processing_times = list(self.processing_times)

            stats = {
                'is_running': self.is_running,
                'target_throughput': self.target_throughput,
                'current_throughput': current_metrics.get('throughput', 0),
                'current_efficiency': current_metrics.get('efficiency', 0),
                'batch_size': self.batch_size,
                'worker_count': self.worker_count,
                'monitoring_window': self.monitoring_window,
                'metrics_history_size': len(self.throughput_history),
                'processing_times_count': len(self.processing_times)
            }

            # Add statistical summaries if data is available
            if throughputs:
                stats['throughput_stats'] = {
                    'mean': statistics.mean(throughputs),
                    'median': statistics.median(throughputs),
                    'min': min(throughputs),
                    'max': max(throughputs),
                    'std_dev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
                }

            if processing_times:
                stats['processing_time_stats'] = {
                    'mean': statistics.mean(processing_times),
                    'median': statistics.median(processing_times),
                    'min': min(processing_times),
                    'max': max(processing_times),
                    'std_dev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get throughput stats: {str(e)}")
            return {}

    def reset_metrics(self) -> None:
        """
        Reset all throughput metrics
        重置所有吞吐量指标
        """
        self.throughput_history.clear()
        self.processing_times.clear()
        self.metrics_window.clear()
        logger.info("Throughput metrics reset")


# Global throughput optimizer instance
# 全局吞吐量优化器实例
throughput_optimizer = ThroughputOptimizer()

__all__ = ['ThroughputOptimizer', 'throughput_optimizer']
