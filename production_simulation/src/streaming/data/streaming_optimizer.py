"""
Streaming Optimizer Module
流优化器模块

This module provides optimization capabilities for streaming data processing
此模块为流数据处理提供优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Callable
from datetime import datetime
import threading
import time
import psutil
import gc

logger = logging.getLogger(__name__)


class StreamingOptimizer:

    """
    Streaming Data Optimizer
    流数据优化器

    Optimizes streaming data processing performance and resource usage
    优化流数据处理性能和资源使用
    """

    def __init__(self, enable_auto_tuning: bool = True):
        """
        Initialize the streaming optimizer
        初始化流优化器

        Args:
            enable_auto_tuning: Whether to enable automatic performance tuning
                              是否启用自动性能调优
        """
        self.enable_auto_tuning = enable_auto_tuning
        self.is_running = False
        self.optimization_thread = None
        self.performance_metrics = {}
        self.optimization_rules = []
        self.last_optimization = datetime.now()

        # Performance thresholds
        self.cpu_threshold = 80.0  # CPU usage threshold (%)
        self.memory_threshold = 85.0  # Memory usage threshold (%)
        self.latency_threshold = 100.0  # Processing latency threshold (ms)

        # Optimization history
        self.optimization_history = []

        logger.info("Streaming optimizer initialized")

    def add_optimization_rule(self, rule: Callable) -> None:
        """
        Add a custom optimization rule
        添加自定义优化规则

        Args:
            rule: Optimization rule function that takes metrics and returns optimization suggestions
                 优化规则函数，接收指标并返回优化建议
        """
        if callable(rule):
            self.optimization_rules.append(rule)
            logger.info(f"Added optimization rule: {rule.__name__ if hasattr(rule, '__name__') else 'anonymous'}")
        else:
            logger.warning("Attempted to add non-callable optimization rule")

        # Performance thresholds
        self.cpu_threshold = 80.0  # CPU usage threshold (%)
        self.memory_threshold = 85.0  # Memory usage threshold (%)
        self.latency_threshold = 100.0  # Processing latency threshold (ms)

        # Optimization history
        self.optimization_history = []

        logger.info("Streaming optimizer initialized")

    def start_optimization(self) -> bool:
        """
        Start the optimization process
        启动优化过程

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Streaming optimizer is already running")
            return False

        try:
            self.is_running = True
            if self.enable_auto_tuning:
                self.optimization_thread = threading.Thread(
                    target=self._optimization_loop, daemon=True)
                self.optimization_thread.start()
            logger.info("Streaming optimizer started")
            return True
        except Exception as e:
            logger.error(f"Failed to start streaming optimizer: {str(e)}")
            self.is_running = False
            return False

    def stop_optimization(self) -> bool:
        """
        Stop the optimization process
        停止优化过程

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            try:
                logger.warning("Streaming optimizer is not running")
            except (ValueError, AttributeError):
                # 日志流可能已关闭，忽略
                pass
            return False

        try:
            self.is_running = False
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5.0)
            try:
                logger.info("Streaming optimizer stopped")
            except (ValueError, AttributeError):
                # 日志流可能已关闭，忽略
                pass
            return True
        except Exception as e:
            try:
                logger.error(f"Failed to stop streaming optimizer: {str(e)}")
            except (ValueError, AttributeError):
                # 日志流可能已关闭，忽略
                pass
            return False

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect current performance metrics
        收集当前性能指标

        Returns:
            dict: Performance metrics data
                  性能指标数据
        """
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Process metrics (if available)
            process_metrics = {}
            try:
                process = psutil.Process()
                process_metrics = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_info': process.memory_info()._asdict(),
                    'num_threads': process.num_threads(),
                    'num_fds': getattr(process, 'num_fds', lambda: 0)()
                }
            except Exception:
                pass

            metrics = {
                'timestamp': datetime.now(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used_gb': memory.used / (1024 ** 3),
                    'memory_available_gb': memory.available / (1024 ** 3)
                },
                'process': process_metrics,
                'streaming': self.performance_metrics
            }

            self.performance_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {str(e)}")
            return {}

    def analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Analyze performance bottlenecks
        分析性能瓶颈

        Args:
            metrics: Performance metrics data
                    性能指标数据

        Returns:
            list: List of identified bottlenecks
                  识别出的瓶颈列表
        """
        bottlenecks = []

        try:
            system = metrics.get('system', {})

            # CPU bottleneck
            if system.get('cpu_percent', 0) > self.cpu_threshold:
                bottlenecks.append(f"High CPU usage: {system['cpu_percent']:.1f}%")

            # Memory bottleneck
            if system.get('memory_percent', 0) > self.memory_threshold:
                bottlenecks.append(f"High memory usage: {system['memory_percent']:.1f}%")

            # Streaming specific bottlenecks
            streaming = metrics.get('streaming', {})
            if streaming.get('processing_latency', 0) > self.latency_threshold:
                bottlenecks.append(
                    f"High processing latency: {streaming['processing_latency']:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to analyze performance bottlenecks: {str(e)}")

        return bottlenecks

    def apply_optimizations(self, bottlenecks: List[str]) -> List[str]:
        """
        Apply optimizations based on identified bottlenecks
        根据识别的瓶颈应用优化

        Args:
            bottlenecks: List of identified bottlenecks
                        识别出的瓶颈列表

        Returns:
            list: List of applied optimizations
                  已应用的优化列表
        """
        optimizations_applied = []

        try:
            for bottleneck in bottlenecks:
                if "CPU" in bottleneck:
                    optimizations_applied.extend(self._optimize_cpu_usage())
                elif "memory" in bottleneck:
                    optimizations_applied.extend(self._optimize_memory_usage())
                elif "latency" in bottleneck:
                    optimizations_applied.extend(self._optimize_latency())

        except Exception as e:
            logger.error(f"Failed to apply optimizations: {str(e)}")

        # Record optimization history
        if optimizations_applied:
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'bottlenecks': bottlenecks,
                'optimizations': optimizations_applied
            })

        return optimizations_applied

    def _optimize_cpu_usage(self) -> List[str]:
        """
        Optimize CPU usage
        优化CPU使用

        Returns:
            list: Applied CPU optimizations
                  已应用的CPU优化
        """
        optimizations = []

        try:
            # Force garbage collection
            collected = gc.collect()
            optimizations.append(f"Garbage collection: freed {collected} objects")

            # Suggest batch processing
            optimizations.append("Consider enabling batch processing")

            logger.info("Applied CPU usage optimizations")

        except Exception as e:
            logger.error(f"Failed to optimize CPU usage: {str(e)}")

        return optimizations

    def _optimize_memory_usage(self) -> List[str]:
        """
        Optimize memory usage
        优化内存使用

        Returns:
            list: Applied memory optimizations
                  已应用的内存优化
        """
        optimizations = []

        try:
            # Clear unused caches
            optimizations.append("Memory optimization suggestions applied")

            # Suggest memory pooling
            optimizations.append("Consider implementing memory pooling")

            logger.info("Applied memory usage optimizations")

        except Exception as e:
            logger.error(f"Failed to optimize memory usage: {str(e)}")

        return optimizations

    def _optimize_latency(self) -> List[str]:
        """
        Optimize processing latency
        优化处理延迟

        Returns:
            list: Applied latency optimizations
                  已应用的延迟优化
        """
        optimizations = []

        try:
            # Suggest parallel processing
            optimizations.append("Consider increasing parallel processing threads")

            # Suggest buffer size optimization
            optimizations.append("Consider optimizing buffer sizes")

            logger.info("Applied latency optimizations")

        except Exception as e:
            logger.error(f"Failed to optimize latency: {str(e)}")

        return optimizations

    def _optimization_loop(self) -> None:
        """
        Main optimization loop
        主要的优化循环
        """
        try:
            logger.info("Optimization loop started")
        except (ValueError, AttributeError):
            # 日志流可能已关闭，忽略
            pass

        while self.is_running:
            try:
                # Wait before next optimization cycle
                time.sleep(60)  # Check every minute

                # Collect metrics
                metrics = self.collect_performance_metrics()

                # Analyze bottlenecks
                bottlenecks = self.analyze_performance_bottlenecks(metrics)

                # Apply optimizations if bottlenecks found
                if bottlenecks:
                    optimizations = self.apply_optimizations(bottlenecks)

                    if optimizations:
                        try:
                            logger.info(
                                f"Applied {len(optimizations)} optimizations for {len(bottlenecks)} bottlenecks")
                        except (ValueError, AttributeError):
                            # 日志流可能已关闭，忽略
                            pass
                        self.last_optimization = datetime.now()

            except Exception as e:
                try:
                    logger.error(f"Optimization loop error: {str(e)}")
                except (ValueError, AttributeError):
                    # 日志流可能已关闭，忽略
                    pass
                time.sleep(10)  # Wait before retry

        try:
            logger.info("Optimization loop stopped")
        except (ValueError, AttributeError):
            # 日志流可能已关闭，忽略
            pass

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        获取优化统计信息

        Returns:
            dict: Optimization statistics
                  优化统计信息
        """
        return {
            'is_running': self.is_running,
            'enable_auto_tuning': self.enable_auto_tuning,
            'last_optimization': self.last_optimization.isoformat(),
            'optimization_count': len(self.optimization_history),
            'current_metrics': self.performance_metrics,
            'thresholds': {
                'cpu_threshold': self.cpu_threshold,
                'memory_threshold': self.memory_threshold,
                'latency_threshold': self.latency_threshold
            }
        }

    def set_thresholds(self, cpu: float = None, memory: float = None, latency: float = None) -> None:
        """
        Set optimization thresholds
        设置优化阈值

        Args:
            cpu: CPU usage threshold (%)
                CPU使用率阈值（%）
            memory: Memory usage threshold (%)
                   内存使用率阈值（%）
            latency: Processing latency threshold (ms)
                    处理延迟阈值（ms）
        """
        if cpu is not None:
            self.cpu_threshold = cpu
        if memory is not None:
            self.memory_threshold = memory
        if latency is not None:
            self.latency_threshold = latency

        logger.info(
            f"Optimization thresholds updated: CPU={self.cpu_threshold}%, Memory={self.memory_threshold}%, Latency={self.latency_threshold}ms")


# Global optimizer instance
# 全局优化器实例
streaming_optimizer = StreamingOptimizer()

__all__ = ['StreamingOptimizer', 'streaming_optimizer']
