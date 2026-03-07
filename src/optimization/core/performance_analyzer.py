"""
Performance Analyzer Module
性能分析器模块

This module provides performance analysis capabilities for optimization algorithms
此模块为优化算法提供性能分析能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
import statistics
import psutil

logger = logging.getLogger(__name__)


class PerformanceMetrics:

    """
    Performance Metrics Class
    性能指标类

    Collects and analyzes performance metrics for optimization algorithms
    为优化算法收集和分析性能指标
    """

    def __init__(self):
        """Initialize performance metrics"""
        self.metrics = defaultdict(list)
        self.timestamps = []
        self.metric_history = deque(maxlen=1000)  # Keep last 1000 measurements

    def record_metric(self, metric_name: str, value: Union[int, float], timestamp: Optional[datetime] = None):
        """
        Record a performance metric
        记录性能指标

        Args:
            metric_name: Name of the metric
                        指标名称
            value: Metric value
                  指标值
            timestamp: Timestamp for the metric
                      指标时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.metrics[metric_name].append(value)
        self.timestamps.append(timestamp)

        # Keep metrics history
        self.metric_history.append({
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp
        })

    def get_metric_stats(self, metric_name: str, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific metric
        获取特定指标的统计信息

        Args:
            metric_name: Name of the metric
                        指标名称
            window_size: Size of the sliding window (None for all data)
                        滑动窗口大小（None表示所有数据）

        Returns:
            dict: Metric statistics
                  指标统计信息
        """
        if metric_name not in self.metrics:
            return {'error': f'Metric {metric_name} not found'}

        values = self.metrics[metric_name]

        if window_size and len(values) > window_size:
            values = values[-window_size:]

        if not values:
            return {'error': 'No data available'}

        try:
            stats = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'latest': values[-1],
                'metric_name': metric_name
            }

            # Calculate percentiles
            if len(values) >= 10:
                sorted_values = sorted(values)
                stats['percentile_25'] = sorted_values[int(len(sorted_values) * 0.25)]
                stats['percentile_75'] = sorted_values[int(len(sorted_values) * 0.75)]
                stats['percentile_95'] = sorted_values[int(len(sorted_values) * 0.95)]

            return stats

        except Exception as e:
            return {'error': f'Statistics calculation failed: {str(e)}'}

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        获取所有指标的摘要

        Returns:
            dict: Metrics summary
                  指标摘要
        """
        summary = {
            'total_metrics': len(self.metrics),
            'total_measurements': sum(len(values) for values in self.metrics.values()),
            'metric_names': list(self.metrics.keys()),
            'timestamp_range': {
                'start': min(self.timestamps) if self.timestamps else None,
                'end': max(self.timestamps) if self.timestamps else None
            },
            'metrics': {}
        }

        for metric_name in self.metrics:
            summary['metrics'][metric_name] = self.get_metric_stats(metric_name, window_size=100)

        return summary

    def clear_metrics(self, metric_name: Optional[str] = None):
        """
        Clear metrics data
        清除指标数据

        Args:
            metric_name: Name of metric to clear (None for all)
                        要清除的指标名称（None表示全部）
        """
        if metric_name:
            if metric_name in self.metrics:
                del self.metrics[metric_name]
        else:
            self.metrics.clear()
            self.timestamps.clear()
            self.metric_history.clear()


class PerformanceAnalyzer:

    """
    Performance Analyzer Class
    性能分析器类

    Analyzes the performance of optimization algorithms and provides insights
    分析优化算法的性能并提供洞察
    """

    def __init__(self, analysis_window: int = 100):
        """
        Initialize performance analyzer
        初始化性能分析器

        Args:
            analysis_window: Size of analysis window
                           分析窗口大小
        """
        self.analysis_window = analysis_window
        self.metrics_collector = PerformanceMetrics()

        # Analysis results
        self.analysis_results = []
        self.performance_trends = defaultdict(list)

        # System performance monitoring
        self.system_monitoring = True
        self.monitoring_interval = 5.0  # seconds
        self.monitoring_thread = None
        self.is_monitoring = False

        logger.info("Performance analyzer initialized")

    def start_performance_monitoring(self) -> bool:
        """
        Start performance monitoring
        开始性能监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return False

        try:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {str(e)}")
            self.is_monitoring = False
            return False

    def stop_performance_monitoring(self) -> bool:
        """
        Stop performance monitoring
        停止性能监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_monitoring:
            logger.warning("Performance monitoring not running")
            return False

        try:
            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Performance monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop performance monitoring: {str(e)}")
            return False

    def analyze_algorithm_performance(self,

                                      algorithm_name: str,
                                      execution_time: float,
                                      iterations: int,
                                      convergence: bool,
                                      final_objective: float,
                                      initial_objective: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze performance of an optimization algorithm
        分析优化算法的性能

        Args:
            algorithm_name: Name of the algorithm
                          算法名称
            execution_time: Time taken for execution
                           执行所用时间
            iterations: Number of iterations
                       迭代次数
            convergence: Whether algorithm converged
                        算法是否收敛
            final_objective: Final objective function value
                           最终目标函数值
            initial_objective: Initial objective function value
                             初始目标函数值

        Returns:
            dict: Performance analysis results
                  性能分析结果
        """
        # Record metrics
        self.metrics_collector.record_metric(f"{algorithm_name}_execution_time", execution_time)
        self.metrics_collector.record_metric(f"{algorithm_name}_iterations", iterations)
        self.metrics_collector.record_metric(
            f"{algorithm_name}_convergence", 1 if convergence else 0)
        self.metrics_collector.record_metric(f"{algorithm_name}_final_objective", final_objective)

        if initial_objective is not None:
            improvement = initial_objective - final_objective
            improvement_rate = improvement / abs(initial_objective) if initial_objective != 0 else 0
            self.metrics_collector.record_metric(f"{algorithm_name}_improvement", improvement)
            self.metrics_collector.record_metric(
                f"{algorithm_name}_improvement_rate", improvement_rate)

        # Calculate performance metrics
        analysis = {
            'algorithm_name': algorithm_name,
            'execution_time': execution_time,
            'iterations': iterations,
            'convergence': convergence,
            'final_objective': final_objective,
            'initial_objective': initial_objective,
            'timestamp': datetime.now(),
            'performance_score': self._calculate_performance_score(
                execution_time, iterations, convergence
            )
        }

        if initial_objective is not None:
            analysis['improvement'] = initial_objective - final_objective
            analysis['improvement_rate'] = improvement_rate

        # Add efficiency metrics
        analysis['efficiency_metrics'] = self._calculate_efficiency_metrics(
            algorithm_name, execution_time, iterations
        )

        # Store analysis result
        self.analysis_results.append(analysis)

        # Update performance trends
        self._update_performance_trends(algorithm_name, analysis)

        return analysis

    def _calculate_performance_score(self,


                                     execution_time: float,
                                     iterations: int,
                                     convergence: bool) -> float:
        """
        Calculate overall performance score
        计算整体性能评分

        Args:
            execution_time: Execution time
                           执行时间
            iterations: Number of iterations
                       迭代次数
            convergence: Whether converged
                        是否收敛

        Returns:
            float: Performance score (0.0 to 1.0)
                   性能评分（0.0到1.0）
        """
        # Base score from convergence
        score = 1.0 if convergence else 0.3

        # Adjust for execution time (prefer faster algorithms)
        time_penalty = min(execution_time / 60.0, 0.5)  # Max 50% penalty for slow execution
        score *= (1.0 - time_penalty)

        # Adjust for iterations (prefer fewer iterations)
        iteration_penalty = min(iterations / 1000.0, 0.3)  # Max 30% penalty for many iterations
        score *= (1.0 - iteration_penalty)

        return max(0.0, score)

    def _calculate_efficiency_metrics(self,


                                      algorithm_name: str,
                                      execution_time: float,
                                      iterations: int) -> Dict[str, Any]:
        """
        Calculate efficiency metrics for an algorithm
        计算算法的效率指标

        Args:
            algorithm_name: Name of the algorithm
                          算法名称
            execution_time: Execution time
                           执行时间
            iterations: Number of iterations
                       迭代次数

        Returns:
            dict: Efficiency metrics
                  效率指标
        """
        metrics = {}

        # Iterations per second
        if execution_time > 0:
            metrics['iterations_per_second'] = iterations / execution_time

        # Average time per iteration
        if iterations > 0:
            metrics['time_per_iteration'] = execution_time / iterations

        # Get historical performance
        time_stats = self.metrics_collector.get_metric_stats(
            f"{algorithm_name}_execution_time", window_size=self.analysis_window
        )
        iteration_stats = self.metrics_collector.get_metric_stats(
            f"{algorithm_name}_iterations", window_size=self.analysis_window
        )

        if 'mean' in time_stats:
            metrics['avg_execution_time'] = time_stats['mean']
            metrics['execution_time_std'] = time_stats.get('std_dev', 0)

        if 'mean' in iteration_stats:
            metrics['avg_iterations'] = iteration_stats['mean']
            metrics['iterations_std'] = iteration_stats.get('std_dev', 0)

        return metrics

    def _update_performance_trends(self, algorithm_name: str, analysis: Dict[str, Any]) -> None:
        """
        Update performance trends for an algorithm
        更新算法的性能趋势

        Args:
            algorithm_name: Name of the algorithm
                          算法名称
            analysis: Performance analysis result
                     性能分析结果
        """
        trend_data = {
            'timestamp': analysis['timestamp'],
            'performance_score': analysis['performance_score'],
            'execution_time': analysis['execution_time'],
            'iterations': analysis['iterations'],
            'convergence': analysis['convergence']
        }

        self.performance_trends[algorithm_name].append(trend_data)

        # Keep only recent trends
        if len(self.performance_trends[algorithm_name]) > self.analysis_window:
            self.performance_trends[algorithm_name] = self.performance_trends[algorithm_name][-self.analysis_window:]

    def compare_algorithms(self, algorithm_names: List[str]) -> Dict[str, Any]:
        """
        Compare performance of multiple algorithms
        比较多个算法的性能

        Args:
            algorithm_names: List of algorithm names to compare
                            要比较的算法名称列表

        Returns:
            dict: Algorithm comparison results
                  算法比较结果
        """
        comparison = {
            'algorithms': algorithm_names,
            'comparison_timestamp': datetime.now(),
            'metrics': {},
            'rankings': {}
        }

        # Collect metrics for each algorithm
        for algorithm in algorithm_names:
            metrics = {
                'execution_time': self.metrics_collector.get_metric_stats(
                    f"{algorithm}_execution_time", window_size=self.analysis_window
                ),
                'iterations': self.metrics_collector.get_metric_stats(
                    f"{algorithm}_iterations", window_size=self.analysis_window
                ),
                'convergence_rate': self.metrics_collector.get_metric_stats(
                    f"{algorithm}_convergence", window_size=self.analysis_window
                ),
                'performance_score': self._calculate_average_performance_score(algorithm)
            }
            comparison['metrics'][algorithm] = metrics

        # Create rankings
        comparison['rankings'] = self._rank_algorithms(comparison['metrics'])

        return comparison

    def _calculate_average_performance_score(self, algorithm_name: str) -> float:
        """
        Calculate average performance score for an algorithm
        计算算法的平均性能评分

        Args:
            algorithm_name: Name of the algorithm
                          算法名称

        Returns:
            float: Average performance score
                   平均性能评分
        """
        trends = self.performance_trends.get(algorithm_name, [])
        if not trends:
            return 0.0

        scores = [trend['performance_score'] for trend in trends]
        return statistics.mean(scores) if scores else 0.0

    def _rank_algorithms(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Rank algorithms based on different criteria
        基于不同标准对算法进行排名

        Args:
            metrics: Algorithm metrics
                    算法指标

        Returns:
            dict: Rankings by different criteria
                  按不同标准的排名
        """
        rankings = {}

        # Rank by average execution time (lower is better)
        execution_times = {}
        for alg, metric_data in metrics.items():
            exec_time = metric_data.get('execution_time', {}).get('mean', float('inf'))
            execution_times[alg] = exec_time

        rankings['by_execution_time'] = sorted(
            execution_times.keys(), key=lambda x: execution_times[x])

        # Rank by convergence rate (higher is better)
        convergence_rates = {}
        for alg, metric_data in metrics.items():
            conv_rate = metric_data.get('convergence_rate', {}).get('mean', 0)
            convergence_rates[alg] = conv_rate

        rankings['by_convergence_rate'] = sorted(
            convergence_rates.keys(),
            key=lambda x: convergence_rates[x],
            reverse=True
        )

        # Rank by overall performance score (higher is better)
        performance_scores = {}
        for alg, metric_data in metrics.items():
            score = metric_data.get('performance_score', 0)
            performance_scores[alg] = score

        rankings['by_performance_score'] = sorted(
            performance_scores.keys(),
            key=lambda x: performance_scores[x],
            reverse=True
        )

        return rankings

    def get_performance_report(self,


                               algorithm_name: Optional[str] = None,
                               time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate performance report
        生成性能报告

        Args:
            algorithm_name: Specific algorithm to report on (None for all)
                           要报告的特定算法（None表示全部）
            time_window: Time window for the report
                        报告的时间窗口

        Returns:
            dict: Performance report
                  性能报告
        """
        report = {
            'report_timestamp': datetime.now(),
            'time_window': time_window,
            'system_info': self._get_system_info()
        }

        if algorithm_name:
            # Single algorithm report
            report['algorithm_name'] = algorithm_name
            report['metrics'] = self.metrics_collector.get_metric_stats(
                f"{algorithm_name}_execution_time"
            )
            report['trends'] = self.performance_trends.get(algorithm_name, [])
            report['analysis_results'] = [
                result for result in self.analysis_results
                if result['algorithm_name'] == algorithm_name
            ]
        else:
            # All algorithms report
            report['overall_metrics'] = self.metrics_collector.get_all_metrics_summary()
            report['algorithm_trends'] = dict(self.performance_trends)
            report['comparison'] = self.compare_algorithms(list(self.performance_trends.keys()))

        return report

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information
        获取系统信息

        Returns:
            dict: System information
                  系统信息
        """
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'python_version': __import__('sys').version,
                'platform': __import__('platform').platform()
            }
        except Exception:
            return {'error': 'System info unavailable'}

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info("Performance monitoring loop started")

        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                self.metrics_collector.record_metric('system_cpu_percent', cpu_percent)
                self.metrics_collector.record_metric('system_memory_percent', memory_percent)

                # Sleep before next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Performance monitoring loop error: {str(e)}")
                time.sleep(self.monitoring_interval)

        logger.info("Performance monitoring loop stopped")

    def export_performance_data(self, filepath: str, format_type: str = 'json') -> bool:
        """
        Export performance data to file
        将性能数据导出到文件

        Args:
            filepath: Path to export file
                     导出文件路径
            format_type: Export format ('json' or 'csv')
                        导出格式（'json'或'csv'）

        Returns:
            bool: True if export successful
                  导出成功返回True
        """
        try:
            data = {
                'metrics': dict(self.metrics_collector.metrics),
                'analysis_results': self.analysis_results,
                'performance_trends': dict(self.performance_trends),
                'export_timestamp': datetime.now().isoformat()
            }

            if format_type == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # CSV export (simplified)
                with open(filepath, 'w') as f:
                    f.write("Performance data export\n")
                    f.write(f"Export timestamp: {datetime.now()}\n")
                    f.write(f"Total metrics: {len(data['metrics'])}\n")
                    f.write(f"Total analysis results: {len(data['analysis_results'])}\n")

            logger.info(f"Performance data exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export performance data: {str(e)}")
            return False


# Global performance analyzer instance
# 全局性能分析器实例
performance_analyzer = PerformanceAnalyzer()

__all__ = [
    'PerformanceMetrics',
    'PerformanceAnalyzer',
    'performance_analyzer'
]
