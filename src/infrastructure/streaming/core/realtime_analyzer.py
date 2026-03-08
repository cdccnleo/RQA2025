"""
Real - time Analyzer Module
实时分析器模块

This module provides real - time data analysis capabilities for streaming systems
此模块为流系统提供实时数据分析能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from collections import deque
import threading
import time
import statistics

logger = logging.getLogger(__name__)


class RealTimeAnalyzer:

    """
    Real - time Data Analyzer
    实时数据分析器

    Provides real - time analysis capabilities for streaming data
    为流数据提供实时分析能力
    """

    def __init__(self, analyzer_name: str = "default_realtime_analyzer",
                 window_size: int = 1000, analysis_interval: float = 1.0):
        """
        Initialize the real - time analyzer
        初始化实时分析器

        Args:
            analyzer_name: Name of this analyzer
                          此分析器的名称
            window_size: Size of the analysis window
                        分析窗口的大小
            analysis_interval: Interval between analyses (seconds)
                             分析间隔（秒）
        """
        self.analyzer_name = analyzer_name
        self.window_size = window_size
        self.analysis_interval = analysis_interval

        # Data storage
        self.data_window = deque(maxlen=window_size)
        self.timestamp_window = deque(maxlen=window_size)

        # Analysis results
        self.current_metrics = {}
        self.historical_metrics = []

        # Analysis functions
        self.analyzers: Dict[str, Callable] = {}

        # Control flags
        self.is_running = False
        self.analysis_thread = None

        # Statistical accumulators
        self.total_samples = 0
        self.error_count = 0

        logger.info(
            f"Real - time analyzer {analyzer_name} initialized with window size {window_size}")

    def add_data_point(self, data: Any, timestamp: Optional[datetime] = None) -> None:
        """
        Add a data point to the analysis window
        向分析窗口添加数据点

        Args:
            data: Data point to add
                 要添加的数据点
            timestamp: Timestamp for the data point (auto - generated if None)
                      数据点的时间戳（如果为None则自动生成）
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.data_window.append(data)
        self.timestamp_window.append(timestamp)
        self.total_samples += 1

    def register_analyzer(self, name: str, analyzer_func: Callable) -> None:
        """
        Register an analysis function
        注册分析函数

        Args:
            name: Name of the analyzer
                分析器名称
            analyzer_func: Function that performs analysis on data window
                          对数据窗口执行分析的函数
        """
        self.analyzers[name] = analyzer_func
        logger.info(f"Registered analyzer: {name}")

    def start_analysis(self) -> bool:
        """
        Start real - time analysis
        开始实时分析

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.analyzer_name} is already running")
            return False

        try:
            self.is_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            logger.info(f"Real - time analysis started for {self.analyzer_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start real - time analysis for {self.analyzer_name}: {str(e)}")
            self.is_running = False
            return False

    def stop_analysis(self) -> bool:
        """
        Stop real - time analysis
        停止实时分析

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.analyzer_name} is not running")
            return False

        try:
            self.is_running = False
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5.0)
            logger.info(f"Real - time analysis stopped for {self.analyzer_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop real - time analysis for {self.analyzer_name}: {str(e)}")
            return False

    def _analysis_loop(self) -> None:
        """
        Main analysis loop
        主要的分析循环
        """
        logger.info(f"Analysis loop started for {self.analyzer_name}")

        while self.is_running:
            try:
                start_time = time.time()

                # Perform analysis
                self._perform_analysis()

                # Wait for next analysis interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - elapsed_time)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Analysis loop error in {self.analyzer_name}: {str(e)}")
                self.error_count += 1
                time.sleep(self.analysis_interval)

        logger.info(f"Analysis loop stopped for {self.analyzer_name}")

    def _perform_analysis(self) -> None:
        """
        Perform analysis on current data window
        对当前数据窗口执行分析
        """
        if not self.data_window:
            return

        try:
            data_list = list(self.data_window)
            timestamp_list = list(self.timestamp_window)

            # Run all registered analyzers
            metrics = {}
            for name, analyzer_func in self.analyzers.items():
                try:
                    result = analyzer_func(data_list, timestamp_list)
                    metrics[name] = result
                except Exception as e:
                    logger.error(f"Analyzer {name} error: {str(e)}")
                    metrics[name] = None

            # Update current metrics
            self.current_metrics = {
                'timestamp': datetime.now(),
                'window_size': len(data_list),
                'metrics': metrics,
                'data_range': {
                    'start': timestamp_list[0] if timestamp_list else None,
                    'end': timestamp_list[-1] if timestamp_list else None
                }
            }

            # Store historical metrics
            self.historical_metrics.append(self.current_metrics)

            # Keep only recent history (last 100 entries)
            if len(self.historical_metrics) > 100:
                self.historical_metrics = self.historical_metrics[-100:]

            logger.debug(f"Analysis completed for {self.analyzer_name}: {len(metrics)} metrics")

        except Exception as e:
            logger.error(f"Analysis error in {self.analyzer_name}: {str(e)}")
            self.error_count += 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current analysis metrics
        获取当前分析指标

        Returns:
            dict: Current metrics data
                  当前指标数据
        """
        return self.current_metrics.copy()

    def get_historical_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical analysis metrics
        获取历史分析指标

        Args:
            limit: Maximum number of historical entries to return
                  返回的最大历史条目数

        Returns:
            list: Historical metrics data
                  历史指标数据
        """
        return self.historical_metrics[-limit:].copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics
        获取分析器统计信息

        Returns:
            dict: Analyzer statistics
                  分析器统计信息
        """
        return {
            'analyzer_name': self.analyzer_name,
            'is_running': self.is_running,
            'total_samples': self.total_samples,
            'current_window_size': len(self.data_window),
            'error_count': self.error_count,
            'registered_analyzers': list(self.analyzers.keys()),
            'historical_entries': len(self.historical_metrics),
            'success_rate': ((self.total_samples - self.error_count) / max(self.total_samples, 1)) * 100
        }

    def clear_data(self) -> None:
        """
        Clear all data and reset analyzer
        清除所有数据并重置分析器
        """
        self.data_window.clear()
        self.timestamp_window.clear()
        self.current_metrics = {}
        self.historical_metrics.clear()
        self.total_samples = 0
        self.error_count = 0
        logger.info(f"Data cleared for {self.analyzer_name}")


# Built - in analysis functions


def statistical_analyzer(data_list: List[Any], timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Statistical analysis of numeric data
    数值数据的统计分析

    Args:
        data_list: List of data points
                  数据点列表
        timestamps: Corresponding timestamps
                   对应时间戳

    Returns:
        dict: Statistical metrics
              统计指标
    """
    if not data_list:
        return {}

    try:
        # Extract numeric values
        numeric_values = []
        for item in data_list:
            if isinstance(item, (int, float)):
                numeric_values.append(item)
            elif isinstance(item, dict):
                # Try to extract numeric values from dict
                for value in item.values():
                    if isinstance(value, (int, float)):
                        numeric_values.append(value)

        if not numeric_values:
            return {'error': 'No numeric data found'}

        return {
            'count': len(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'std_dev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            'min': min(numeric_values),
            'max': max(numeric_values),
            'range': max(numeric_values) - min(numeric_values)
        }

    except Exception as e:
        return {'error': str(e)}


def trend_analyzer(data_list: List[Any], timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Trend analysis of time series data
    时间序列数据的趋势分析

    Args:
        data_list: List of data points
                  数据点列表
        timestamps: Corresponding timestamps
                   对应时间戳

    Returns:
        dict: Trend analysis results
              趋势分析结果
    """
    if len(data_list) < 3 or len(timestamps) < 3:
        return {'error': 'Insufficient data for trend analysis'}

    try:
        # Simple linear trend analysis
        x_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
        y_values = []

        for item in data_list:
            if isinstance(item, (int, float)):
                y_values.append(item)
            elif isinstance(item, dict):
                # Use first numeric value found
                for value in item.values():
                    if isinstance(value, (int, float)):
                        y_values.append(value)
                        break
                    else:
                        y_values.append(0)  # Default value

        if len(y_values) != len(x_values):
            return {'error': 'Data length mismatch'}

        # Calculate trend using linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        return {
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'data_points': n
        }

    except Exception as e:
        return {'error': str(e)}


def anomaly_detector(data_list: List[Any], timestamps: List[datetime],


                     threshold: float = 2.0) -> Dict[str, Any]:
    """
    Anomaly detection in data stream
    数据流中的异常检测

    Args:
        data_list: List of data points
                  数据点列表
        timestamps: Corresponding timestamps
                   对应时间戳
        threshold: Anomaly detection threshold (standard deviations)
                  异常检测阈值（标准差倍数）

    Returns:
        dict: Anomaly detection results
              异常检测结果
    """
    if len(data_list) < 5:
        return {'error': 'Insufficient data for anomaly detection'}

    try:
        # Extract numeric values
        values = []
        for item in data_list:
            if isinstance(item, (int, float)):
                values.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, (int, float)):
                        values.append(value)
                        break

        if len(values) < 5:
            return {'error': 'Insufficient numeric data'}

        # Calculate statistics
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)

        # Detect anomalies
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_dev if std_dev > 0 else 0
        if z_score > threshold:
            anomalies.append({
                'index': i,
                'value': value,
                'z_score': z_score,
                'timestamp': timestamps[i].isoformat() if i < len(timestamps) else None
            })

        return {
            'total_points': len(values),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': (len(anomalies) / len(values)) * 100,
            'mean_value': mean_val,
            'std_deviation': std_dev,
            'anomalies': anomalies[:10]  # Return top 10 anomalies
        }

    except Exception as e:
        return {'error': str(e)}


# Global default analyzer instance
# 全局默认分析器实例

default_realtime_analyzer = RealTimeAnalyzer("default_realtime_analyzer")

__all__ = [
    'RealTimeAnalyzer',
    'default_realtime_analyzer',
    'statistical_analyzer',
    'trend_analyzer',
    'anomaly_detector'
]
