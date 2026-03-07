"""
slow_query_monitor 模块

提供 slow_query_monitor 相关功能和接口。
"""

import logging

import time

from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from src.infrastructure.utils.core.interfaces import QueryResult
from threading import Lock
from typing import Dict, Any, Optional, List, Callable
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础设施层 - 日志系统组件slow_query_monitor 模块日志系统相关的文件提供日志系统相关的功能实现。"""

"""slow_query_monitor - 日志系统职责说明：
负责系统日志记录、日志格式化、日志存储和日志分析
核心职责：
- 日志记录和格式化
- 日志级别管理
- 日志存储和轮转
- 日志分析和监控
- 日志搜索和过滤
- 日志性能优化
相关接口：
- ILoggingComponent
- ILogger
- ILogHandler
"""

"""慢查询监控器实现查询时间统计、慢查询日志和性能告警"""

from .base_monitor import AlertData, AlertLevel


@dataclass
class SlowQueryRecord:
    """慢查询记录"""
    query: str
    params: Optional[Dict[str, Any]]
    execution_time: float
    timestamp: datetime
    database_type: str
    success: bool
    row_count: int
    error_message: Optional[str] = None


@dataclass
class PerformanceAlert:
    """性能告警"""
    level: AlertLevel
    message: str
    timestamp: datetime
    details: Dict[str, Any]


class SlowQueryMonitor:
    """慢查询监控器"""

    def __init__(
        self,
        slow_query_threshold: float = 1.0,
        max_records: int = 1000,
        alert_callbacks: Optional[List[Callable]] = None,
        *,
        threshold: Optional[float] = None,
    ):
        """
        初始化慢查询监控器
        Args:
            slow_query_threshold: 慢查询阈值（秒）
            max_records: 最大记录数
            alert_callbacks: 告警回调函数列表
        """
        self._slow_query_threshold = threshold if threshold is not None else slow_query_threshold
        self._max_records = max_records
        self._alert_callbacks = alert_callbacks or []
        self._slow_queries: deque = deque(maxlen=max_records)
        self._query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'last_execution': None
        })
        self._alerts: deque = deque(maxlen=100)
        self._lock = Lock()
        # 设置日志
        self._logger = logging.getLogger('slow_query_monitor')
        self._setup_logging()

    def record_query(self, query: str, params: Optional[Dict[str, Any]],
                     execution_time: float, database_type: str,
                     success: bool, row_count: int, error_message: Optional[str] = None):
        """
        记录查询执行
        Args:
            query: 查询语句
            params: 查询参数
            execution_time: 执行时间
            database_type: 数据库类型
            success: 是否成功
            row_count: 返回行数
            error_message: 错误信息
        """
        with self._lock:
            # 更新查询统计
            self._update_query_stats(query, execution_time)
            # 检查是否为慢查询
            if execution_time >= self._slow_query_threshold:
                record = SlowQueryRecord(
                    query=query,
                    params=params,
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    database_type=database_type,
                    success=success,
                    row_count=row_count,
                    error_message=error_message
                )
                self._slow_queries.append(record)
                self._logger.warning(f"慢查询检测: {execution_time:.3f}s - {query[:100]}...")
                # 生成告警
                self._generate_alert(record)

    def get_slow_queries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取慢查询记录
        Args:
            limit: 限制返回数量
        Returns:
            慢查询记录列表
        """
        with self._lock:
            records = list(self._slow_queries)
            if limit:
                records = records[-limit:]
            return [asdict(record) for record in records]

    def get_query_stats(self, query_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        获取查询统计信息
        Args:
            query_pattern: 查询模式过滤
        Returns:
            查询统计信息
        """
        with self._lock:
            if query_pattern:
                filtered_stats = {}
                for query, stats in self._query_stats.items():
                    if query_pattern in query:
                        filtered_stats[query] = stats.copy()
                return filtered_stats
            else:
                return {k: v.copy() for k, v in self._query_stats.items()}

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        Returns:
            性能摘要信息
        """
        with self._lock:
            total_queries = sum(stats['count'] for stats in self._query_stats.values())
            total_time = sum(stats['total_time'] for stats in self._query_stats.values())
            slow_query_count = len(self._slow_queries)
            return {
                'total_queries': total_queries,
                'total_execution_time': total_time,
                'average_execution_time': total_time / total_queries if total_queries > 0 else 0,
                'slow_query_count': slow_query_count,
                'slow_query_threshold': self._slow_query_threshold,
                'unique_queries': len(self._query_stats),
                'alerts_count': len(self._alerts)
            }

    def get_alerts(self, level: Optional[AlertLevel] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取告警信息
        Args:
            level: 告警级别过滤
            limit: 限制返回数量
        Returns:
            告警信息列表
        """
        with self._lock:
            alerts = list(self._alerts)
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            if limit:
                alerts = alerts[-limit:]
            return [asdict(alert) for alert in alerts]

    def set_slow_query_threshold(self, threshold: float) -> None:
        """
        设置慢查询阈值
        Args:
            threshold: 新的阈值（秒）
        """
        self._slow_query_threshold = threshold
        self._logger.info(f"慢查询阈值已更新为 {threshold} 秒")

    def clear_slow_queries(self) -> int:
        """
        清除慢查询记录
        Returns:
            清除的记录数量
        """
        with self._lock:
            count = len(self._slow_queries)
            self._slow_queries.clear()
            return count

    def clear_alerts(self) -> int:
        """
        清除告警记录
        Returns:
            清除的告警数量
        """
        with self._lock:
            count = len(self._alerts)
            self._alerts.clear()
            return count

    def add_alert_callback(self, callback: Callable) -> None:
        """
        添加告警回调函数
        Args:
            callback: 回调函数
        """
        self._alert_callbacks.append(callback)

    def _update_query_stats(self, query: str, execution_time: float) -> None:
        """
        更新查询统计信息
        Args:
            query: 查询语句
            execution_time: 执行时间
        """
        stats = self._query_stats[query]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['last_execution'] = datetime.now()

    def _generate_alert(self, record: SlowQueryRecord) -> None:
        """
        生成告警

        Args:
            record: 慢查询记录
        """
        alert_level = self._determine_alert_level(record)
        alert = self._create_performance_alert(record, alert_level)
        self._store_alert(alert)
        self._trigger_alert_callbacks(alert)

    def _determine_alert_level(self, record: SlowQueryRecord) -> AlertLevel:
        """
        根据执行时间确定告警级别

        Args:
            record: 慢查询记录

        Returns:
            告警级别
        """
        execution_time = record.execution_time
        threshold = self._slow_query_threshold

        if execution_time >= threshold * 5:
            return AlertLevel.CRITICAL
        elif execution_time >= threshold * 3:
            return AlertLevel.ERROR
        elif execution_time >= threshold * 2:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def _create_performance_alert(self, record: SlowQueryRecord, level: AlertLevel) -> PerformanceAlert:
        """
        创建性能告警对象

        Args:
            record: 慢查询记录
            level: 告警级别

        Returns:
            性能告警对象
        """
        # 截断查询字符串以避免过长的告警消息
        query_preview = record.query[:100] + "..." if len(record.query) > 100 else record.query
        return PerformanceAlert(
            level=level,
            message=f"检测到慢查询: {record.execution_time:.3f}秒 - {query_preview}",
            timestamp=record.timestamp,
            details={
                'query': record.query,
                'execution_time': record.execution_time,
                'params': record.params,
                'database_type': record.database_type,
                'success': record.success,
                'row_count': record.row_count,
                'error_message': record.error_message
            }
        )

    def _store_alert(self, alert: PerformanceAlert) -> None:
        """
        存储告警到列表

        Args:
            alert: 告警对象
        """
        self._alerts.append(alert)

    def _trigger_alert_callbacks(self, alert: PerformanceAlert) -> None:
        """
        触发告警回调函数

        Args:
            alert: 告警对象
        """
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._logger.error(f"告警回调函数执行失败: {e}")

    def _setup_logging(self) -> None:
        """设置日志配置"""
        try:
            # 确保logs目录存在
            import os
            os.makedirs('logs', exist_ok=True)

            # 创建慢查询日志处理器
            slow_query_handler = logging.FileHandler('logs/slow_queries.log')
            slow_query_handler.setLevel(logging.WARNING)
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            slow_query_handler.setFormatter(formatter)
            # 添加处理器
            self._logger.addHandler(slow_query_handler)
            self._logger.setLevel(logging.WARNING)
        except Exception as e:
            # 如果日志设置失败，继续运行但不抛出异常
            self._logger.warning(f"Failed to setup slow query logging: {e}")


class MonitoredDatabaseAdapter:
    """带监控的数据库适配器装饰器"""

    def __init__(self, adapter, monitor: SlowQueryMonitor):
        """
        初始化监控适配器
        Args:
            adapter: 数据库适配器
            monitor: 慢查询监控器
        """
        self._adapter = adapter
        self._monitor = monitor

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        执行查询（带监控）
        Args:
            query: 查询语句
            params: 查询参数
        Returns:
            查询结果
        """
        start_time = time.time()
        try:
            # 执行查询
            result = self._adapter.execute_query(query, params)
            # 计算执行时间
            execution_time = time.time() - start_time
            # 记录查询
            self._monitor.record_query(
                query=query,
                params=params,
                execution_time=execution_time,
                database_type=self._adapter.get_connection_info().get('database_type', 'unknown'),
                success=result.success,
                row_count=result.row_count,
                error_message=result.error_message
            )
            return result
        except Exception as e:
            # 记录异常查询
            execution_time = time.time() - start_time
            self._monitor.record_query(
                query=query,
                params=params,
                execution_time=execution_time,
                database_type=self._adapter.get_connection_info().get('database_type', 'unknown'),
                success=False,
                row_count=0,
                error_message=str(e)
            )
            raise

    def __getattr__(self, name):
        """代理其他方法到原始适配器"""
        return getattr(self._adapter, name)
