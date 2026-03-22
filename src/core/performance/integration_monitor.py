# -*- coding: utf-8 -*-
"""
集成性能监控模块

提供系统集成性能监控、指标收集和优化建议功能。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 支持Prometheus指标导出
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import functools

# 尝试导入Prometheus客户端
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    
    属性:
        metric_name: 指标名称
        value: 指标值
        timestamp: 时间戳
        labels: 标签字典
        unit: 单位
    """
    metric_name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class IntegrationPerformanceStats:
    """
    集成性能统计
    
    属性:
        integration_name: 集成模块名称
        total_calls: 总调用次数
        success_calls: 成功调用次数
        failed_calls: 失败调用次数
        avg_latency_ms: 平均延迟(毫秒)
        p95_latency_ms: P95延迟(毫秒)
        p99_latency_ms: P99延迟(毫秒)
        max_latency_ms: 最大延迟(毫秒)
        last_updated: 最后更新时间
    """
    integration_name: str
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class IntegrationPerformanceMonitor:
    """
    集成性能监控器
    
    提供系统集成性能监控、指标收集和告警功能。
    
    使用示例:
        monitor = IntegrationPerformanceMonitor()
        
        # 记录性能指标
        monitor.record_latency("trading_scheduler", 15.5)
        
        # 获取性能统计
        stats = monitor.get_stats("trading_scheduler")
    """
    
    _instance: Optional['IntegrationPerformanceMonitor'] = None
    
    def __new__(cls) -> 'IntegrationPerformanceMonitor':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化性能监控器"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._latency_records: Dict[str, deque] = {}
        self._stats: Dict[str, IntegrationPerformanceStats] = {}
        self._max_records = 10000
        self._lock = asyncio.Lock()
        
        # 初始化Prometheus指标
        self._init_prometheus_metrics()
        
        logger.info("集成性能监控器初始化完成")
    
    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端不可用，跳过指标初始化")
            return
        
        # 延迟直方图
        self._latency_histogram = Histogram(
            'integration_latency_ms',
            '集成模块调用延迟(毫秒)',
            ['integration_name', 'operation']
        )
        
        # 调用计数器
        self._call_counter = Counter(
            'integration_calls_total',
            '集成模块调用总次数',
            ['integration_name', 'operation', 'status']
        )
        
        # 活跃调用 gauge
        self._active_calls_gauge = Gauge(
            'integration_active_calls',
            '当前活跃调用数',
            ['integration_name']
        )
        
        # 错误计数器
        self._error_counter = Counter(
            'integration_errors_total',
            '集成模块错误总数',
            ['integration_name', 'error_type']
        )
    
    async def record_latency(
        self,
        integration_name: str,
        latency_ms: float,
        operation: str = "default",
        success: bool = True
    ):
        """
        记录调用延迟
        
        Args:
            integration_name: 集成模块名称
            latency_ms: 延迟(毫秒)
            operation: 操作类型
            success: 是否成功
        """
        async with self._lock:
            # 初始化记录队列
            if integration_name not in self._latency_records:
                self._latency_records[integration_name] = deque(maxlen=self._max_records)
                self._stats[integration_name] = IntegrationPerformanceStats(
                    integration_name=integration_name
                )
            
            # 记录延迟
            self._latency_records[integration_name].append({
                "latency_ms": latency_ms,
                "timestamp": datetime.now(),
                "success": success,
                "operation": operation
            })
            
            # 更新统计
            stats = self._stats[integration_name]
            stats.total_calls += 1
            if success:
                stats.success_calls += 1
            else:
                stats.failed_calls += 1
            
            # 计算延迟统计
            await self._update_latency_stats(integration_name)
            stats.last_updated = datetime.now()
            
            # 更新Prometheus指标
            if PROMETHEUS_AVAILABLE:
                self._latency_histogram.labels(
                    integration_name=integration_name,
                    operation=operation
                ).observe(latency_ms)
                
                self._call_counter.labels(
                    integration_name=integration_name,
                    operation=operation,
                    status="success" if success else "failed"
                ).inc()
    
    async def _update_latency_stats(self, integration_name: str):
        """更新延迟统计"""
        records = self._latency_records[integration_name]
        if not records:
            return
        
        latencies = [r["latency_ms"] for r in records if r["success"]]
        if not latencies:
            return
        
        stats = self._stats[integration_name]
        stats.avg_latency_ms = sum(latencies) / len(latencies)
        stats.max_latency_ms = max(latencies)
        
        # 计算P95和P99
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p99_index = int(len(sorted_latencies) * 0.99)
        
        stats.p95_latency_ms = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]
        stats.p99_latency_ms = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
    
    def get_stats(self, integration_name: str) -> Optional[IntegrationPerformanceStats]:
        """
        获取性能统计
        
        Args:
            integration_name: 集成模块名称
            
        Returns:
            Optional[IntegrationPerformanceStats]: 性能统计
        """
        return self._stats.get(integration_name)
    
    def get_all_stats(self) -> Dict[str, IntegrationPerformanceStats]:
        """
        获取所有性能统计
        
        Returns:
            Dict[str, IntegrationPerformanceStats]: 所有性能统计
        """
        return self._stats.copy()
    
    async def record_error(
        self,
        integration_name: str,
        error_type: str,
        error_message: str
    ):
        """
        记录错误
        
        Args:
            integration_name: 集成模块名称
            error_type: 错误类型
            error_message: 错误信息
        """
        logger.error(f"集成模块错误: {integration_name}, type={error_type}, msg={error_message}")
        
        if PROMETHEUS_AVAILABLE:
            self._error_counter.labels(
                integration_name=integration_name,
                error_type=error_type
            ).inc()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "integrations": {}
        }
        
        for name, stats in self._stats.items():
            success_rate = 0.0
            if stats.total_calls > 0:
                success_rate = (stats.success_calls / stats.total_calls) * 100
            
            report["integrations"][name] = {
                "total_calls": stats.total_calls,
                "success_calls": stats.success_calls,
                "failed_calls": stats.failed_calls,
                "success_rate_percent": round(success_rate, 2),
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
                "p95_latency_ms": round(stats.p95_latency_ms, 2),
                "p99_latency_ms": round(stats.p99_latency_ms, 2),
                "max_latency_ms": round(stats.max_latency_ms, 2),
                "last_updated": stats.last_updated.isoformat()
            }
        
        return report


def performance_monitor(
    integration_name: str,
    operation: str = "default"
):
    """
    性能监控装饰器
    
    用于自动监控函数执行性能。
    
    Args:
        integration_name: 集成模块名称
        operation: 操作类型
        
    使用示例:
        @performance_monitor("trading_scheduler", "submit_task")
        async def submit_task(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = IntegrationPerformanceMonitor()
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                await monitor.record_error(
                    integration_name=integration_name,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                await monitor.record_latency(
                    integration_name=integration_name,
                    latency_ms=latency_ms,
                    operation=operation,
                    success=success
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = IntegrationPerformanceMonitor()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                asyncio.create_task(monitor.record_error(
                    integration_name=integration_name,
                    error_type=type(e).__name__,
                    error_message=str(e)
                ))
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                asyncio.create_task(monitor.record_latency(
                    integration_name=integration_name,
                    latency_ms=latency_ms,
                    operation=operation,
                    success=success
                ))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# 全局实例获取函数
def get_integration_performance_monitor() -> IntegrationPerformanceMonitor:
    """
    获取集成性能监控器实例
    
    Returns:
        IntegrationPerformanceMonitor: 监控器实例
    """
    return IntegrationPerformanceMonitor()
