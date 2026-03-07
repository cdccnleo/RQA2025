"""
decorators 模块

提供 decorators 相关功能和接口。
"""

import logging

# -*- coding: utf-8 -*-
import time
import weakref

from functools import wraps
from prometheus_client import Histogram, Counter, CollectorRegistry, REGISTRY
from typing import Optional, Dict, Any
"""
基础设施层 - 日志系统组件

decorators 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr / bin / env python
"""
监控装饰器模块
提供方法级别的资源监控装饰器
"""

# 缓存指标以防止重复注册
_metric_cache = weakref.WeakKeyDictionary()

logger = logging.getLogger(__name__)


def monitor_performance(operation_name: Optional[str] = None, registry: Optional[CollectorRegistry] = None):
    """
    性能监控装饰器
    Args:
        operation_name: 操作名称，默认为函数名
        registry: Prometheus CollectorRegistry（可选，测试用）
    """
    _registry = registry if registry is not None else REGISTRY

    if _registry not in _metric_cache:
        _metric_cache[_registry] = {}
    cache = _metric_cache[_registry]

    if 'performance_duration_seconds' not in cache:
        cache['performance_duration_seconds'] = Histogram(
            'performance_duration_seconds',
            'Time spent on operation',
            ['operation'],
            buckets=(0.01, 0.1, 0.5, 1, 5, 10, float('inf')),
            registry=_registry
        )

    performance_histogram = cache['performance_duration_seconds']

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            op_name = operation_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_histogram.labels(operation=op_name).observe(duration)
                logger.debug(f"Performance monitored - {op_name}: duration={duration:.3f}s")
        return wrapper
    return decorator


def monitor_errors(error_types: Optional[list] = None, registry: Optional[CollectorRegistry] = None):
    """
    错误监控装饰器
    Args:
        error_types: 要监控的错误类型列表，默认为所有异常
        registry: Prometheus CollectorRegistry（可选，测试用）
    """
    _registry = registry if registry is not None else REGISTRY

    if _registry not in _metric_cache:
        _metric_cache[_registry] = {}
    cache = _metric_cache[_registry]

    if 'error_count_total' not in cache:
        cache['error_count_total'] = Counter(
            'error_count_total',
            'Total number of errors',
            ['function', 'error_type'],
            registry=_registry
        )

    error_counter = cache['error_count_total']

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_type = e.__class__.__name__
                if error_types is None or error_type in error_types:
                    error_counter.labels(
                        function=func.__name__,
                        error_type=error_type
                    ).inc()
                    logger.error(f"Error monitored - {func.__name__}: {error_type}")
                raise
        return wrapper
    return decorator


def monitor_resource(resource_type: str, labels: Optional[Dict] = None, registry: Optional[CollectorRegistry] = None):
    """
    资源监控装饰器 - 重构后的统一入口

    将指标创建和管理逻辑分离到专门的方法中
    """
    # 获取监控指标
    metrics = _get_resource_metrics(registry)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return _execute_with_monitoring(func, args, kwargs, resource_type, metrics)

        return wrapper

    return decorator


def _get_resource_metrics(registry: Optional[CollectorRegistry] = None):
    """
    获取或创建资源监控指标

    使用单例模式确保每个registry只创建一次指标
    """
    _registry = registry if registry is not None else REGISTRY

    # 从缓存获取或创建指标
    if _registry not in _metric_cache:
        _metric_cache[_registry] = _create_resource_metrics(_registry)

    return _metric_cache[_registry]


def _create_resource_metrics(registry: CollectorRegistry) -> Dict[str, Any]:
    """创建资源监控指标"""
    return {
        'usage_histogram': Histogram(
            'resource_usage_seconds',
            'Time spent processing resource',
            ['resource_type', 'operation'],
            buckets=(0.01, 0.1, 0.5, 1, 5, 10, float('inf')),
            registry=registry
        ),
        'error_counter': Counter(
            'resource_errors_total',
            'Total number of resource processing errors',
            ['resource_type', 'operation', 'error_type'],
            registry=registry
        )
    }


def _execute_with_monitoring(func, args, kwargs, resource_type: str, metrics: Dict[str, Any]):
    """
    执行函数并进行监控

    分离监控逻辑和业务逻辑
    """
    operation = func.__name__
    start_time = time.time()
    status = 'success'

    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        status = 'failed'
        _record_error(metrics, resource_type, operation, e)
        raise
    finally:
        duration = time.time() - start_time
        _record_duration(metrics, resource_type, operation, duration, status)


def _record_error(metrics: Dict[str, Any], resource_type: str, operation: str, error: Exception):
    """记录错误指标"""
    error_type = error.__class__.__name__
    metrics['error_counter'].labels(
        resource_type=resource_type,
        operation=operation,
        error_type=error_type
    ).inc()


def _record_duration(metrics: Dict[str, Any], resource_type: str, operation: str, duration: float, status: str):
    """记录执行时间指标"""
    metrics['usage_histogram'].labels(
        resource_type=resource_type,
        operation=operation
    ).observe(duration)

    logger.debug(
        f"Resource monitored - {resource_type}.{operation}: "
        f"status={status}, duration={duration:.3f}s"
    )
