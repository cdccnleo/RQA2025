#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监控装饰器模块
提供方法级别的资源监控装饰器
"""

import time
from functools import wraps
from typing import Callable, Optional, Dict, Any
import logging
from prometheus_client import Histogram, Counter

logger = logging.getLogger(__name__)

# 定义Prometheus指标
RESOURCE_USAGE_HISTOGRAM = Histogram(
    'resource_usage_seconds',
    'Time spent processing resource',
    ['resource_type', 'operation'],
    buckets=(0.01, 0.1, 0.5, 1, 5, 10, float('inf'))
)

RESOURCE_ERROR_COUNTER = Counter(
    'resource_errors_total',
    'Total number of resource processing errors',
    ['resource_type', 'operation', 'error_type']
)

def monitor_resource(resource_type: str, labels: Optional[Dict] = None):
    """
    资源监控装饰器

    Args:
        resource_type: 资源类型标识
        labels: 额外的监控标签

    Example:
        @monitor_resource('json_loader')
        def load_json_file(path):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation = func.__name__
            labels_dict = labels or {}

            # 记录开始时间和初始化状态
            start_time = time.time()
            status = 'success'

            try:
                # 执行被装饰的方法
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # 记录错误状态和类型
                status = 'failed'
                error_type = e.__class__.__name__
                RESOURCE_ERROR_COUNTER.labels(
                    resource_type=resource_type,
                    operation=operation,
                    error_type=error_type
                ).inc()
                raise
            finally:
                # 记录执行时间
                duration = time.time() - start_time
                RESOURCE_USAGE_HISTOGRAM.labels(
                    resource_type=resource_type,
                    operation=operation
                ).observe(duration)

                # 记录日志
                logger.debug(
                    f"Resource monitored - {resource_type}.{operation}: "
                    f"status={status}, duration={duration:.3f}s"
                )
        return wrapper
    return decorator
