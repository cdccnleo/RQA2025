"""
弹性模块

Resilience module providing graceful degradation and fault tolerance capabilities.
"""

from .degradation.graceful_degradation import (
    GracefulDegradationManager,
    CircuitBreaker,
    ServiceHealthChecker
)

# 别名
GracefulDegradation = GracefulDegradationManager

__all__ = [
    'GracefulDegradationManager',
    'GracefulDegradation',  # 别名
    'CircuitBreaker', 
    'ServiceHealthChecker'
]
