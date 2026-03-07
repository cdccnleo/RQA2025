#!/usr/bin/env python3
"""
RQA2025 子系统边界优化层
Subsystem Boundary Optimization Layer

优化各子系统间的职责分工和接口标准化。
"""

from .core.boundary_optimizer import (
    BoundaryOptimizer, SubsystemBoundary, InterfaceContract, BoundaryOptimizationResult
)
from .core.unified_service_manager import (
    UnifiedServiceManager, ServiceRegistration, ServiceCall, get_unified_service_manager
)

__version__ = "1.0.0"
__author__ = "RQA2025 Team"

__all__ = [
    # 边界优化器
    'BoundaryOptimizer', 'SubsystemBoundary', 'InterfaceContract', 'BoundaryOptimizationResult',

    # 统一服务管理器
    'UnifiedServiceManager', 'ServiceRegistration', 'ServiceCall', 'get_unified_service_manager'
]
