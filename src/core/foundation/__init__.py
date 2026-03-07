"""
核心服务层 - 基础组件模块

此模块提供核心服务层的基础组件，包括：
- 基础组件类 (BaseComponent, StandardComponent)
- 异常定义 (CoreException, CoreServiceException等)
- 接口抽象 (LayerInterface, StandardInterface等)

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .base import BaseComponent, ComponentStatus, ComponentHealth
from .exceptions.core_exceptions import (
    CoreException,
    CoreServiceException,
    BusinessProcessError,
    OrchestratorException,
    EventBusException,
    ContainerException
)
from .exceptions.unified_exceptions import (
    RQA2025Exception,
    ValidationError,
    DatabaseError,
    NetworkError
)
from .interfaces.layer_interfaces import LayerInterface
try:
    from .interfaces.standard_interfaces import StandardInterface
except ImportError:
    # 提供基础定义
    class StandardInterface:
        pass

__all__ = [
    'BaseComponent',
    'ComponentStatus',
    'ComponentHealth',
    'CoreException',
    'CoreServiceException',
    'BusinessProcessError',
    'OrchestratorException',
    'EventBusException',
    'ContainerException',
    'RQA2025Exception',
    'ValidationError',
    'DatabaseError',
    'NetworkError',
    'LayerInterface',
    'StandardInterface'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
