"""
RQA2025 基础设施层工具系统 - 核心组件模块

本模块包含基础设施层工具系统的核心组件和接口定义。

包含的核心组件:
- 基础组件 (BaseComponent)
- 组件接口 (IResourceComponent, IDatabaseAdapter)
- 异常处理 (InfrastructureError, ConfigurationError)
- 统一工厂 (ComponentFactory)
- 重复代码治理 (InfrastructureStatusManager, BaseComponentWithStatus)

作者: RQA2025 Team
创建日期: 2025年9月27日
"""

from .base_components import *
from .error import UnifiedErrorHandler, get_error_handler
from .exceptions import *
from .interfaces import *
from .storage import StorageAdapter
from .duplicate_resolver import (
    InfrastructureStatusManager,
    BaseComponentWithStatus,
    InfrastructureDuplicateResolver
)

__all__ = [
    # 基础组件
    "IBaseComponent",
    "BaseComponent",
    "BaseComponentConstants",
    "BaseComponentFactory",
    # 接口定义
    "IDatabaseAdapter",
    "IResourceComponent",
    "ConnectionStatus",
    "QueryResult",
    "WriteResult",
    "HealthCheckResult",
    # 异常类
    "InfrastructureError",
    "ConfigurationError",
    "DataProcessingError",
    # 错误处理
    "UnifiedErrorHandler",
    "get_error_handler",
    # 存储适配器
    "StorageAdapter",
    # 重复代码治理
    "InfrastructureStatusManager",
    "BaseComponentWithStatus",
    "InfrastructureDuplicateResolver",
]
