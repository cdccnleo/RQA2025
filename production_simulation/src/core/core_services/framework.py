"""
服务治理框架
提供统一的服务治理和生命周期管理功能
"""

from src.core.service_framework import (
    IService,
    BaseService,
    ServiceRegistry,
    ServiceStatus,
    ServicePriority,
    get_service_registry,
    register_service,
    get_service
)

# 重新导出所有类和函数
__all__ = [
    'IService',
    'BaseService',
    'ServiceRegistry',
    'ServiceStatus',
    'ServicePriority',
    'get_service_registry',
    'register_service',
    'get_service'
]
