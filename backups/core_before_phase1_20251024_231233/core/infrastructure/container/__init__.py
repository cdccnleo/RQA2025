"""
核心服务层基础设施 - 依赖注入容器模块

此模块提供依赖注入容器的核心功能，包括：
- 依赖注入容器 (DependencyContainer)
- 服务注册和解析
- 生命周期管理
- 容器组件和工厂

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .container import DependencyContainer, ServiceDescriptor, ServiceStatus
from .container_components import ContainerComponent
from .factory_components import FactoryComponent
from .locator_components import ServiceLocator
from .registry_components import ServiceRegistry
from .resolver_components import DependencyResolver
from .unified_container_interface import IContainer

__all__ = [
    'DependencyContainer',
    'ServiceDescriptor',
    'ServiceStatus',
    'ContainerComponent',
    'FactoryComponent',
    'ServiceLocator',
    'ServiceRegistry',
    'DependencyResolver',
    'IContainer'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
