"""
服务容器别名文件

提供对infrastructure.container模块的别名导入，保持向后兼容性。

真正的实现位于: src/core/infrastructure/container/
"""

from .container import (
    DependencyContainer,
    ServiceLifecycle as Lifecycle,
    ServiceStatus as ServiceHealth
)

__all__ = [
    'DependencyContainer',
    'Lifecycle',
    'ServiceHealth'
]

