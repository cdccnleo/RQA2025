"""
Manager组件别名模块

提供向后兼容的导入路径
实际实现在 core.orchestration.business_process.manager_components 中
"""

from ..orchestration.business_process.manager_components import (
    IManagerComponent,
    ComponentFactory
)

__all__ = ['IManagerComponent', 'ComponentFactory']

