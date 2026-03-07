"""
核心服务层集成 - 中间件组件模块

此模块提供系统集成中间件组件，包括：
- 中间件组件 (MiddlewareComponent)
- 桥接组件 (BridgeComponent)
- 连接器组件 (ConnectorComponent)

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .middleware_components import MiddlewareComponent
from .bridge_components import BridgeComponent
from .connector_components import ConnectorComponent

__all__ = [
    'MiddlewareComponent',
    'BridgeComponent',
    'ConnectorComponent'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
