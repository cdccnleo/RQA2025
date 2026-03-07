"""
核心服务层集成 - 接口定义模块

此模块提供系统集成相关的接口定义，包括：
- 集成接口 (IntegrationInterface)
- 适配器接口 (AdapterInterface)
- 层间接口 (LayerInterface)

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .interface import IntegrationInterface
from .interfaces import AdapterInterface
from .layer_interface import LayerInterface

__all__ = [
    'IntegrationInterface',
    'AdapterInterface',
    'LayerInterface'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
