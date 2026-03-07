"""
核心服务层集成 - 核心集成组件模块

此模块提供系统集成的核心组件，包括：
- 系统集成管理器 (SystemIntegrationManager)
- 业务适配器工厂 (UnifiedBusinessAdapterFactory)
- 集成组件基类 (IntegrationComponent)

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .system_integration_manager import SystemIntegrationManager
from .business_adapters import UnifiedBusinessAdapterFactory
from .integration_components import IntegrationComponent

__all__ = [
    'SystemIntegrationManager',
    'UnifiedBusinessAdapterFactory',
    'IntegrationComponent'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
