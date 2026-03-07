"""
服务通信器模块别名（向后兼容）

这是一个别名导入文件，提供向后兼容的导入路径。
实际实现在 src/core/integration/services/service_communicator.py

使用方式：
    # 新的推荐导入方式
    from src.core.integration.services.service_communicator import ServiceCommunicator
    
    # 向后兼容的导入方式（仍然支持）
    from src.core.integration.service_communicator import ServiceCommunicator

注意：这不是代码重复，而是导入代理模式
创建时间: 2025-08-24
更新时间: 2025-11-03
"""

from .services.service_communicator import (
    ServiceCommunicator,
    ServiceEndpoint,
    Message,
    ServiceRegistry
)

__all__ = [
    'ServiceCommunicator',
    'ServiceEndpoint', 
    'Message',
    'ServiceRegistry'
]

