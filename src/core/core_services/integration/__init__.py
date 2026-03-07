"""
集成服务组件

提供服务集成和管理功能：
- ServiceIntegrationManager: 服务集成管理器
- ServiceRegistry: 服务注册表
- ServiceCommunicator: 服务通信器
- ServiceDiscovery: 服务发现
- CacheManager: 缓存管理器
- ConnectionPool: 连接池
- IntegrationMonitor: 集成监控
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .service_integration_manager import ServiceIntegrationManager
    from .service_registry import ServiceRegistry
    from src.core.integration.services.service_communicator import ServiceCommunicator
    from src.core.integration.services.service_discovery import ServiceDiscoveryClient as ServiceDiscovery
    from .cache_manager import CacheManager
    from .connection_pool import ConnectionPool
    from .integration_monitor import PerformanceMonitor as IntegrationMonitor
    from .service_executor import ServiceExecutor
except ImportError as e:
    # 如果导入失败，提供基础实现
    logger.warning(f"Integration services import failed: {e}")

    class ServiceIntegrationManager:
        pass
    class ServiceRegistry:
        pass
    class ServiceCommunicator:
        pass
    class ServiceDiscovery:
        pass
    class CacheManager:
        pass
    class ConnectionPool:
        pass
    class IntegrationMonitor:
        pass
    class ServiceExecutor:
        pass

__all__ = [
    "ServiceIntegrationManager",
    "ServiceRegistry",
    "ServiceCommunicator",
    "ServiceDiscovery",
    "CacheManager",
    "ConnectionPool",
    "IntegrationMonitor",
    "ServiceExecutor"
]
