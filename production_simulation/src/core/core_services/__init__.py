"""
服务层子系统

统一的服务组件管理，包括：
├── core/         # 核心业务服务
├── security/     # 安全服务
├── integration/  # 集成服务
├── infrastructure/ # 基础设施服务
├── utils/        # 工具服务
└── api/          # API服务
"""

# 服务治理框架 - 核心组件
from .framework import (
    IService,
    BaseService,
    ServiceRegistry,
    ServiceStatus,
    ServicePriority,
    get_service_registry,
    register_service,
    get_service
)

# 子模块导入 - 按需导入避免循环依赖
# from .core import BusinessService, DatabaseService, StrategyManager
# from .security import AuthenticationService, EncryptionService, WebManagementService
# from .integration import ServiceIntegrationManager, ServiceCommunicator, ServiceDiscovery
# from .infrastructure import ServiceContainer
# from .utils import ServiceFactory
# from .api import APIService, APIGateway

__all__ = [
    # 服务治理框架 - 核心导出
    "IService",
    "BaseService",
    "ServiceRegistry",
    "ServiceStatus",
    "ServicePriority",
    "get_service_registry",
    "register_service",
    "get_service",

    # 子模块 - 按需导入
    # "BusinessService", "DatabaseService", "StrategyManager",
    # "AuthenticationService", "EncryptionService", "WebManagementService",
    # "ServiceIntegrationManager", "ServiceCommunicator", "ServiceDiscovery",
    # "ServiceContainer", "ServiceFactory", "APIService", "APIGateway"
]
