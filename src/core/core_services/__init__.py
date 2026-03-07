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

import logging

logger = logging.getLogger(__name__)

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
try:
    # 核心业务服务
    from .core.business_service import (
        BusinessProcess,
        BusinessProcessStatus,
        BusinessProcessType,
        TradingStrategy,
        StrategyService,
        OrderService,
        PortfolioService,
        ProcessService,
        DataAnalysisService,
        BusinessService
    )
    from .core.database_service import DatabaseService
    from .core.strategy_manager import Strategy, StrategyManager

    # 集成服务
    from .integration.service_integration_manager import ServiceIntegrationManager
    from .integration.service_registry import ServiceRegistry as ServiceRegistryImpl
    from .integration import ServiceCommunicator, ServiceDiscovery

    # API服务
    from .api.api_service import APIService, APIGateway

    _submodules_available = True
    logger.info("Core services submodules imported successfully")
except ImportError as e:
    _submodules_available = False
    logger.warning(f"Core services submodules import failed: {e}")

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
    "BusinessService", "DatabaseService", "StrategyManager",
    "ServiceIntegrationManager", "ServiceCommunicator", "ServiceDiscovery",
    "APIService", "APIGateway"
]
