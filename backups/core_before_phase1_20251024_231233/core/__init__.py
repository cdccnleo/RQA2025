"""
核心服务层 (Core Services Layer)

提供系统核心服务：事件总线、依赖注入、业务流程编排
"""

import logging
from typing import Dict, Any, List, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 默认类定义（在模块级别可见）


class BusinessProcessState:
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class EventType:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EventPriority:
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Lifecycle:
    """组件生命周期管理"""
    CREATED = "created"
    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


# 尝试导入实际实现（如果可用）
try:
    from .orchestration.business_process_orchestrator import BusinessProcessOrchestrator as RealOrchestrator
    BusinessProcessOrchestrator = RealOrchestrator  # noqa: F811
except ImportError:
    logger.warning("Using fallback BusinessProcessOrchestrator implementation")

try:
    from .orchestration.business_process_orchestrator import BusinessProcessState as RealState
    BusinessProcessState = RealState  # noqa: F811
except ImportError:
    logger.warning("Using fallback BusinessProcessState implementation")

try:
    from .orchestration.business_process_orchestrator import EventType as RealEventType
    EventType = RealEventType  # noqa: F811
except ImportError:
    logger.warning("Using fallback EventType implementation")

try:
    from .event_bus.event_bus import EventPriority as RealEventPriority
    EventPriority = RealEventPriority  # noqa: F811
except ImportError:
    logger.warning("Using fallback EventPriority implementation")

try:
    from .examples.architecture_demo import ArchitectureDemo as RealDemo
    ArchitectureDemo = RealDemo  # noqa: F811
except ImportError:
    logger.warning("Using fallback ArchitectureDemo implementation")

# 提供基础实现，避免导入错误


class EventBus:
    """事件总线基础实现"""

    def __init__(self):
        self.name = "EventBus"


class DependencyContainer:
    """依赖注入容器基础实现"""

    def __init__(self):
        self.name = "DependencyContainer"


class ServiceContainer:
    """服务容器基础实现"""

    def __init__(self):
        self.name = "ServiceContainer"


class InterfaceFactory:
    """接口工厂基础实现"""
    @staticmethod
    def register_interface(name, interface): pass


class CoreServicesLayer:
    """核心服务层基础实现"""

    def __init__(self): self.name = "CoreServicesLayer"


    # 尝试导入实际实现
try:
    from .event_bus.event_bus import EventBus as RealEventBus
    EventBus = RealEventBus  # noqa: F811
except ImportError:
    logger.warning("Using fallback EventBus implementation")

try:
    from .container import DependencyContainer as RealDependencyContainer
    DependencyContainer = RealDependencyContainer  # noqa: F811
except ImportError:
    logger.warning("Using fallback DependencyContainer implementation")

try:
    from .business_process_orchestrator import BusinessProcessOrchestrator as RealOrchestrator
    BusinessProcessOrchestrator = RealOrchestrator  # noqa: F811
except ImportError:
    logger.warning("Using fallback BusinessProcessOrchestrator implementation")

try:
    from .event_bus.event_bus import Event as RealEvent
    Event = RealEvent
except ImportError:
    logger.warning("Using fallback Event implementation")

    class Event:
        def __init__(self, event_type, data=None):
            self.event_type = event_type
            self.data = data or {}

try:
    from .service_container import ServiceContainer as RealServiceContainer
    ServiceContainer = RealServiceContainer  # noqa: F811
except ImportError:
    logger.warning("Using fallback ServiceContainer implementation")

# 可视化工具
try:
    from .visualization import BacktestVisualizer
    _visualization_available = True
except ImportError:
    logger.warning("Visualization tools not available")
    _visualization_available = False

__all__ = [
    'EventBus',
    'DependencyContainer',
    'BusinessProcessOrchestrator',
    'InterfaceFactory',
    'CoreServicesLayer',
    'Event',
    'ServiceContainer',
    'EventType',
    'EventPriority',
    'Lifecycle'
]

# 条件性添加可视化工具到导出列表
if _visualization_available:
    __all__.append('BacktestVisualizer')

# 尝试导入API服务
try:
    from .services.api_service import TradingAPIService as APIService
    from .services.api_service import create_trading_api_app
    _api_service_available = True
except ImportError:
    logger.warning("APIService not available, using fallback")
    _api_service_available = False

    class APIService:
        def __init__(self):
            self.name = "APIService (fallback)"

    def create_trading_api_app():
        return None

# 如果API服务可用，添加到导出列表
if _api_service_available:
    __all__.extend(['APIService', 'create_trading_api_app'])
