
# 导入基础设施层核心服务
import logging

logger = logging.getLogger(__name__)

try:
    from src.infrastructure.cache.core.disk_cache_manager import DiskCacheManager
except ImportError:
    DiskCacheManager = None

try:
    from src.infrastructure.cache.manager.memory_cache_manager import MemoryCacheManager
except ImportError:
    MemoryCacheManager = None

try:
    from src.infrastructure.error.handlers.error_handler import ErrorHandler
except ImportError:
    ErrorHandler = None

try:
    from src.infrastructure.health.components.health_checker import HealthChecker
except ImportError:
    HealthChecker = None

try:
    from src.infrastructure.logging.core.unified_logging_interface import Logger
except ImportError:
    Logger = None

try:
    from src.infrastructure.monitoring.components.automation_monitor import AutomationMonitor
except ImportError:
    AutomationMonitor = None
from ..core.config_manager_complete import UnifiedConfigManager
from typing import Dict, Any
try:
    from src.infrastructure.utils.adapters.unified_database_manager import UnifiedDatabaseManager
except ImportError:
    UnifiedDatabaseManager = None

try:
    from container import get_container
except ImportError:
    get_container = None

try:
    from deployment_validator import DeploymentValidator
except ImportError:
    DeploymentValidator = None

try:
    from service_launcher import ServiceLauncher
except ImportError:
    ServiceLauncher = None
"""
基础设施层服务注册

提供基础设施层所有核心服务的注册和配置。
"""


class InfrastructureServiceRegistry:
    """
    service_registry - 配置管理

    职责说明：
    负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

    核心职责：
    - 配置文件的读取和解析
    - 配置参数的验证
    - 配置的热重载
    - 配置的分发和同步
    - 环境变量管理
    - 配置加密和安全

    相关接口：
    - IConfigComponent
    - IConfigManager
    - IConfigValidator
    """
    """基础设施层服务注册器"""

    def __init__(self):
        self.container = get_container() if get_container is not None else None
        self._registered = False

    def register_all_services(self) -> None:
        """注册所有基础设施层服务"""
        if self._registered:
            return
        
        # 检查容器是否可用
        if self.container is None:
            return

        # 配置管理服务
        self._register_config_services()

        # 数据库服务
        self._register_database_services()

        # 缓存服务
        self._register_cache_services()

        # 监控服务
        self._register_monitoring_services()

        # 错误处理服务
        self._register_error_services()

        # 日志服务
        self._register_logging_services()

        # 健康检查服务
        self._register_health_services()

        # 部署服务
        self._register_deployment_services()

        self._registered = True

    def _register_config_services(self) -> None:
        """注册配置管理服务"""
        if self.container is None:
            return
        try:
            # 配置管理器 - 单例
            self.container.register_singleton(
                UnifiedConfigManager,
                factory=lambda container: UnifiedConfigManager()
            )
        except Exception as e:
            logger.warning(f"Failed to register config services: {e}")
            # 继续执行，不抛出异常

    def _register_database_services(self) -> None:
        """注册数据库服务"""
        if self.container is None:
            return
        # 数据库管理器 - 单例
        def create_database_manager(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return UnifiedDatabaseManager(config_manager=config_manager)

        self.container.register_singleton(
            UnifiedDatabaseManager,
            factory=create_database_manager
        )

    def _register_cache_services(self) -> None:
        """注册缓存服务"""
        if self.container is None:
            return
        # 内存缓存管理器 - 单例
        def create_memory_cache_manager(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return MemoryCacheManager(
                max_size=config_manager.get('cache.memory.max_size', 1000),
                ttl=config_manager.get('cache.memory.ttl', 600)
            )

        self.container.register_singleton(
            MemoryCacheManager,
            factory=create_memory_cache_manager
        )

        # 磁盘缓存管理器 - 单例
        def create_disk_cache_manager(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return DiskCacheManager(
                cache_dir=config_manager.get('cache.disk.directory', './cache'),
                max_size=config_manager.get('cache.disk.max_size', 1000000)
            )

        self.container.register_singleton(
            DiskCacheManager,
            factory=create_disk_cache_manager
        )

    def _register_monitoring_services(self) -> None:
        """注册监控服务"""
        if self.container is None:
            return
        # 自动化监控器 - 单例
        def create_automation_monitor(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return AutomationMonitor(
                config=config_manager.get('monitoring.automation', {})
            )

        self.container.register_singleton(
            AutomationMonitor,
            factory=create_automation_monitor
        )

    def _register_error_services(self) -> None:
        """注册错误处理服务"""
        if self.container is None:
            return
        # 错误处理器 - 单例
        def create_error_handler(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return ErrorHandler(
                config=config_manager.get('error.handler', {})
            )

        self.container.register_singleton(
            ErrorHandler,
            factory=create_error_handler
        )

    def _register_logging_services(self) -> None:
        """注册日志服务"""
        if self.container is None:
            return
        # 日志管理器 - 单例
        def create_logger(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return Logger(
                config=config_manager.get('logging', {})
            )

        self.container.register_singleton(
            Logger,
            factory=create_logger
        )

    def _register_health_services(self) -> None:
        """注册健康检查服务"""
        if self.container is None:
            return
        # 健康检查器 - 单例
        def create_health_checker(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return HealthChecker(config_manager=config_manager)

        self.container.register_singleton(
            HealthChecker,
            factory=create_health_checker
        )

    def _register_deployment_services(self) -> None:
        """注册部署服务"""
        if self.container is None:
            return
        # 服务启动器 - 单例
        def create_service_launcher(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return ServiceLauncher(
                config=config_manager.get('deployment.service_launcher', {})
            )

        self.container.register_singleton(
            ServiceLauncher,
            factory=create_service_launcher
        )

        # 部署验证器 - 单例
        def create_deployment_validator(container):
            config_manager = container.resolve(UnifiedConfigManager)
            return DeploymentValidator(
                config=config_manager.get('deployment.validator', {})
            )

        self.container.register_singleton(
            DeploymentValidator,
            factory=create_deployment_validator
        )

    def get_service(self, service_type: type) -> Any:
        """获取服务实例"""
        if self.container is None:
            return None
        try:
            return self.container.resolve(service_type)
        except Exception as e:
            logger.warning(f"Failed to resolve service {service_type}: {e}")
            return None

    def has_service(self, service_type: type) -> bool:
        """检查服务是否已注册"""
        if self.container is None:
            return False
        try:
            return self.container.has_service(service_type)
        except Exception as e:
            logger.warning(f"Failed to check service {service_type}: {e}")
            return False

    def get_registered_services(self) -> Dict[type, Any]:
        """获取所有已注册的服务"""
        if self.container is None:
            return {}
        return self.container.get_registered_services()


# 全局服务注册器实例 - 延迟初始化
_service_registry = None

def _get_or_create_registry():
    global _service_registry
    if _service_registry is None:
        _service_registry = InfrastructureServiceRegistry()
    return _service_registry


def get_service_registry() -> InfrastructureServiceRegistry:
    """获取全局服务注册器"""
    return _get_or_create_registry()


def register_infrastructure_services() -> None:
    """注册所有基础设施层服务"""
    registry = _get_or_create_registry()
    registry.register_all_services()


def get_service(service_type: type) -> Any:
    """获取服务实例"""
    registry = _get_or_create_registry()
    return registry.get_service(service_type)


def has_service(service_type: type) -> bool:
    """检查服务是否已注册"""
    registry = _get_or_create_registry()
    return registry.has_service(service_type)




