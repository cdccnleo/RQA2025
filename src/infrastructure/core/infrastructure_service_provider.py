"""
RQA2025 基础设施服务提供者实现

本模块提供基础设施服务提供者的具体实现，
统一管理所有基础设施服务的访问和生命周期。
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from ..interfaces.infrastructure_services import (
    IInfrastructureServiceProvider,
    IConfigManager,
    ICacheService,
    ILogger,
    IMonitor,
    ISecurityManager,
    IHealthChecker,
    IResourceManager,
    IEventBus,
    IServiceContainer,
    InfrastructureServiceStatus,
    HealthCheckResult,
)


class InfrastructureServiceProvider(IInfrastructureServiceProvider):
    """基础设施服务提供者实现"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._shutdown = False
        self._start_time = datetime.now()

    @property
    def config_manager(self) -> IConfigManager:
        """配置管理器"""
        return self._get_service('config_manager')

    @property
    def cache_service(self) -> ICacheService:
        """缓存服务"""
        return self._get_service('cache_service')

    @property
    def logger(self) -> ILogger:
        """日志器"""
        return self._get_service('logger')

    @property
    def monitor(self) -> IMonitor:
        """监控器"""
        return self._get_service('monitor')

    @property
    def security_manager(self) -> ISecurityManager:
        """安全管理器"""
        return self._get_service('security_manager')

    @property
    def health_checker(self) -> IHealthChecker:
        """健康检查器"""
        return self._get_service('health_checker')

    @property
    def resource_manager(self) -> IResourceManager:
        """资源管理器"""
        return self._get_service('resource_manager')

    @property
    def event_bus(self) -> IEventBus:
        """事件总线"""
        return self._get_service('event_bus')

    @property
    def service_container(self) -> IServiceContainer:
        """服务容器"""
        return self._get_service('service_container')

    def _get_service(self, service_name: str) -> Any:
        """获取服务实例"""
        if service_name not in self._services:
            # 延迟初始化服务
            self._initialize_service(service_name)
            # 如果这是第一次初始化服务，设置初始化标志
            if not self._initialized and len(self._services) >= 3:  # 至少有几个服务时认为已初始化
                self._initialized = True
        return self._services[service_name]

    def _initialize_service(self, service_name: str) -> None:
        """初始化单个服务"""
        try:
            if service_name == 'config_manager':
                # 这里应该从具体实现导入
                # from ..config.unified_config_manager import UnifiedConfigManager
                # self._services[service_name] = UnifiedConfigManager()
                self._services[service_name] = MockConfigManager()

            elif service_name == 'cache_service':
                # from ..cache.unified_cache_manager import UnifiedCacheManager
                # self._services[service_name] = UnifiedCacheManager()
                self._services[service_name] = MockCacheService()

            elif service_name == 'logger':
                # from ..logging.unified_logger import UnifiedLogger
                # self._services[service_name] = UnifiedLogger()
                self._services[service_name] = MockLogger()

            elif service_name == 'monitor':
                # from ..monitoring.unified_monitor import UnifiedMonitor
                # self._services[service_name] = UnifiedMonitor()
                self._services[service_name] = MockMonitor()

            elif service_name == 'security_manager':
                self._services[service_name] = MockSecurityManager()

            elif service_name == 'health_checker':
                self._services[service_name] = MockHealthChecker()

            elif service_name == 'resource_manager':
                self._services[service_name] = MockResourceManager()

            elif service_name == 'event_bus':
                self._services[service_name] = MockEventBus()

            elif service_name == 'service_container':
                self._services[service_name] = MockServiceContainer()

        except Exception as e:
            # 服务初始化失败时记录警告
            print(f"Failed to initialize service {service_name}: {e}")
            # 提供降级的mock实现
            self._services[service_name] = MockService()

    def initialize_all_services(self) -> bool:
        """初始化所有基础设施服务"""
        try:
            # 强制初始化所有服务
            service_names = [
                'config_manager', 'cache_service', 'logger', 'monitor',
                'security_manager', 'health_checker', 'resource_manager',
                'event_bus', 'service_container'
            ]

            for service_name in service_names:
                self._get_service(service_name)

            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize all infrastructure services: {e}")
            return False

    def shutdown_all_services(self) -> bool:
        """关闭所有基础设施服务"""
        try:
            for service in self._services.values():
                if hasattr(service, 'shutdown'):
                    service.shutdown()

            self._services.clear()
            self._initialized = False
            self._shutdown = True
            return True

        except Exception as e:
            print(f"Failed to shutdown infrastructure services: {e}")
            return False

    def register_service(self, service_name: str, service_instance: Any) -> bool:
        """注册服务"""
        try:
            self._services[service_name] = service_instance
            return True
        except Exception:
            return False

    def get_service(self, service_name: str) -> Any:
        """获取服务"""
        return self._services.get(service_name)

    def list_services(self) -> List[str]:
        """列出所有已注册的服务"""
        return list(self._services.keys())

    def get_service_status(self) -> InfrastructureServiceStatus:
        """获取基础设施服务整体状态"""
        # 如果已被关闭，返回停止状态
        if self._shutdown:
            return InfrastructureServiceStatus.STOPPED

        # 如果没有初始化过，返回初始化中状态
        if not self._initialized:
            return InfrastructureServiceStatus.INITIALIZING

        # 检查所有服务的健康状态
        healthy_services = 0
        total_services = len(self._services)

        for service in self._services.values():
            if hasattr(service, 'is_healthy'):
                if service.is_healthy():
                    healthy_services += 1
            else:
                # 如果服务没有健康检查方法，假设它是健康的
                healthy_services += 1

        if healthy_services == total_services:
            return InfrastructureServiceStatus.RUNNING
        elif healthy_services > 0:
            return InfrastructureServiceStatus.DEGRADED
        else:
            return InfrastructureServiceStatus.ERROR

    def get_service_health_report(self) -> Dict[str, HealthCheckResult]:
        """获取所有服务的健康报告"""
        report = {}

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'check_health'):
                    result = service.check_health()
                    # 确保服务名称正确
                    result.service_name = service_name
                else:
                    # 默认健康状态
                    result = HealthCheckResult(
                        service_name=service_name,
                        status="healthy",
                        response_time=0.0,
                        message="Service initialized successfully"
                    )
            except Exception as e:
                result = HealthCheckResult(
                    service_name=service_name,
                    status="unhealthy",
                    response_time=0.0,
                    error=str(e)
                )

            report[service_name] = result

        return report


# =============================================================================
# Mock 实现（用于开发和测试阶段）
# =============================================================================

class MockService:
    """Mock 服务基类"""

    def is_healthy(self) -> bool:
        return True

    def check_health(self) -> HealthCheckResult:
        return HealthCheckResult(
            service_name=self.__class__.__name__,
            status="healthy",
            response_time=0.001,
            message="Mock service is healthy"
        )


class MockConfigManager(MockService, IConfigManager):
    """Mock 配置管理器"""

    def __init__(self):
        self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        self._config[key] = value
        return True

    def get_section(self, section: str) -> Dict[str, Any]:
        return {k: v for k, v in self._config.items() if k.startswith(f"{section}.")}

    def reload(self) -> bool:
        return True

    def validate_config(self) -> list:
        return []


class MockCacheService(MockService, ICacheService):
    """Mock 缓存服务"""

    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        return self._cache.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> bool:
        self._cache.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {"total_keys": len(self._cache)}


class MockLogger(MockService, ILogger):
    """Mock 日志器"""

    def debug(self, message: str, **kwargs) -> None:
        print(f"[DEBUG] {message}")

    def info(self, message: str, **kwargs) -> None:
        print(f"[INFO] {message}")

    def warning(self, message: str, **kwargs) -> None:
        print(f"[WARNING] {message}")

    def error(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        print(f"[ERROR] {message}")

    def critical(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        print(f"[CRITICAL] {message}")

    def log(self, level, message: str, **kwargs) -> None:
        print(f"[{level}] {message}")

    def is_enabled_for(self, level) -> bool:
        return True


class MockMonitor(MockService, IMonitor):
    """Mock 监控器"""

    def record_metric(self, name: str, value, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        return f"timer_{name}"

    def stop_timer(self, timer_id: str) -> float:
        return 0.001

    def get_metrics(self, pattern: Optional[str] = None) -> list:
        return []


class MockSecurityManager(MockService, ISecurityManager):
    """Mock 安全管理器"""

    def authenticate(self, username: str, password: str):
        return None

    def validate_token(self, token: str):
        return None

    def authorize(self, token: str, resource: str, action: str) -> bool:
        return True

    def create_user(self, credentials) -> bool:
        return True

    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        return True

    def delete_user(self, username: str) -> bool:
        return True

    def get_user_permissions(self, username: str) -> List[str]:
        return []


class MockHealthChecker(MockService, IHealthChecker):
    """Mock 健康检查器"""

    def get_health_history(self, limit: int = 10) -> List[HealthCheckResult]:
        return []

    def get_detailed_status(self) -> Dict[str, Any]:
        return {"status": "healthy"}


class MockResourceManager(MockService, IResourceManager):
    """Mock 资源管理器"""

    def get_resource_usage(self, resource_type: str):
        return None

    def set_resource_limit(self, resource_type: str, limit, unit: str) -> bool:
        return True

    def check_resource_available(self, resource_type: str, required) -> bool:
        return True

    def allocate_resource(self, resource_type: str, amount) -> bool:
        return True

    def release_resource(self, resource_type: str, amount) -> bool:
        return True

    def get_all_resource_quotas(self) -> Dict[str, Any]:
        return {}


class MockEventBus(MockService, IEventBus):
    """Mock 事件总线"""

    def publish(self, event) -> bool:
        return True

    def subscribe(self, event_type: str, handler):
        return "sub_123"

    def unsubscribe(self, subscription_id: str) -> bool:
        return True

    def publish_async(self, event) -> str:
        return "task_123"

    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100):
        return []


class MockServiceContainer(MockService, IServiceContainer):
    """Mock 服务容器"""

    def register(self, interface: type, implementation: type, singleton: bool = True) -> None:
        pass

    def register_instance(self, interface: type, instance: Any) -> None:
        pass

    def resolve(self, interface: type) -> Any:
        return None

    def has_service(self, interface: type) -> bool:
        return False

    def unregister(self, interface: type) -> bool:
        return True

    def get_registered_services(self) -> List[type]:
        return []


# =============================================================================
# 全局服务提供者实例
# =============================================================================

# 创建全局基础设施服务提供者实例
_infrastructure_provider = InfrastructureServiceProvider()


def get_infrastructure_provider() -> IInfrastructureServiceProvider:
    """获取基础设施服务提供者实例"""
    return _infrastructure_provider


# 别名，便于不同模块使用
def get_infrastructure_service_provider() -> IInfrastructureServiceProvider:
    """获取基础设施服务提供者实例（别名）"""
    return _infrastructure_provider


def initialize_infrastructure() -> bool:
    """初始化基础设施服务"""
    return _infrastructure_provider.initialize_all_services()


def shutdown_infrastructure() -> bool:
    """关闭基础设施服务"""
    return _infrastructure_provider.shutdown_all_services()
