"""
服务治理框架
提供统一的服务治理和生命周期管理功能
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic
from threading import Lock
import logging

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')

class ServiceStatus(Enum):
    """服务状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class ServicePriority(Enum):
    """服务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class IService(ABC):
    """服务接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """服务名称"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """服务版本"""
        pass

    @property
    @abstractmethod
    def status(self) -> ServiceStatus:
        """服务状态"""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """启动服务"""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """停止服务"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

class BaseService(IService, ABC):
    """基础服务类"""

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._status = ServiceStatus.STOPPED
        self._lock = Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def status(self) -> ServiceStatus:
        with self._lock:
            return self._status

    def _set_status(self, status: ServiceStatus):
        with self._lock:
            self._status = status
            logger.info(f"Service {self.name} status changed to {status.value}")

    async def start(self) -> bool:
        """启动服务"""
        try:
            self._set_status(ServiceStatus.STARTING)
            # 子类实现具体的启动逻辑
            await self._start_impl()
            self._set_status(ServiceStatus.RUNNING)
            logger.info(f"Service {self.name} started successfully")
            return True
        except Exception as e:
            self._set_status(ServiceStatus.ERROR)
            logger.error(f"Failed to start service {self.name}: {e}")
            return False

    async def stop(self) -> bool:
        """停止服务"""
        try:
            self._set_status(ServiceStatus.STOPPING)
            # 子类实现具体的停止逻辑
            await self._stop_impl()
            self._set_status(ServiceStatus.STOPPED)
            logger.info(f"Service {self.name} stopped successfully")
            return True
        except Exception as e:
            self._set_status(ServiceStatus.ERROR)
            logger.error(f"Failed to stop service {self.name}: {e}")
            return False

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return await self._health_check_impl()
        except Exception as e:
            logger.error(f"Health check failed for service {self.name}: {e}")
            return False

    @abstractmethod
    async def _start_impl(self) -> None:
        """具体的启动实现"""
        pass

    @abstractmethod
    async def _stop_impl(self) -> None:
        """具体的停止实现"""
        pass

    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """具体的健康检查实现"""
        pass

class ServiceRegistry:
    """服务注册表"""

    def __init__(self):
        self._services: Dict[str, IService] = {}
        self._lock = Lock()

    def register_service(self, service: IService, priority: ServicePriority = ServicePriority.NORMAL) -> bool:
        """注册服务"""
        try:
            with self._lock:
                service_name = service.name
                if service_name in self._services:
                    logger.warning(f"Service {service_name} is already registered")
                    return False

                self._services[service_name] = service
                logger.info(f"Service {service_name} registered with priority {priority.name}")
                return True
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False

    def unregister_service(self, service_name: str) -> bool:
        """注销服务"""
        try:
            with self._lock:
                if service_name not in self._services:
                    logger.warning(f"Service {service_name} is not registered")
                    return False

                del self._services[service_name]
                logger.info(f"Service {service_name} unregistered")
                return True
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}: {e}")
            return False

    def get_service(self, service_name: str) -> Optional[IService]:
        """获取服务"""
        with self._lock:
            return self._services.get(service_name)

    def get_service_count(self) -> int:
        """获取服务数量"""
        with self._lock:
            return len(self._services)

    def list_services(self) -> Dict[str, IService]:
        """列出所有服务"""
        with self._lock:
            return self._services.copy()

# 全局服务注册表实例
_service_registry = ServiceRegistry()

def get_service_registry() -> ServiceRegistry:
    """获取全局服务注册表"""
    return _service_registry

def register_service(service: IService, priority: ServicePriority = ServicePriority.NORMAL) -> bool:
    """注册服务到全局注册表"""
    return _service_registry.register_service(service, priority)

def get_service(service_name: str) -> Optional[IService]:
    """从全局注册表获取服务"""
    return _service_registry.get_service(service_name)

# 重新导出所有类和函数
__all__ = [
    'IService',
    'BaseService',
    'ServiceRegistry',
    'ServiceStatus',
    'ServicePriority',
    'get_service_registry',
    'register_service',
    'get_service'
]
