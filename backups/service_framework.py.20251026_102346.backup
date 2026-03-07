"""
服务治理框架

提供统一的服务治理和生命周期管理功能。
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Type
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ServicePriority(Enum):
    """服务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ServiceInfo:
    """服务信息"""
    name: str
    service_class: Type
    priority: ServicePriority = ServicePriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.STOPPED
    start_time: Optional[float] = None
    error_message: Optional[str] = None


class IService(ABC):
    """服务接口"""

    @abstractmethod
    def start(self) -> bool:
        """启动服务"""

    @abstractmethod
    def stop(self) -> bool:
        """停止服务"""

    @abstractmethod
    def is_running(self) -> bool:
        """检查服务是否运行中"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""


class BaseService(IService):
    """基础服务类"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._running = False
        self._start_time = None
        self._lock = threading.RLock()

    def start(self) -> bool:
        """启动服务"""
        with self._lock:
            if self._running:
                logger.warning(f"服务 {self.name} 已经在运行中")
                return True

            try:
                logger.info(f"正在启动服务 {self.name}")
                if self._do_start():
                    self._running = True
                    self._start_time = time.time()
                    logger.info(f"服务 {self.name} 启动成功")
                    return True
                else:
                    logger.error(f"服务 {self.name} 启动失败")
                    return False
            except Exception as e:
                logger.error(f"服务 {self.name} 启动异常: {e}")
                return False

    def stop(self) -> bool:
        """停止服务"""
        with self._lock:
            if not self._running:
                logger.warning(f"服务 {self.name} 已经停止")
                return True

            try:
                logger.info(f"正在停止服务 {self.name}")
                if self._do_stop():
                    self._running = False
                    logger.info(f"服务 {self.name} 停止成功")
                    return True
                else:
                    logger.error(f"服务 {self.name} 停止失败")
                    return False
            except Exception as e:
                logger.error(f"服务 {self.name} 停止异常: {e}")
                return False

    def is_running(self) -> bool:
        """检查服务是否运行中"""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0

            return {
                "name": self.name,
                "running": self._running,
                "uptime": uptime,
                "config": self.config,
                "health": self.health_check()
            }

    def health_check(self) -> bool:
        """健康检查"""
        return self._running

    @abstractmethod
    def _do_start(self) -> bool:
        """实际的启动逻辑"""

    @abstractmethod
    def _do_stop(self) -> bool:
        """实际的停止逻辑"""


class ServiceRegistry:
    """服务注册表"""

    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._instances: Dict[str, IService] = {}
        self._lock = threading.RLock()

    def register_service(self, name: str, service_class: Type[IService],
                         priority: ServicePriority = ServicePriority.NORMAL,
                         dependencies: List[str] = None,
                         config: Dict[str, Any] = None) -> bool:
        """
        注册服务

        Args:
            name: 服务名称
            service_class: 服务类
            priority: 优先级
            dependencies: 依赖的服务列表
            config: 服务配置

        Returns:
            注册是否成功
        """
        with self._lock:
            if name in self._services:
                logger.warning(f"服务 {name} 已经注册")
                return False

            self._services[name] = ServiceInfo(
                name=name,
                service_class=service_class,
                priority=priority,
                dependencies=dependencies or [],
                config=config or {}
            )

            logger.info(f"服务 {name} 注册成功")
            return True

    def unregister_service(self, name: str) -> bool:
        """
        注销服务

        Args:
            name: 服务名称

        Returns:
            注销是否成功
        """
        with self._lock:
            if name not in self._services:
                logger.warning(f"服务 {name} 未注册")
                return False

            # 停止服务实例
            if name in self._instances:
                self._instances[name].stop()
                del self._instances[name]

            del self._services[name]
            logger.info(f"服务 {name} 注销成功")
            return True

    def get_service(self, name: str) -> Optional[IService]:
        """
        获取服务实例

        Args:
            name: 服务名称

        Returns:
            服务实例
        """
        with self._lock:
            if name not in self._instances:
                if name not in self._services:
                    logger.error(f"服务 {name} 未注册")
                    return None

                # 创建服务实例
                service_info = self._services[name]
                try:
                    instance = service_info.service_class(
                        name=name,
                        config=service_info.config
                    )
                    self._instances[name] = instance
                    service_info.status = ServiceStatus.STOPPED
                except Exception as e:
                    logger.error(f"创建服务 {name} 实例失败: {e}")
                    service_info.status = ServiceStatus.ERROR
                    service_info.error_message = str(e)
                    return None

            return self._instances[name]

    def start_service(self, name: str) -> bool:
        """
        启动服务

        Args:
            name: 服务名称

        Returns:
            启动是否成功
        """
        service = self.get_service(name)
        if not service:
            return False

        service_info = self._services[name]
        service_info.status = ServiceStatus.STARTING

        if service.start():
            service_info.status = ServiceStatus.RUNNING
            service_info.start_time = time.time()
            return True
        else:
            service_info.status = ServiceStatus.ERROR
            return False

    def stop_service(self, name: str) -> bool:
        """
        停止服务

        Args:
            name: 服务名称

        Returns:
            停止是否成功
        """
        service = self.get_service(name)
        if not service:
            return False

        service_info = self._services[name]
        service_info.status = ServiceStatus.STOPPING

        if service.stop():
            service_info.status = ServiceStatus.STOPPED
            return True
        else:
            service_info.status = ServiceStatus.ERROR
            return False

    def start_all_services(self) -> Dict[str, bool]:
        """
        启动所有服务（按依赖关系和优先级）

        Returns:
            服务启动结果
        """
        results = {}

        # 按优先级和依赖关系排序
        sorted_services = self._sort_services_by_dependencies()

        for service_name in sorted_services:
            results[service_name] = self.start_service(service_name)

        return results

    def stop_all_services(self) -> Dict[str, bool]:
        """
        停止所有服务

        Returns:
            服务停止结果
        """
        results = {}

        # 按相反顺序停止
        sorted_services = self._sort_services_by_dependencies()
        sorted_services.reverse()

        for service_name in sorted_services:
            results[service_name] = self.stop_service(service_name)

        return results

    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取服务状态

        Args:
            name: 服务名称

        Returns:
            服务状态信息
        """
        service = self.get_service(name)
        if not service:
            return None

        status = service.get_status()
        service_info = self._services[name]

        return {
            **status,
            "priority": service_info.priority.value,
            "dependencies": service_info.dependencies.copy(),
            "status": service_info.status.value,
            "error_message": service_info.error_message
        }

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有服务

        Returns:
            服务列表
        """
        result = {}
        for name, service_info in self._services.items():
            result[name] = {
                "priority": service_info.priority.value,
                "dependencies": service_info.dependencies.copy(),
                "status": service_info.status.value,
                "config": service_info.config.copy()
            }
        return result

    def _sort_services_by_dependencies(self) -> List[str]:
        """
        按依赖关系和优先级排序服务

        Returns:
            排序后的服务名称列表
        """
        # 简单的拓扑排序实现
        # 实际项目中可以使用更复杂的算法处理循环依赖

        sorted_services = []
        visited = set()
        visiting = set()

        def visit(service_name):
            if service_name in visiting:
                raise ValueError(f"检测到循环依赖: {service_name}")
            if service_name in visited:
                return

            visiting.add(service_name)

            # 访问依赖
            for dep in self._services[service_name].dependencies:
                if dep in self._services:
                    visit(dep)

            visiting.remove(service_name)
            visited.add(service_name)
            sorted_services.append(service_name)

        # 按优先级分组
        priority_groups = {}
        for name, info in self._services.items():
            priority = info.priority.value
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(name)

        # 按优先级从高到低处理
        for priority in sorted(priority_groups.keys(), reverse=True):
            for service_name in priority_groups[priority]:
                if service_name not in visited:
                    visit(service_name)

        return sorted_services


# 全局服务注册表实例
_service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """获取全局服务注册表"""
    return _service_registry


def register_service(name: str, service_class: Type[IService],
                     priority: ServicePriority = ServicePriority.NORMAL,
                     dependencies: List[str] = None,
                     config: Dict[str, Any] = None) -> bool:
    """
    注册服务到全局注册表

    Args:
        name: 服务名称
        service_class: 服务类
        priority: 优先级
        dependencies: 依赖服务列表
        config: 服务配置

    Returns:
        注册是否成功
    """
    return _service_registry.register_service(
        name=name,
        service_class=service_class,
        priority=priority,
        dependencies=dependencies,
        config=config
    )


def get_service(name: str) -> Optional[IService]:
    """
    从全局注册表获取服务

    Args:
        name: 服务名称

    Returns:
        服务实例
    """
    return _service_registry.get_service(name)
