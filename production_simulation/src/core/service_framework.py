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

from src.core.constants import MAX_RECORDS, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# 参数封装数据类 - 解决长参数列表问题
# 注意：使用字符串类型注解以支持前向引用
@dataclass
class ServiceRegistrationConfig:
    """服务注册配置参数"""
    name: str
    service_class: Type['IService']  # 使用字符串以支持前向引用
    priority: 'ServicePriority' = None  # 使用字符串以支持前向引用
    dependencies: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # ServicePriority在文件后面定义，延迟赋值
        if self.priority is None:
            # 在文件末尾，ServicePriority已定义，直接引用
            import sys
            current_module = sys.modules[__name__]
            ServicePriority = getattr(current_module, 'ServicePriority')
            self.priority = ServicePriority.NORMAL
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


@dataclass
class ServiceStatusQuery:
    """服务状态查询参数"""
    include_config: bool = True
    include_health: bool = True
    include_metrics: bool = False
    timeout: float = 5.0


@dataclass
class ServiceListQuery:
    """服务列表查询参数"""
    status_filter: Optional['ServiceStatus'] = None  # 前向引用
    priority_filter: Optional['ServicePriority'] = None  # 前向引用
    include_details: bool = False
    max_results: int = MAX_RECORDS


@dataclass
class HealthCheckConfig:
    """健康检查配置参数"""
    timeout: float = DEFAULT_TIMEOUT
    retry_count: int = 3
    retry_delay: float = 1.0
    include_detailed_report: bool = False


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

    def get_status(self, query: Optional[ServiceStatusQuery] = None) -> Dict[str, Any]:
        """获取服务状态 - 重构版：参数封装

        Args:
            query: 状态查询配置对象
        """
        # 使用参数对象或默认配置
        status_query = query or ServiceStatusQuery()

        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0

            status = {
                "name": self.name,
                "running": self._running,
                "uptime": uptime,
            }

            # 根据查询配置添加可选字段
            if status_query.include_config:
                status["config"] = self.config

            if status_query.include_health:
                status["health"] = self.health_check()

            if status_query.include_metrics:
                status["metrics"] = self._get_metrics()

            return status

    def _get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        # 这里可以实现具体的指标收集逻辑
        return {
            "requests_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0
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

    def list_services(self, query: Optional[ServiceListQuery] = None) -> Dict[str, Dict[str, Any]]:
        """
        列出所有服务 - 重构版：参数封装

        Args:
            query: 服务列表查询配置对象

        Returns:
            服务列表
        """
        # 使用参数对象或默认配置
        list_query = query or ServiceListQuery()

        result = {}
        count = 0

        for name, service_info in self._services.items():
            # 应用过滤器
            if list_query.status_filter and service_info.status != list_query.status_filter:
                continue
            if list_query.priority_filter and service_info.priority != list_query.priority_filter:
                continue

            # 达到最大结果数限制
            if count >= list_query.max_results:
                break

            service_data = {
                "priority": service_info.priority.value,
                "dependencies": service_info.dependencies.copy(),
                "status": service_info.status.value,
            }

            # 根据查询配置添加详细信息
            if list_query.include_details:
                service_data["config"] = service_info.config.copy()
                service_data["error_message"] = service_info.error_message

            result[name] = service_data
            count += 1

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
    注册服务到全局注册表 - 重构版：参数封装兼容

    Args:
        name: 服务名称
        service_class: 服务类
        priority: 优先级
        dependencies: 依赖服务列表
        config: 服务配置

    Returns:
        注册是否成功
    """
    # 构造参数对象
    registration_config = ServiceRegistrationConfig(
        name=name,
        service_class=service_class,
        priority=priority,
        dependencies=dependencies,
        config=config
    )

    return _register_service_with_config(registration_config)


def _register_service_with_config(config: ServiceRegistrationConfig) -> bool:
    """使用配置对象注册服务"""
    global _service_registry

    if not _service_registry:
        _service_registry = ServiceRegistry()

    try:
        service = config.service_class(config.name, config.config or {})
        service.priority = config.priority
        service.dependencies = config.dependencies or []

        return _service_registry.register_service(service)
    except Exception as e:
        logger.error(f"注册服务失败 {config.name}: {e}")
        return False


def register_service_with_config(config: ServiceRegistrationConfig) -> bool:
    """
    使用配置对象注册服务 - 新增接口

    Args:
        config: 服务注册配置对象

    Returns:
        注册是否成功
    """
    return _register_service_with_config(config)


def get_service(name: str) -> Optional[IService]:
    """
    从全局注册表获取服务

    Args:
        name: 服务名称

    Returns:
        服务实例
    """
    return _service_registry.get_service(name)
