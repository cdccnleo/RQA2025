#!/usr/bin/env python3
"""
统一服务容器接口

定义核心服务层服务容器的标准接口，确保所有服务容器实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from enum import Enum


class ServiceLifecycle(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"  # 单例模式
    TRANSIENT = "transient"  # 每次请求都创建新实例
    SCOPED = "scoped"       # 作用域内单例
    POOL = "pool"          # 对象池模式


class ServiceScope(Enum):
    """服务作用域"""
    GLOBAL = "global"
    REQUEST = "request"
    SESSION = "session"
    THREAD = "thread"


class ServiceStatus(Enum):
    """服务状态"""
    REGISTERED = "registered"   # 已注册
    INITIALIZING = "initializing"  # 初始化中
    RUNNING = "running"         # 运行中
    STOPPED = "stopped"         # 已停止
    ERROR = "error"            # 错误状态
    DESTROYED = "destroyed"     # 已销毁


class IServiceContainer(ABC):
    """
    服务容器统一接口

    所有服务容器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def register(self, service_type: Type, implementation: Optional[Type] = None,
                 factory: Optional[Callable] = None, lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
                 name: Optional[str] = None) -> bool:
        """
        注册服务

        Args:
            service_type: 服务接口类型
            implementation: 服务实现类
            factory: 服务工厂函数
            lifecycle: 服务生命周期
            name: 服务名称（可选，用于区分同一接口的多个实现）

        Returns:
            是否注册成功
        """

    @abstractmethod
    def register_instance(self, service_type: Type, instance: Any, name: Optional[str] = None) -> bool:
        """
        注册服务实例

        Args:
            service_type: 服务接口类型
            instance: 服务实例
            name: 服务名称

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister(self, service_type: Type, name: Optional[str] = None) -> bool:
        """
        注销服务

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            是否注销成功
        """

    @abstractmethod
    def resolve(self, service_type: Type, name: Optional[str] = None) -> Any:
        """
        解析服务实例

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            服务实例

        Raises:
            ServiceNotFoundError: 服务未找到
        """

    @abstractmethod
    def is_registered(self, service_type: Type, name: Optional[str] = None) -> bool:
        """
        检查服务是否已注册

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            是否已注册
        """

    @abstractmethod
    def get_service_info(self, service_type: Type, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取服务信息

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            服务信息字典
        """

    @abstractmethod
    def get_all_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有已注册的服务

        Returns:
            服务信息字典 {service_type: [service_infos]}
        """

    @abstractmethod
    def create_scope(self, scope_type: ServiceScope) -> 'IServiceScope':
        """
        创建服务作用域

        Args:
            scope_type: 作用域类型

        Returns:
            服务作用域对象
        """

    @abstractmethod
    def begin_scope(self, scope_type: ServiceScope) -> 'IServiceScope':
        """
        开始一个新的作用域

        Args:
            scope_type: 作用域类型

        Returns:
            服务作用域对象
        """

    @abstractmethod
    def get_service_status(self, service_type: Type, name: Optional[str] = None) -> ServiceStatus:
        """
        获取服务状态

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            服务状态
        """

    @abstractmethod
    def initialize_service(self, service_type: Type, name: Optional[str] = None) -> bool:
        """
        初始化服务

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            是否初始化成功
        """

    @abstractmethod
    def dispose_service(self, service_type: Type, name: Optional[str] = None) -> bool:
        """
        销毁服务

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            是否销毁成功
        """

    @abstractmethod
    def get_service_dependencies(self, service_type: Type, name: Optional[str] = None) -> List[Type]:
        """
        获取服务依赖

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            依赖的服务类型列表
        """

    @abstractmethod
    def validate_service(self, service_type: Type, name: Optional[str] = None) -> Dict[str, Any]:
        """
        验证服务配置

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            验证结果字典
        """

    @abstractmethod
    def get_service_metrics(self, service_type: Type = None, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取服务指标

        Args:
            service_type: 服务接口类型（可选）
            name: 服务名称（可选）

        Returns:
            服务指标字典
        """

    @abstractmethod
    def enable_monitoring(self, enabled: bool = True) -> None:
        """
        启用/禁用监控

        Args:
            enabled: 是否启用
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """清空服务实例缓存"""

    @abstractmethod
    def dispose(self) -> None:
        """销毁容器，清理所有资源"""


class IServiceScope(ABC):
    """
    服务作用域接口
    """

    @abstractmethod
    def resolve(self, service_type: Type, name: Optional[str] = None) -> Any:
        """
        在当前作用域内解析服务

        Args:
            service_type: 服务接口类型
            name: 服务名称

        Returns:
            服务实例
        """

    @abstractmethod
    def dispose(self) -> None:
        """销毁作用域，清理作用域内的服务实例"""

    @abstractmethod
    def get_scope_type(self) -> ServiceScope:
        """
        获取作用域类型

        Returns:
            作用域类型
        """

    @abstractmethod
    def get_services_in_scope(self) -> List[Dict[str, Any]]:
        """
        获取作用域内的服务

        Returns:
            服务信息列表
        """


class IServiceFactory(ABC):
    """
    服务工厂接口
    """

    @abstractmethod
    def create_service(self, service_type: Type, config: Dict[str, Any]) -> Any:
        """
        创建服务实例

        Args:
            service_type: 服务接口类型
            config: 服务配置

        Returns:
            服务实例
        """

    @abstractmethod
    def can_create(self, service_type: Type) -> bool:
        """
        检查是否可以创建指定类型的服务

        Args:
            service_type: 服务接口类型

        Returns:
            是否可以创建
        """

    @abstractmethod
    def get_supported_services(self) -> List[Type]:
        """
        获取支持的服务类型列表

        Returns:
            服务类型列表
        """


class IServiceRegistry(ABC):
    """
    服务注册表接口
    """

    @abstractmethod
    def register(self, service_type: Type, descriptor: Dict[str, Any]) -> bool:
        """
        注册服务描述符

        Args:
            service_type: 服务接口类型
            descriptor: 服务描述符

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister(self, service_type: Type) -> bool:
        """
        注销服务

        Args:
            service_type: 服务接口类型

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_descriptor(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """
        获取服务描述符

        Args:
            service_type: 服务接口类型

        Returns:
            服务描述符
        """

    @abstractmethod
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有注册的服务

        Returns:
            服务描述符字典 {service_type: descriptor}
        """


class IServiceHealthChecker(ABC):
    """
    服务健康检查器接口
    """

    @abstractmethod
    def check_health(self, service_instance: Any) -> Dict[str, Any]:
        """
        检查服务健康状态

        Args:
            service_instance: 服务实例

        Returns:
            健康检查结果字典
        """

    @abstractmethod
    def is_healthy(self, health_result: Dict[str, Any]) -> bool:
        """
        判断健康检查结果是否健康

        Args:
            health_result: 健康检查结果

        Returns:
            是否健康
        """

    @abstractmethod
    def get_health_score(self, health_result: Dict[str, Any]) -> float:
        """
        获取健康评分

        Args:
            health_result: 健康检查结果

        Returns:
            健康评分 (0-1)
        """
