"""
dependency_container 模块

提供 dependency_container 相关功能和接口。
"""

import logging

import inspect
import uuid
import threading

from .shared_interfaces import ILogger, StandardLogger
from contextlib import contextmanager
from typing import Any, Dict, Type, TypeVar, Optional, Callable, List
"""
依赖注入容器

Phase 2: 结构优化 - 依赖注入框架

实现依赖注入容器，管理组件间的依赖关系，提高代码的可测试性和可维护性。
"""

T = TypeVar('T')
FactoryFunc = Callable[[], Any]

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """依赖解析异常"""


class CircularDependencyError(DependencyResolutionError):
    """循环依赖异常"""


class ServiceNotFoundError(DependencyResolutionError):
    """服务未找到异常"""


class ServiceRegistrationError(Exception):
    """服务注册异常"""


class ServiceLifetime:
    """服务生命周期枚举"""
    SINGLETON = "singleton"  # 单例模式
    TRANSIENT = "transient"  # 每次都创建新实例
    SCOPED = "scoped"       # 作用域模式


class ServiceDescriptor:
    """服务描述符"""

    def __init__(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                 factory: Optional[FactoryFunc] = None, lifetime: str = ServiceLifetime.SINGLETON):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.lifetime = lifetime
        self.instance: Optional[T] = None
        self.instance_lock = threading.RLock()


class DependencyContainer:
    """依赖注入容器

    提供服务注册、解析和管理功能，支持多种生命周期模式。
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self._services: Dict[Type[Any], ServiceDescriptor] = {}
        self._scoped_instances: Dict[str, Dict[Type[Any], Any]] = {}
        self._scope_stack: List[str] = []
        self._lock = threading.RLock()
        self.logger = logger or StandardLogger(self.__class__.__name__)

    def register(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                 lifetime: str = ServiceLifetime.SINGLETON) -> 'DependencyContainer':
        """注册服务类型"""
        with self._lock:
            if service_type in self._services:
                raise ServiceRegistrationError(f"服务 {service_type.__name__} 已被注册")

            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                lifetime=lifetime
            )
            self._services[service_type] = descriptor
            self.logger.log_info(
                f"已注册服务: {service_type.__name__} -> {implementation_type.__name__ if implementation_type else 'self'}")

        return self

    def register_factory(self, service_type: Type[T], factory: FactoryFunc,
                         lifetime: str = ServiceLifetime.SINGLETON) -> 'DependencyContainer':
        """注册服务工厂"""
        with self._lock:
            if service_type in self._services:
                raise ServiceRegistrationError(f"服务 {service_type.__name__} 已被注册")

            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime
            )
            self._services[service_type] = descriptor
            self.logger.log_info(f"已注册服务工厂: {service_type.__name__}")

        return self

    def register_instance(self, service_type: Type[T], instance: T) -> 'DependencyContainer':
        """注册服务实例"""
        with self._lock:
            if service_type in self._services:
                raise ServiceRegistrationError(f"服务 {service_type.__name__} 已被注册")

            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=type(instance),
                lifetime=ServiceLifetime.SINGLETON
            )
            descriptor.instance = instance
            self._services[service_type] = descriptor
            self.logger.log_info(f"已注册服务实例: {service_type.__name__}")

        return self

    def unregister(self, service_type: Type[T]) -> bool:
        """注销服务"""
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
                self.logger.log_info(f"已注销服务: {service_type.__name__}")
                return True
        return False

    def resolve(self, service_type: Type[T]) -> T:
        """解析服务实例"""
        return self._resolve(service_type, set())

    def _resolve(self, service_type: Type[T], resolving: set) -> T:
        """内部解析方法，支持循环依赖检测"""
        with self._lock:
            # 检查循环依赖
            if service_type in resolving:
                raise CircularDependencyError(f"检测到循环依赖: {service_type.__name__}")

            # 检查服务是否已注册
            if service_type not in self._services:
                raise ServiceNotFoundError(f"服务 {service_type.__name__} 未注册")

            descriptor = self._services[service_type]
            resolving.add(service_type)

            try:
                # 根据生命周期创建或返回实例
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    return self._resolve_singleton(descriptor, resolving)
                elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                    return self._resolve_transient(descriptor, resolving)
                elif descriptor.lifetime == ServiceLifetime.SCOPED:
                    return self._resolve_scoped(descriptor, resolving)
                else:
                    raise DependencyResolutionError(f"未知的服务生命周期: {descriptor.lifetime}")

            finally:
                resolving.remove(service_type)

    def _resolve_singleton(self, descriptor: ServiceDescriptor, resolving: set) -> Any:
        """解析单例服务"""
        if descriptor.instance is None:
            with descriptor.instance_lock:
                if descriptor.instance is None:  # 双重检查锁定
                    descriptor.instance = self._create_instance(descriptor, resolving)

        return descriptor.instance

    def _resolve_transient(self, descriptor: ServiceDescriptor, resolving: set) -> Any:
        """解析临时服务"""
        return self._create_instance(descriptor, resolving)

    def _resolve_scoped(self, descriptor: ServiceDescriptor, resolving: set) -> Any:
        """解析作用域服务"""
        if not self._scope_stack:
            raise DependencyResolutionError("当前没有活跃的作用域")

        scope_id = self._scope_stack[-1]
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}

        scoped_instances = self._scoped_instances[scope_id]

        if descriptor.service_type not in scoped_instances:
            scoped_instances[descriptor.service_type] = self._create_instance(descriptor, resolving)

        return scoped_instances[descriptor.service_type]

    def _create_instance(self, descriptor: ServiceDescriptor, resolving: set) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                # 使用工厂方法创建
                instance = descriptor.factory()
            else:
                # 使用构造函数创建
                instance = self._create_instance_with_dependencies(
                    descriptor.implementation_type, resolving)

            self.logger.log_info(f"已创建服务实例: {descriptor.service_type.__name__}")
            return instance

        except Exception as e:
            self.logger.log_error(f"创建服务实例失败: {descriptor.service_type.__name__}", error=e)
            raise DependencyResolutionError(f"创建服务实例失败: {descriptor.service_type.__name__}") from e

    def _create_instance_with_dependencies(self, implementation_type: Type[T], resolving: set) -> T:
        """通过构造函数注入依赖创建实例"""
        try:
            # 获取构造函数参数
            signature = inspect.signature(implementation_type.__init__)

            # 过滤self参数
            parameters = {name: param for name, param in signature.parameters.items()
                          if name != 'self'}

            # 解析依赖
            kwargs = {}
            for param_name, param in parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    # 有类型注解，尝试解析依赖
                    try:
                        kwargs[param_name] = self._resolve(param.annotation, resolving)
                    except DependencyResolutionError:
                        # 如果无法解析依赖，跳过（假设有默认值）
                        pass
                else:
                    # 没有类型注解，跳过
                    pass

            # 创建实例
            return implementation_type(**kwargs)

        except Exception as e:
            raise DependencyResolutionError(f"依赖注入失败: {implementation_type.__name__}") from e

    @contextmanager
    def begin_scope(self, scope_id: Optional[str] = None):
        """开始作用域"""
        if scope_id is None:
            scope_id = str(uuid.uuid4())

        self._scope_stack.append(scope_id)
        self.logger.log_info(f"开始作用域: {scope_id}")

        try:
            yield scope_id
        finally:
            self._scope_stack.pop()
            if scope_id in self._scoped_instances:
                del self._scoped_instances[scope_id]
            self.logger.log_info(f"结束作用域: {scope_id}")

    def is_registered(self, service_type: Type[T]) -> bool:
        """检查服务是否已注册"""
        with self._lock:
            return service_type in self._services

    def get_registered_services(self) -> List[Type[Any]]:
        """获取已注册的服务类型"""
        with self._lock:
            return list(self._services.keys())

    def get_service_info(self, service_type: Type[T]) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        with self._lock:
            if service_type not in self._services:
                return None

            descriptor = self._services[service_type]
            return {
                "service_type": service_type.__name__,
                "implementation_type": descriptor.implementation_type.__name__,
                "lifetime": descriptor.lifetime,
                "has_instance": descriptor.instance is not None,
                "has_factory": descriptor.factory is not None
            }

    def clear(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._scoped_instances.clear()
            self._scope_stack.clear()
            self.logger.log_info("依赖注入容器已清空")

    def __contains__(self, service_type: Type[T]) -> bool:
        """检查容器是否包含服务"""
        return self.is_registered(service_type)

    def __getitem__(self, service_type: Type[T]) -> T:
        """通过索引获取服务实例"""
        return self.resolve(service_type)

    def __len__(self) -> int:
        """获取已注册的服务数量"""
        with self._lock:
            return len(self._services)

# =============================================================================
# 便捷函数和装饰器
# =============================================================================


def injectable(lifetime: str = ServiceLifetime.SINGLETON):
    """服务注入装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._service_lifetime = lifetime
        return cls
    return decorator


def singleton(cls: Type[T]) -> Type[T]:
    """单例服务装饰器"""
    return injectable(ServiceLifetime.SINGLETON)(cls)


def transient(cls: Type[T]) -> Type[T]:
    """临时服务装饰器"""
    return injectable(ServiceLifetime.TRANSIENT)(cls)


def scoped(cls: Type[T]) -> Type[T]:
    """作用域服务装饰器"""
    return injectable(ServiceLifetime.SCOPED)(cls)

# =============================================================================
# 服务集合
# =============================================================================


class ServiceCollection:
    """服务集合

    提供批量服务注册的便捷方式。
    """

    def __init__(self, container: DependencyContainer):
        self.container = container

    def add_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None):
        """添加单例服务"""
        return self.container.register(service_type, implementation_type, ServiceLifetime.SINGLETON)

    def add_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None):
        """添加临时服务"""
        return self.container.register(service_type, implementation_type, ServiceLifetime.TRANSIENT)

    def add_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None):
        """添加作用域服务"""
        return self.container.register(service_type, implementation_type, ServiceLifetime.SCOPED)

    def add_singleton_factory(self, service_type: Type[T], factory: FactoryFunc):
        """添加单例服务工厂"""
        return self.container.register_factory(service_type, factory, ServiceLifetime.SINGLETON)

    def add_transient_factory(self, service_type: Type[T], factory: FactoryFunc):
        """添加临时服务工厂"""
        return self.container.register_factory(service_type, factory, ServiceLifetime.TRANSIENT)

    def add_scoped_factory(self, service_type: Type[T], factory: FactoryFunc):
        """添加作用域服务工厂"""
        return self.container.register_factory(service_type, factory, ServiceLifetime.SCOPED)

    def add_instance(self, service_type: Type[T], instance: T):
        """添加服务实例"""
        return self.container.register_instance(service_type, instance)
