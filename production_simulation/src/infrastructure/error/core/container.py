
import inspect
import threading

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type
"""
依赖注入容器

提供统一的依赖注入容器，支持服务注册、解析和生命周期管理。
"""


class Lifecycle(Enum):

    """服务生命周期枚举"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬时
    SCOPED = "scoped"  # 作用域


@dataclass
class ServiceRegistrationConfig:
    """服务注册配置参数对象"""
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    instance: Optional[Any] = None


class ServiceDescriptor:
    """服务描述符 - 重构后使用参数对象"""

    def __init__(self, config: ServiceRegistrationConfig):
        self.service_type = config.service_type
        self.implementation = config.implementation or config.service_type
        self.factory = config.factory
        self.lifecycle = config.lifecycle
        self.instance = config.instance
        self._lock = threading.RLock()


class DependencyContainer:

    """依赖注入容器"""

    def __init__(self):

        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._scope_stack = []

    def register(self, service_type: Type, implementation: Optional[Type] = None, 
                 factory: Optional[Callable] = None, lifecycle: Lifecycle = Lifecycle.SINGLETON):
        """
        注册服务 - 重构后保持向后兼容性

        Args:
            service_type: 服务类型
            implementation: 实现类型
            factory: 工厂函数
            lifecycle: 生命周期

        Returns:
            容器实例，支持链式调用
        """
        config = ServiceRegistrationConfig(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifecycle=lifecycle
        )
        return self._register_with_config(config)

    def _register_with_config(self, config: ServiceRegistrationConfig):
        """使用配置对象注册服务"""
        with self._lock:
            descriptor = ServiceDescriptor(config)
            self._services[config.service_type] = descriptor
        return self

    def register_singleton(self, service_type: Type, implementation: Optional[Type] = None, 
                          factory: Optional[Callable] = None):
        """注册单例服务"""
        return self.register(service_type, implementation, factory, Lifecycle.SINGLETON)

    def register_transient(self, service_type: Type, implementation: Optional[Type] = None, 
                          factory: Optional[Callable] = None):
        """注册瞬时服务"""
        return self.register(service_type, implementation, factory, Lifecycle.TRANSIENT)

    def register_scoped(self, service_type: Type, implementation: Optional[Type] = None, 
                       factory: Optional[Callable] = None):
        """注册作用域服务"""
        return self.register(service_type, implementation, factory, Lifecycle.SCOPED)

    def resolve(self, service_type: Type) -> Any:
        """
        解析服务

        Args:
            service_type: 服务类型

        Returns:
            服务实例

        Raises:
            KeyError: 服务未注册
            Exception: 服务创建失败
        """
        with self._lock:
            if service_type not in self._services:
                raise KeyError(f"Service {service_type.__name__} not registered")

            descriptor = self._services[service_type]

            # 根据生命周期返回实例
            if descriptor.lifecycle == Lifecycle.SINGLETON:
                return self._resolve_singleton(descriptor)
            elif descriptor.lifecycle == Lifecycle.TRANSIENT:
                return self._resolve_transient(descriptor)
            elif descriptor.lifecycle == Lifecycle.SCOPED:
                return self._resolve_scoped(descriptor)
            else:
                raise ValueError(f"Unknown lifecycle: {descriptor.lifecycle}")

    def _resolve_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """解析单例服务"""
        with descriptor._lock:
            if descriptor.instance is None:
                descriptor.instance = self._create_instance(descriptor)
            return descriptor.instance

    def _resolve_transient(self, descriptor: ServiceDescriptor) -> Any:
        """解析瞬时服务"""
        return self._create_instance(descriptor)

    def _resolve_scoped(self, descriptor: ServiceDescriptor) -> Any:
        """解析作用域服务"""
        if not self._scope_stack:
            raise RuntimeError("Cannot resolve scoped service outside of scope")

        scope_id = id(self._scope_stack[-1])
        key = (descriptor.service_type, scope_id)

        if key not in self._scoped_instances:
            self._scoped_instances[key] = self._create_instance(descriptor)

        return self._scoped_instances[key]

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                return descriptor.factory(self)
            else:
                params = self._resolve_constructor_parameters(descriptor)
                return descriptor.implementation(**params)
        except Exception as e:
            raise Exception(
                f"Failed to create instance of {descriptor.service_type.__name__}: {str(e)}") from e

    def _resolve_constructor_parameters(self, descriptor: ServiceDescriptor) -> dict:
        """解析构造函数参数 - 减少深层嵌套"""
        sig = inspect.signature(descriptor.implementation.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            param_value = self._resolve_parameter_value(descriptor, name, param)
            params[name] = param_value

        return params

    def _resolve_parameter_value(self, descriptor: ServiceDescriptor, name: str, param) -> Any:
        """解析单个参数值 - 进一步减少嵌套"""
        if param.annotation == inspect.Parameter.empty:
            return self._handle_unannotated_parameter(descriptor, name, param)
        
        return self._handle_annotated_parameter(descriptor, name, param)

    def _handle_annotated_parameter(self, descriptor: ServiceDescriptor, name: str, param) -> Any:
        """处理有类型注解的参数"""
        try:
            return self.resolve(param.annotation)
        except KeyError:
            if param.default == inspect.Parameter.empty:
                raise KeyError(
                    f"Cannot resolve dependency {param.annotation.__name__} for {descriptor.service_type.__name__}")
            return param.default

    def _handle_unannotated_parameter(self, descriptor: ServiceDescriptor, name: str, param) -> Any:
        """处理无类型注解的参数"""
        if param.default == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter {name} in {descriptor.service_type.__name__} has no type annotation")
        return param.default

    @contextmanager
    def scope(self):
        """创建作用域上下文"""
        scope_id = object()
        self._scope_stack.append(scope_id)
        try:
            yield self
        finally:
            self._scope_stack.pop()
            # 清理作用域实例
            self._cleanup_scope_instances(id(scope_id))

    def _cleanup_scope_instances(self, scope_id: int) -> None:
        """清理作用域实例 - 减少嵌套"""
        keys_to_remove = [
            key for key in self._scoped_instances.keys() 
            if key[1] == scope_id
        ]
        for key in keys_to_remove:
            del self._scoped_instances[key]

    def has_service(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services

    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """获取所有已注册的服务"""
        return self._services.copy()

    def clear(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._scoped_instances.clear()
            self._scope_stack.clear()


# 全局容器实例
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """获取全局容器实例"""
    return _container


def register(service_type: Type, implementation: Optional[Type] = None, 
             factory: Optional[Callable] = None, lifecycle: Lifecycle = Lifecycle.SINGLETON):
    """注册服务到全局容器"""
    return _container.register(service_type, implementation, factory, lifecycle)


def resolve(service_type: Type) -> Any:
    """从全局容器解析服务"""
    return _container.resolve(service_type)


def has_service(service_type: Type) -> bool:
    """检查全局容器中是否有服务"""
    return _container.has_service(service_type)


@contextmanager
def scope():
    """创建全局容器作用域"""
    with _container.scope() as container:
        yield container
