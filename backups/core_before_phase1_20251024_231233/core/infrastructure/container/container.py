"""
服务容器实现
提供依赖注入功能 - 优化版
"""

from typing import Dict, Any, Callable, Optional, Type, List, Set
import logging
import threading
import time
import inspect
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime
import importlib

from .base import ComponentStatus, ComponentHealth
from .patterns.standard_interface_template import StandardComponent
from .exceptions.core_exceptions import ContainerException
from ...foundation.base import generate_id

logger = logging.getLogger(__name__)


class Lifecycle(Enum):

    """服务生命周期枚举"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬时
    SCOPED = "scoped"  # 作用域


class ServiceHealth(Enum):

    """服务健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceStatus(Enum):

    """服务状态枚举"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    REGISTERED = "registered"
    RESOLVED = "resolved"
    DISPOSED = "disposed"


@dataclass
class ServiceMetrics:

    """服务性能指标"""
    resolve_count: int = 0
    total_resolve_time: float = 0.0
    last_resolve_time: Optional[datetime] = None
    error_count: int = 0
    last_error_time: Optional[datetime] = None
    memory_usage: Optional[int] = None

    @property
    def avg_resolve_time(self) -> float:
        """平均解析时间"""
        return self.total_resolve_time / self.resolve_count if self.resolve_count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.resolve_count + self.error_count
        return (self.resolve_count / total * 100) if total > 0 else 0.0


@dataclass
class ServiceDescriptor:

    """服务描述符 - 增强版"""
    name: str
    service_type: Optional[Type] = None
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    instance: Optional[Any] = None
    version: str = "1.0.0"
    dependencies: List[str] = None
    health_check: Optional[Callable] = None
    created_time: Optional[float] = None
    last_health_check: Optional[float] = None
    health_status: ServiceHealth = ServiceHealth.UNKNOWN
    status: ServiceStatus = ServiceStatus.UNKNOWN
    error_count: int = 0
    max_errors: int = 3
    description: str = ""
    tags: List[str] = None
    config: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    def __post_init__(self):

        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}
        if self.metadata is None:
            self.metadata = {}
        if self.created_time is None:
            self.created_time = time.time()


@dataclass
class HealthCheckResult:

    """健康检查结果"""
    service_name: str
    is_healthy: bool
    details: Dict[str, Any] = None
    timestamp: float = None
    response_time: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:

        return {
            'service_name': self.service_name,
            'is_healthy': self.is_healthy,
            'details': self.details,
            'timestamp': self.timestamp,
            'response_time': self.response_time,
            'error_message': self.error_message
        }


class ServiceHealthMonitor:

    """服务健康监控器"""

    def __init__(self, check_interval: float = 30.0):

        self.check_interval = check_interval
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable] = []
        self._lock = threading.RLock()

    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            self.monitor_thread.start()
            logger.info("服务健康监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("服务健康监控已停止")

    def _monitor_worker(self):
        """监控工作线程"""
        while self.monitoring:
            time.sleep(self.check_interval)
            # 这里可以添加定期健康检查逻辑

    def register_health_check_callback(self, callback: Callable):
        """注册健康检查回调"""
        self.callbacks.append(callback)

    def update_health_result(self, service_name: str, result: HealthCheckResult):
        """更新健康检查结果"""
        with self._lock:
            self.health_results[service_name] = result

    def get_health_results(self) -> Dict[str, HealthCheckResult]:
        """获取健康检查结果"""
        with self._lock:
            return self.health_results.copy()

    def get_unhealthy_services(self) -> List[str]:
        """获取不健康的服务"""
        with self._lock:
            return [name for name, result in self.health_results.items()
                    if not result.is_healthy]

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康检查摘要"""
        with self._lock:
            total = len(self.health_results)
            healthy = sum(1 for result in self.health_results.values() if result.is_healthy)
            return {
                'total_services': total,
                'healthy_services': healthy,
                'unhealthy_services': total - healthy,
                'health_rate': (healthy / total * 100) if total > 0 else 0.0
            }


class ServiceScope:

    """服务作用域"""

    def __init__(self, scope_id: str):

        self.scope_id = scope_id
        self.services: Dict[str, Any] = {}

    def add_service(self, name: str, instance: Any):
        """添加服务到作用域"""
        self.services[name] = instance

    def get_service(self, name: str) -> Any:
        """从作用域获取服务"""
        return self.services.get(name)

    def has_service(self, name: str) -> bool:
        """检查作用域是否包含服务"""
        return name in self.services

    def clear(self):
        """清空作用域"""
        self.services.clear()


class ServiceAutoDiscovery:

    """服务自动发现"""

    def __init__(self, container: 'DependencyContainer'):

        self.container = container

    def discover_services(self, module_path: str) -> List[Type]:
        """发现服务"""
        discovered_services = []
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                obj = getattr(module, name)
                if self._is_injectable_service(obj):
                    discovered_services.append(obj)
        except ImportError as e:
            logger.warning(f"无法导入模块 {module_path}: {e}")
        return discovered_services

    def _is_injectable_service(self, obj: Any) -> bool:
        """检查是否为可注入服务"""
        return (inspect.isclass(obj)
                and hasattr(obj, '__init__')
                and not obj.__name__.startswith('_'))

    def auto_register_discovered_services(self, module_path: str) -> None:
        """自动注册发现的服务"""
        services = self.discover_services(module_path)
        for service in services:
            try:
                lifecycle = self._determine_lifecycle(service)
                self.container.register(
                    name=service.__name__,
                    service_type=service,
                    lifecycle=lifecycle
                )
                logger.info(f"自动注册服务: {service.__name__}")
            except Exception as e:
                logger.warning(f"自动注册服务失败 {service.__name__}: {e}")

    def _determine_lifecycle(self, service: Type) -> Lifecycle:
        """确定服务生命周期"""
        if hasattr(service, '_is_singleton') and service._is_singleton:
            return Lifecycle.SINGLETON
        elif hasattr(service, '_is_transient') and service._is_transient:
            return Lifecycle.TRANSIENT
        elif hasattr(service, '_is_scoped') and service._is_scoped:
            return Lifecycle.SCOPED
        else:
            return Lifecycle.SINGLETON


class DependencyContainer(StandardComponent):

    """依赖注入容器 - 优化版"""

    def __init__(self, enable_health_monitoring: bool = True, enable_service_discovery: bool = True):

        super().__init__("DependencyContainer", "2.0.0", "依赖注入容器核心组件")

        self.enable_health_monitoring = enable_health_monitoring
        self.enable_service_discovery = enable_service_discovery

        # 初始化各个组件
        self._initialize_storage()
        self._initialize_dependencies()
        self._initialize_monitoring()
        self._initialize_threading()
        self._initialize_statistics()

    def _initialize_storage(self):
        """初始化服务存储"""
        self._services = {}
        self._service_descriptors = {}
        self._singleton_instances = {}
        self._scoped_instances = {}
        self._current_scope = None

    def _initialize_dependencies(self):
        """初始化依赖关系"""
        self._dependencies = defaultdict(list)
        self._reverse_dependencies = defaultdict(list)

    def _initialize_monitoring(self):
        """初始化监控和发现"""
        self._health_monitor = None
        self._service_discovery = {}
        self._auto_discovery = None
        # 初始化新的组件
        self._instance_creator = InstanceCreator(self)

    def _initialize_threading(self):
        """初始化线程安全"""
        self._lock = threading.RLock()

    def _initialize_statistics(self):
        """初始化统计信息"""
        self._stats = {
            'total_services': 0,
            'singleton_services': 0,
            'transient_services': 0,
            'scoped_services': 0,
            'health_checks': 0,
            'failed_health_checks': 0
        }

    def initialize(self) -> bool:
        """初始化容器"""
        try:
            self.set_status(ComponentStatus.INITIALIZING)

            # 初始化健康监控
            if self.enable_health_monitoring:
                self._health_monitor = ServiceHealthMonitor()
                self._health_monitor.start_monitoring()

            # 初始化自动发现
            if self.enable_service_discovery:
                self._auto_discovery = ServiceAutoDiscovery(self)

            self._initialized = True
            self.set_status(ComponentStatus.INITIALIZED)
            self.set_health(ComponentHealth.HEALTHY)

            logger.info("依赖注入容器初始化完成")
            return True

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            logger.error(f"依赖注入容器初始化失败: {e}")
            raise ContainerException(f"依赖注入容器初始化失败: {e}")

    def shutdown(self) -> bool:
        """关闭容器"""
        try:
            self.set_status(ComponentStatus.STOPPING)

            # 停止健康监控
            if self._health_monitor:
                self._health_monitor.stop_monitoring()

            # 清理所有服务
            self.clear()

            self.set_status(ComponentStatus.STOPPED)
            logger.info("依赖注入容器已关闭")
            return True

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            logger.error(f"依赖注入容器关闭失败: {e}")
            return False

    def register(self, name: str, service: Any = None, service_type: Optional[Type] = None,
                 factory: Optional[Callable] = None, lifecycle: Lifecycle = Lifecycle.SINGLETON,
                 version: str = "1.0.0", dependencies: Optional[List[str]] = None,
                 health_check: Optional[Callable] = None, description: str = "",
                 tags: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> 'DependencyContainer':
        """注册服务"""
        try:
            with self._lock:
                self._validate_service_registration(name)
                descriptor = self._create_service_descriptor(
                    name, service, service_type, factory, lifecycle, version,
                    dependencies, health_check, description, tags, config
                )
                self._store_service_descriptor(name, descriptor, service)
                self._update_service_statistics(lifecycle)
                self._log_service_registration(name, lifecycle)
                return self

        except Exception as e:
            logger.error(f"注册服务失败: {name}, 错误: {e}")
            raise ContainerException(f"注册服务失败: {name}, 错误: {e}")

    def _validate_service_registration(self, name: str):
        """验证服务注册"""
        if name in self._service_descriptors:
            raise ContainerException(f"服务已存在: {name}")

    def _create_service_descriptor(self, name: str, service: Any, service_type: Optional[Type],
                                   factory: Optional[Callable], lifecycle: Lifecycle, version: str,
                                   dependencies: Optional[List[str]], health_check: Optional[Callable],
                                   description: str, tags: Optional[List[str]],
                                   config: Optional[Dict[str, Any]]) -> ServiceDescriptor:
        """创建服务描述符"""
        return ServiceDescriptor(
            name=name,
            service_type=service_type,
            implementation=service,
            factory=factory,
            lifecycle=lifecycle,
            version=version,
            dependencies=dependencies or [],
            health_check=health_check,
            description=description,
            tags=tags or [],
            config=config or {}
        )

    def _store_service_descriptor(self, name: str, descriptor: ServiceDescriptor, service: Any):
        """存储服务描述符"""
        self._service_descriptors[name] = descriptor
        self._services[name] = service

    def _update_service_statistics(self, lifecycle: Lifecycle):
        """更新服务统计信息"""
        if lifecycle == Lifecycle.SINGLETON:
            self._stats['singleton_services'] += 1
        elif lifecycle == Lifecycle.TRANSIENT:
            self._stats['transient_services'] += 1
        elif lifecycle == Lifecycle.SCOPED:
            self._stats['scoped_services'] += 1
        self._stats['total_services'] += 1

    def _log_service_registration(self, name: str, lifecycle: Lifecycle):
        """记录服务注册日志"""
        logger.info(f"注册服务: {name} (生命周期: {lifecycle.value})")

    def register_singleton(self, name: str, service: Any = None, service_type: Optional[Type] = None,


                           factory: Optional[Callable] = None, version: str = "1.0.0",
                           dependencies: Optional[List[str]] = None,
                           health_check: Optional[Callable] = None, **kwargs) -> 'DependencyContainer':
        """注册单例服务"""
        return self.register(name, service, service_type, factory, Lifecycle.SINGLETON,
                             version, dependencies, health_check, **kwargs)

    def register_transient(self, name: str, service_type: Optional[Type] = None,


                           factory: Optional[Callable] = None, version: str = "1.0.0",
                           dependencies: Optional[List[str]] = None,
                           health_check: Optional[Callable] = None, **kwargs) -> 'DependencyContainer':
        """注册瞬时服务"""
        return self.register(name, service_type, service_type, factory, Lifecycle.TRANSIENT,
                             version, dependencies, health_check, **kwargs)

    def register_scoped(self, name: str, service_type: Optional[Type] = None,


                        factory: Optional[Callable] = None, version: str = "1.0.0",
                        dependencies: Optional[List[str]] = None,
                        health_check: Optional[Callable] = None, **kwargs) -> 'DependencyContainer':
        """注册作用域服务"""
        return self.register(name, service_type, service_type, factory, Lifecycle.SCOPED,
                             version, dependencies, health_check, **kwargs)

    def register_factory(self, name: str, factory: Callable, lifecycle: Lifecycle = Lifecycle.SINGLETON,


                         version: str = "1.0.0", dependencies: Optional[List[str]] = None,
                         health_check: Optional[Callable] = None, **kwargs) -> 'DependencyContainer':
        """注册工厂服务"""
        return self.register(name, None, None, factory, lifecycle, version, dependencies, health_check, **kwargs)

    def get(self, name: str) -> Any:
        """获取服务（兼容性方法）"""
        return self.resolve(name)

    def resolve(self, name: str) -> Any:
        """解析服务（增强版）"""
        start_time = time.time()

        try:
            with self._lock:
                if name not in self._service_descriptors:
                    raise ContainerException(f"服务未找到: {name}")

                descriptor = self._service_descriptors[name]

                # 检查循环依赖
                if self._has_circular_dependency(name):
                    raise ContainerException(f"检测到循环依赖: {name}")

                # 根据生命周期返回实例
                if descriptor.lifecycle == Lifecycle.SINGLETON:
                    instance = self._get_singleton(descriptor)
                elif descriptor.lifecycle == Lifecycle.TRANSIENT:
                    instance = self._get_transient(descriptor)
                elif descriptor.lifecycle == Lifecycle.SCOPED:
                    instance = self._get_scoped(descriptor)
                else:
                    raise ContainerException(f"未知的生命周期: {descriptor.lifecycle}")

                # 更新服务信息
                descriptor.status = ServiceStatus.RESOLVED
                descriptor.metrics.resolve_count += 1
                descriptor.metrics.last_resolve_time = datetime.now()

                return instance

        except Exception as e:
            if name in self._service_descriptors:
                descriptor = self._service_descriptors[name]
                descriptor.metrics.error_count += 1
                descriptor.metrics.last_error_time = datetime.now()
                descriptor.status = ServiceStatus.ERROR
            logger.error(f"解析服务失败: {name}, 错误: {e}")
            raise
        finally:
            resolve_time = time.time() - start_time
            if name in self._service_descriptors:
                descriptor = self._service_descriptors[name]
                descriptor.metrics.total_resolve_time += resolve_time

    def _resolve_without_lock(self, name: str) -> Any:
        """无锁解析服务（内部使用）"""
        if name not in self._service_descriptors:
            raise ContainerException(f"服务未找到: {name}")

        descriptor = self._service_descriptors[name]

        if descriptor.lifecycle == Lifecycle.SINGLETON:
            return self._get_singleton(descriptor)
        elif descriptor.lifecycle == Lifecycle.TRANSIENT:
            return self._get_transient(descriptor)
        elif descriptor.lifecycle == Lifecycle.SCOPED:
            return self._get_scoped(descriptor)
        else:
            raise ContainerException(f"未知的生命周期: {descriptor.lifecycle}")

    def _get_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """获取单例服务"""
        if descriptor.instance is None:
            descriptor.instance = self._create_instance(descriptor)
        return descriptor.instance

    def _get_transient(self, descriptor: ServiceDescriptor) -> Any:
        """获取瞬时服务"""
        return self._create_instance(descriptor)

    def _get_scoped(self, descriptor: ServiceDescriptor) -> Any:
        """获取作用域服务"""
        if not self._current_scope:
            raise ContainerException("无法在作用域外解析作用域服务")

        scope_id = self._current_scope.scope_id
        key = (descriptor.name, scope_id)

        if key not in self._scoped_instances:
            self._scoped_instances[key] = self._create_instance(descriptor)
            self._current_scope.add_service(descriptor.name, self._scoped_instances[key])

        return self._scoped_instances[key]

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例 - 使用新的InstanceCreator"""
        return self._instance_creator.create_instance(descriptor)

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查的具体逻辑"""
        try:
            # 检查容器基本状态
            services_count = len(self._services)
            healthy_services = sum(1 for desc in self._service_descriptors.values()
                                   if desc.status == ServiceStatus.RESOLVED)

            # 检查是否有循环依赖
            circular_deps = []
            for name in self._service_descriptors:
                if self._has_circular_dependency(name):
                    circular_deps.append(name)

            issues = []
            if circular_deps:
                issues.append(f"发现循环依赖: {circular_deps}")
            if services_count == 0:
                issues.append("容器中没有注册任何服务")

            return {
                'healthy': len(circular_deps) == 0 and services_count > 0,
                'total_services': services_count,
                'healthy_services': healthy_services,
                'issues': issues,
                'recommendations': [
                    '检查服务依赖关系' if circular_deps else '容器运行正常',
                    '考虑注册更多服务' if services_count == 0 else '服务注册正常'
                ]
            }

        except Exception as e:
            return {
                'healthy': False,
                'issues': [f'健康检查执行失败: {str(e)}'],
                'recommendations': ['检查容器配置和状态']
            }


class DependencyResolver:
    """依赖解析器 - 将复杂的依赖解析逻辑分离出来"""

    def __init__(self, container: 'DependencyContainer'):
        self.container = container

    def resolve_constructor_params(self, descriptor: ServiceDescriptor) -> Dict[str, Any]:
        """解析构造函数参数"""
        sig = inspect.signature(descriptor.implementation.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            params[name] = self._resolve_parameter(descriptor, name, param)

        return params

    def _resolve_parameter(self, descriptor: ServiceDescriptor, param_name: str, param: inspect.Parameter) -> Any:
        """解析单个参数"""
        if param.annotation != inspect.Parameter.empty:
            try:
                return self._resolve_annotated_parameter(descriptor, param_name, param)
            except Exception:
                return self._get_parameter_default(param, param_name, descriptor.name)
        else:
            return self._get_parameter_default(param, param_name, descriptor.name)

    def _resolve_annotated_parameter(self, descriptor: ServiceDescriptor, param_name: str, param: inspect.Parameter) -> Any:
        """解析带类型注解的参数"""
        dependency_name = self._get_dependency_name(param.annotation)

        # 尝试直接解析依赖
        resolved_value = self._try_resolve_direct_dependency(dependency_name)
        if resolved_value is not None:
            return resolved_value

        # 使用默认值或抛出异常
        return self._get_parameter_default(param, param_name, descriptor.name)

    def _try_resolve_direct_dependency(self, dependency_name: str) -> Optional[Any]:
        """尝试直接解析依赖"""
        # 直接查找服务
        if dependency_name in self.container._service_descriptors:
            return self.container._resolve_without_lock(dependency_name)

        # 通过类型查找服务
        found_service = self._find_service_by_type(dependency_name)
        if found_service:
            return self.container._resolve_without_lock(found_service)

        return None

    def _get_dependency_name(self, annotation) -> str:
        """获取依赖名称"""
        return annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)

    def _find_service_by_type(self, dependency_name: str) -> Optional[str]:
        """通过类型查找服务"""
        for service_name, service_desc in self.container._service_descriptors.items():
            if self._matches_service_type(service_desc, dependency_name):
                return service_name
        return None

    def _matches_service_type(self, service_desc: ServiceDescriptor, dependency_name: str) -> bool:
        """检查服务是否匹配依赖类型"""
        return (
            (service_desc.service_type and
             hasattr(service_desc.service_type, '__name__') and
             service_desc.service_type.__name__ == dependency_name) or
            (service_desc.implementation and
             hasattr(service_desc.implementation, '__name__') and
             service_desc.implementation.__name__ == dependency_name)
        )

    def _get_parameter_default(self, param: inspect.Parameter, param_name: str, service_name: str) -> Any:
        """获取参数默认值"""
        if param.default == inspect.Parameter.empty:
            raise ContainerException(f"无法解析依赖 {param_name} for {service_name}")
        return param.default


class InstanceCreator:
    """实例创建器 - 负责创建服务实例"""

    def __init__(self, container: 'DependencyContainer'):
        self.container = container
        self.dependency_resolver = DependencyResolver(container)

    def create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                return self._create_from_factory(descriptor)
            elif descriptor.implementation:
                return self._create_from_implementation(descriptor)
            elif descriptor.service_type:
                return self._create_from_service_type(descriptor)
            else:
                raise ContainerException(f"服务 {descriptor.name} 没有实现或工厂")

        except Exception as e:
            self._handle_creation_error(descriptor, e)
            raise ContainerException(f"创建服务实例失败 {descriptor.name}: {str(e)}") from e

    def _create_from_factory(self, descriptor: ServiceDescriptor) -> Any:
        """从工厂方法创建实例"""
        return descriptor.factory()

    def _create_from_implementation(self, descriptor: ServiceDescriptor) -> Any:
        """从实现类创建实例"""
        if isinstance(descriptor.implementation, type):
            # 解析构造函数参数并创建实例
            constructor_params = self.dependency_resolver.resolve_constructor_params(descriptor)
            return descriptor.implementation(**constructor_params)
        else:
            # 已经是实例，直接返回
            return descriptor.implementation

    def _create_from_service_type(self, descriptor: ServiceDescriptor) -> Any:
        """从服务类型创建实例"""
        return descriptor.service_type()

    def _handle_creation_error(self, descriptor: ServiceDescriptor, error: Exception):
        """处理创建错误"""
        descriptor.metrics.error_count += 1
        descriptor.metrics.last_error_time = datetime.now()
        descriptor.status = ServiceStatus.ERROR
        logger.error(f"创建服务实例失败 {descriptor.name}: {error}")

    def _perform_health_check(self, descriptor: ServiceDescriptor):
        """执行健康检查"""
        if not descriptor.health_check:
            return

        try:
            start_time = time.time()
            # 获取服务实例
            instance = None
            if descriptor.lifecycle == Lifecycle.SINGLETON and descriptor.instance:
                instance = descriptor.instance
            elif descriptor.lifecycle == Lifecycle.TRANSIENT:
                # 对于瞬时服务，创建一个临时实例进行检查
                instance = self._create_instance(descriptor)
            elif descriptor.lifecycle == Lifecycle.SCOPED:
                # 对于作用域服务，尝试获取当前作用域的实例
                if self._current_scope and self._current_scope.has_service(descriptor.name):
                    instance = self._current_scope.get_service(descriptor.name)

            if instance is None:
                # 如果无法获取实例，跳过健康检查
                return

            result = descriptor.health_check(instance)
            end_time = time.time()

            health_result = HealthCheckResult(
                service_name=descriptor.name,
                is_healthy=bool(result),
                response_time=end_time - start_time
            )

            if self._health_monitor:
                self._health_monitor.update_health_result(descriptor.name, health_result)

            descriptor.health_status = ServiceHealth.HEALTHY if result else ServiceHealth.UNHEALTHY
            descriptor.last_health_check = time.time()

        except Exception as e:
            descriptor.health_status = ServiceHealth.UNHEALTHY
            descriptor.error_count += 1
            logger.error(f"健康检查失败: {descriptor.name}, 错误: {e}")

    def _has_circular_dependency(self, service_name: str, visited: Set[str] = None) -> bool:
        """检查循环依赖"""
        if visited is None:
            visited = set()

        if service_name in visited:
            return True

        visited.add(service_name)

        if service_name not in self._service_descriptors:
            return False

        descriptor = self._service_descriptors[service_name]

        for dep_name in descriptor.dependencies:
            if self._has_circular_dependency(dep_name, visited.copy()):
                return True

        return False

    def has(self, name: str) -> bool:
        """检查服务是否存在"""
        return name in self._service_descriptors

    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        if name not in self._service_descriptors:
            return None

        descriptor = self._service_descriptors[name]
        return {
            'name': descriptor.name,
            'service_type': descriptor.service_type,
            'lifecycle': descriptor.lifecycle.value,
            'version': descriptor.version,
            'dependencies': descriptor.dependencies,
            'status': descriptor.status.value,
            'health_status': descriptor.health_status.value,
            'created_time': descriptor.created_time,
            'last_health_check': descriptor.last_health_check,
            'error_count': descriptor.error_count,
            'description': descriptor.description,
            'tags': descriptor.tags,
            'metrics': {
                'resolve_count': descriptor.metrics.resolve_count,
                'avg_resolve_time': descriptor.metrics.avg_resolve_time,
                'success_rate': descriptor.metrics.success_rate,
                'error_count': descriptor.metrics.error_count
            }
        }

    def list_services(self) -> List[Dict[str, Any]]:
        """列出所有服务"""
        return [self.get_service_info(name) for name in self._service_descriptors.keys()]

    def get_services_by_type(self, service_type: Type) -> List[str]:
        """根据类型获取服务"""
        return [name for name, descriptor in self._service_descriptors.items()
                if descriptor.service_type == service_type]

    def get_services_by_lifecycle(self, lifecycle: Lifecycle) -> List[str]:
        """根据生命周期获取服务"""
        return [name for name, descriptor in self._service_descriptors.items()
                if descriptor.lifecycle == lifecycle]

    @contextmanager
    def scope(self, scope_id: str):
        """创建作用域上下文"""
        scope = ServiceScope(scope_id)
        self.enter_scope(scope)
        try:
            yield scope
        finally:
            self.exit_scope()

    def create_scope(self, scope_id: str = None) -> ServiceScope:
        """创建作用域"""
        if scope_id is None:
            scope_id = generate_id("scope")
        return ServiceScope(scope_id)

    def enter_scope(self, scope: ServiceScope):
        """进入作用域"""
        self._current_scope = scope

    def exit_scope(self):
        """退出作用域"""
        if self._current_scope:
            # 清理作用域服务
            scope_id = self._current_scope.scope_id
            keys_to_remove = [key for key in self._scoped_instances.keys()
                              if key[1] == scope_id]
            for key in keys_to_remove:
                del self._scoped_instances[key]

            self._current_scope = None

    def check_health(self, service_name: str) -> ServiceHealth:
        """检查服务健康状态"""
        if service_name not in self._service_descriptors:
            return ServiceHealth.UNKNOWN

        descriptor = self._service_descriptors[service_name]

        if descriptor.health_check:
            self._perform_health_check(descriptor)

        return descriptor.health_status

    def check_all_health(self) -> Dict[str, ServiceHealth]:
        """检查所有服务健康状态"""
        results = {}
        for name in self._service_descriptors.keys():
            results[name] = self.check_health(name)
        return results

    def get_unhealthy_services(self) -> List[str]:
        """获取不健康的服务"""
        unhealthy = []
        for name, descriptor in self._service_descriptors.items():
            if descriptor.health_status == ServiceHealth.UNHEALTHY:
                unhealthy.append(name)
        return unhealthy

    def restart_service(self, name: str) -> bool:
        """重启服务"""
        try:
            if name not in self._service_descriptors:
                return False

            descriptor = self._service_descriptors[name]

            # 清理现有实例
            if descriptor.lifecycle == Lifecycle.SINGLETON:
                descriptor.instance = None
            elif descriptor.lifecycle == Lifecycle.SCOPED:
                # 清理作用域实例
                keys_to_remove = [key for key in self._scoped_instances.keys()
                                  if key[0] == name]
                for key in keys_to_remove:
                    del self._scoped_instances[key]

            # 重置状态
            descriptor.status = ServiceStatus.UNKNOWN
            descriptor.health_status = ServiceHealth.UNKNOWN
            descriptor.error_count = 0

            logger.info(f"服务已重启: {name}")
            return True

        except Exception as e:
            logger.error(f"重启服务失败: {name}, 错误: {e}")
            return False

    def auto_discover_services(self, module_path: str):
        """自动发现服务"""
        if not self._auto_discovery:
            logger.warning("自动发现功能未启用")
            return

        discovered_services = self._auto_discovery.discover_services(module_path)
        logger.info(f"发现 {len(discovered_services)} 个服务")
        return discovered_services

    def auto_register_discovered_services(self, module_path: str):
        """自动注册发现的服务"""
        if not self._auto_discovery:
            logger.warning("自动发现功能未启用")
            return

        self._auto_discovery.auto_register_discovered_services(module_path)

    def remove_service(self, name: str) -> bool:
        """移除服务"""
        try:
            with self._lock:
                if name not in self._service_descriptors:
                    return False

                descriptor = self._service_descriptors[name]

                # 清理实例
                if descriptor.lifecycle == Lifecycle.SINGLETON:
                    descriptor.instance = None
                elif descriptor.lifecycle == Lifecycle.SCOPED:
                    keys_to_remove = [key for key in self._scoped_instances.keys()
                                      if key[0] == name]
                    for key in keys_to_remove:
                        del self._scoped_instances[key]

                # 移除服务描述符
                del self._service_descriptors[name]
                if name in self._services:
                    del self._services[name]

                # 更新统计信息
                if descriptor.lifecycle == Lifecycle.SINGLETON:
                    self._stats['singleton_services'] -= 1
                elif descriptor.lifecycle == Lifecycle.TRANSIENT:
                    self._stats['transient_services'] -= 1
                elif descriptor.lifecycle == Lifecycle.SCOPED:
                    self._stats['scoped_services'] -= 1

                self._stats['total_services'] -= 1

                logger.info(f"服务已移除: {name}")
                return True

        except Exception as e:
            logger.error(f"移除服务失败: {name}, 错误: {e}")
            return False

    def clear(self):
        """清空容器"""
        with self._lock:
            self._service_descriptors.clear()
            self._services.clear()
            self._singleton_instances.clear()
            self._scoped_instances.clear()
            self._current_scope = None

            # 重置统计信息
            self._stats = {
                'total_services': 0,
                'singleton_services': 0,
                'transient_services': 0,
                'scoped_services': 0,
                'health_checks': 0,
                'failed_health_checks': 0
            }

            logger.info("容器已清空")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'health_summary': self._health_monitor.get_health_summary() if self._health_monitor else {}
        }

    def get_performance_metrics(self, service_name: str) -> Optional[ServiceMetrics]:
        """获取服务性能指标"""
        if service_name not in self._service_descriptors:
            return None
        return self._service_descriptors[service_name].metrics

    def get_all_performance_metrics(self) -> Dict[str, ServiceMetrics]:
        """获取所有服务性能指标"""
        return {name: descriptor.metrics for name, descriptor in self._service_descriptors.items()}

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._initialized:
                return False

            # 检查所有服务的健康状态
            health_results = self.check_all_health()
            unhealthy_count = sum(1 for status in health_results.values()
                                  if status == ServiceHealth.UNHEALTHY)

            return unhealthy_count == 0

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False


def service(name: str = None, lifecycle: Lifecycle = Lifecycle.SINGLETON,


            dependencies: List[str] = None, health_check: Callable = None,
            version: str = "1.0.0", description: str = "", tags: List[str] = None):
    """服务装饰器"""

    def decorator(cls):

        service_name = name or cls.__name__
        cls._service_metadata = {
            'name': service_name,
            'lifecycle': lifecycle,
            'dependencies': dependencies or [],
            'health_check': health_check,
            'version': version,
            'description': description,
            'tags': tags or []
        }
        return cls
    return decorator


def injectable(cls: Type) -> Type:
    """可注入装饰器"""
    cls._is_injectable = True
    return cls


def singleton(cls: Type) -> Type:
    """单例装饰器"""
    cls._is_singleton = True
    return cls


def transient(cls: Type) -> Type:
    """瞬时装饰器"""
    cls._is_transient = True
    return cls


def scoped(cls: Type) -> Type:
    """作用域装饰器"""
    cls._is_scoped = True
    return cls
