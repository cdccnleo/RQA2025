"""
依赖注入容器实现

提供完整的依赖注入功能，包括服务注册、解析、生命周期管理等。
"""

from typing import Dict, Any, Optional, Callable, List, Type
from enum import Enum
import threading


class ServiceLifecycle(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceStatus(Enum):
    """服务状态"""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServiceDefinition:
    """服务定义"""
    
    def __init__(
        self,
        name: str,
        service_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        self.name = name
        self.service_type = service_type
        self.factory = factory
        self.instance = instance
        self.lifecycle = lifecycle
        self.dependencies = dependencies or []
        self.metadata = kwargs
        self.status = ServiceStatus.REGISTERED


class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._scopes: List['DependencyContainer'] = []
    
    def register(
        self,
        name: str,
        service: Any = None,
        service_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        lifecycle: str = "singleton",
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """注册服务"""
        with self._lock:
            try:
                lifecycle_enum = ServiceLifecycle(lifecycle.lower())
            except ValueError:
                lifecycle_enum = ServiceLifecycle.SINGLETON
            
            definition = ServiceDefinition(
                name=name,
                service_type=service_type,
                factory=factory,
                instance=service,
                lifecycle=lifecycle_enum,
                dependencies=dependencies,
                **kwargs
            )
            
            self._services[name] = definition
            
            # 如果提供了实例且是单例，直接缓存
            if service is not None and lifecycle_enum == ServiceLifecycle.SINGLETON:
                self._instances[name] = service
            
            return True
    
    def resolve(self, name: str) -> Any:
        """解析服务"""
        with self._lock:
            if name not in self._services:
                raise KeyError(f"Service '{name}' not registered")
            
            definition = self._services[name]
            
            # 单例模式：返回缓存的实例
            if definition.lifecycle == ServiceLifecycle.SINGLETON:
                if name in self._instances:
                    return self._instances[name]
                
                # 创建新实例
                instance = self._create_instance(definition)
                self._instances[name] = instance
                return instance
            
            # 临时模式：每次创建新实例
            elif definition.lifecycle == ServiceLifecycle.TRANSIENT:
                return self._create_instance(definition)
            
            # 作用域模式：在作用域内单例
            else:
                # 简化实现：当前作用域内单例
                if name in self._instances:
                    return self._instances[name]
                instance = self._create_instance(definition)
                self._instances[name] = instance
                return instance
    
    def _create_instance(self, definition: ServiceDefinition) -> Any:
        """创建服务实例"""
        # 如果已有实例，直接返回
        if definition.instance is not None:
            return definition.instance
        
        # 使用工厂方法创建
        if definition.factory is not None:
            # 解析依赖
            deps = []
            for dep_name in definition.dependencies:
                if dep_name in self._services:
                    deps.append(self.resolve(dep_name))
            
            if deps:
                return definition.factory(*deps)
            else:
                return definition.factory()
        
        # 使用类型创建
        if definition.service_type is not None:
            # 解析依赖
            deps = []
            for dep_name in definition.dependencies:
                if dep_name in self._services:
                    deps.append(self.resolve(dep_name))
            
            if deps:
                return definition.service_type(*deps)
            else:
                return definition.service_type()
        
        raise ValueError(f"Cannot create instance for service '{definition.name}'")
    
    def get_service(self, service_name: str) -> Any:
        """获取服务（resolve的别名）"""
        return self.resolve(service_name)
    
    def has_service(self, name: str) -> bool:
        """检查服务是否已注册"""
        return name in self._services
    
    def unregister(self, name: str) -> bool:
        """注销服务"""
        with self._lock:
            if name in self._services:
                del self._services[name]
                if name in self._instances:
                    del self._instances[name]
                return True
            return False
    
    def create_scope(self) -> 'DependencyContainer':
        """创建作用域容器"""
        scope = DependencyContainer()
        scope._services = self._services.copy()
        self._scopes.append(scope)
        return scope
    
    def get_status(self) -> str:
        """获取容器状态"""
        return "active"
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'status': 'active',
            'services_count': len(self._services),
            'instances_count': len(self._instances),
            'scopes_count': len(self._scopes)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'healthy': True,
            'services': len(self._services),
            'instances': len(self._instances)
        }
    
    def clear(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._instances.clear()
            self._scopes.clear()


# ServiceDescriptor别名
ServiceDescriptor = ServiceDefinition

__all__ = [
    'DependencyContainer',
    'ServiceLifecycle',
    'ServiceStatus',
    'ServiceDefinition',
    'ServiceDescriptor'
]

