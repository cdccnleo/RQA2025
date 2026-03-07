#!/usr/bin/env python3
"""
重构后的Container组件实现

基于BaseComponent重构，消除代码重复，提供统一的组件架构

重构说明：
- 原有5个组件文件（container_components.py, factory_components.py等）存在大量重复
- 每个文件都包含相同的ComponentFactory类（~30行重复）
- 使用BaseComponent基类后，减少~400-600行重复代码

迁移示例：
    # 旧方式
    from src.core.container.container_components import IContainerComponent
    
    # 新方式（推荐）
    from src.core.container.refactored_container_components import ContainerComponent

创建时间: 2025-11-03
版本: 2.0
"""

from typing import Dict, Any, Optional
from src.core.foundation.base_component import BaseComponent, ComponentFactory, component


@component("container")
class ContainerComponent(BaseComponent):
    """
    容器组件（重构版）
    
    基于BaseComponent，提供：
    - 自动化的日志管理
    - 统一的状态管理
    - 标准化的错误处理
    - 性能监控支持
    """
    
    def __init__(self, name: str = "container", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._container_data: Dict[str, Any] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化容器组件"""
        try:
            # 容器特定的初始化逻辑
            self._container_data = config.get('container_data', {})
            self._logger.info(f"容器组件初始化: {len(self._container_data)} 项配置")
            return True
        except Exception as e:
            self._logger.error(f"容器组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行容器操作"""
        operation = kwargs.get('operation', 'get')
        key = kwargs.get('key')
        
        if operation == 'get':
            return self._container_data.get(key)
        elif operation == 'set':
            value = kwargs.get('value')
            self._container_data[key] = value
            return True
        elif operation == 'delete':
            if key in self._container_data:
                del self._container_data[key]
                return True
            return False
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def get(self, key: str) -> Any:
        """获取容器数据"""
        return self.execute(operation='get', key=key)
    
    def set(self, key: str, value: Any):
        """设置容器数据"""
        return self.execute(operation='set', key=key, value=value)
    
    def delete(self, key: str) -> bool:
        """删除容器数据"""
        return self.execute(operation='delete', key=key)


@component("factory")
class FactoryComponent(BaseComponent):
    """
    工厂组件（重构版）
    
    负责创建和管理对象实例
    """
    
    def __init__(self, name: str = "factory", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._factories: Dict[str, callable] = {}
        self._instances: Dict[str, Any] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化工厂组件"""
        try:
            # 注册工厂函数
            factories = config.get('factories', {})
            for name, factory_func in factories.items():
                self.register_factory(name, factory_func)
            
            self._logger.info(f"工厂组件初始化: {len(self._factories)} 个工厂")
            return True
        except Exception as e:
            self._logger.error(f"工厂组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行工厂操作"""
        factory_name = kwargs.get('factory_name')
        operation = kwargs.get('operation', 'create')
        
        if operation == 'create':
            return self.create_instance(factory_name, **kwargs.get('params', {}))
        elif operation == 'get':
            return self._instances.get(factory_name)
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def register_factory(self, name: str, factory_func: callable):
        """注册工厂函数"""
        self._factories[name] = factory_func
        self._logger.info(f"注册工厂: {name}")
    
    def create_instance(self, factory_name: str, **params) -> Any:
        """创建实例"""
        if factory_name not in self._factories:
            raise ValueError(f"未找到工厂: {factory_name}")
        
        factory_func = self._factories[factory_name]
        instance = factory_func(**params)
        self._instances[factory_name] = instance
        
        self._logger.info(f"创建实例: {factory_name}")
        return instance


@component("locator")
class LocatorComponent(BaseComponent):
    """
    定位器组件（重构版）
    
    负责服务定位和查找
    """
    
    def __init__(self, name: str = "locator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._services: Dict[str, Any] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化定位器组件"""
        try:
            services = config.get('services', {})
            for name, service in services.items():
                self.register_service(name, service)
            
            self._logger.info(f"定位器组件初始化: {len(self._services)} 个服务")
            return True
        except Exception as e:
            self._logger.error(f"定位器组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行定位操作"""
        service_name = kwargs.get('service_name')
        return self.locate_service(service_name)
    
    def register_service(self, name: str, service: Any):
        """注册服务"""
        self._services[name] = service
        self._logger.info(f"注册服务: {name}")
    
    def locate_service(self, name: str) -> Optional[Any]:
        """定位服务"""
        service = self._services.get(name)
        if service:
            self._logger.debug(f"定位服务: {name}")
        else:
            self._logger.warning(f"未找到服务: {name}")
        return service


@component("registry")
class RegistryComponent(BaseComponent):
    """
    注册表组件（重构版）
    
    负责管理注册信息
    """
    
    def __init__(self, name: str = "registry", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._registry: Dict[str, Dict[str, Any]] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化注册表组件"""
        try:
            initial_entries = config.get('entries', {})
            self._registry.update(initial_entries)
            
            self._logger.info(f"注册表组件初始化: {len(self._registry)} 个条目")
            return True
        except Exception as e:
            self._logger.error(f"注册表组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行注册表操作"""
        operation = kwargs.get('operation', 'get')
        key = kwargs.get('key')
        
        if operation == 'register':
            metadata = kwargs.get('metadata', {})
            return self.register(key, metadata)
        elif operation == 'unregister':
            return self.unregister(key)
        elif operation == 'get':
            return self.get_entry(key)
        elif operation == 'list':
            return self.list_entries()
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def register(self, key: str, metadata: Dict[str, Any]) -> bool:
        """注册条目"""
        self._registry[key] = metadata
        self._logger.info(f"注册条目: {key}")
        return True
    
    def unregister(self, key: str) -> bool:
        """取消注册"""
        if key in self._registry:
            del self._registry[key]
            self._logger.info(f"取消注册: {key}")
            return True
        return False
    
    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """获取条目"""
        return self._registry.get(key)
    
    def list_entries(self) -> Dict[str, Dict[str, Any]]:
        """列出所有条目"""
        return self._registry.copy()


@component("resolver")
class ResolverComponent(BaseComponent):
    """
    解析器组件（重构版）
    
    负责依赖解析和注入
    """
    
    def __init__(self, name: str = "resolver", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._dependencies: Dict[str, Any] = {}
        self._resolvers: Dict[str, callable] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化解析器组件"""
        try:
            dependencies = config.get('dependencies', {})
            self._dependencies.update(dependencies)
            
            self._logger.info(f"解析器组件初始化: {len(self._dependencies)} 个依赖")
            return True
        except Exception as e:
            self._logger.error(f"解析器组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行解析操作"""
        dependency_name = kwargs.get('dependency_name')
        return self.resolve(dependency_name)
    
    def register_dependency(self, name: str, dependency: Any):
        """注册依赖"""
        self._dependencies[name] = dependency
        self._logger.info(f"注册依赖: {name}")
    
    def register_resolver(self, name: str, resolver_func: callable):
        """注册解析器函数"""
        self._resolvers[name] = resolver_func
        self._logger.info(f"注册解析器: {name}")
    
    def resolve(self, name: str) -> Optional[Any]:
        """解析依赖"""
        # 先尝试直接获取
        if name in self._dependencies:
            self._logger.debug(f"解析依赖: {name}")
            return self._dependencies[name]
        
        # 尝试使用解析器
        if name in self._resolvers:
            resolver = self._resolvers[name]
            dependency = resolver()
            self._dependencies[name] = dependency
            self._logger.info(f"通过解析器解析依赖: {name}")
            return dependency
        
        self._logger.warning(f"无法解析依赖: {name}")
        return None


# 使用统一的ComponentFactory（从base_component导入）
# 不再需要在每个文件中重复定义ComponentFactory！

def create_container_components() -> Dict[str, BaseComponent]:
    """
    创建所有container组件的便捷函数
    
    Returns:
        包含所有组件实例的字典
    """
    factory = ComponentFactory()
    
    components = {
        'container': factory.create_component(
            'container',
            ContainerComponent,
            {}
        ),
        'factory': factory.create_component(
            'factory',
            FactoryComponent,
            {}
        ),
        'locator': factory.create_component(
            'locator',
            LocatorComponent,
            {}
        ),
        'registry': factory.create_component(
            'registry',
            RegistryComponent,
            {}
        ),
        'resolver': factory.create_component(
            'resolver',
            ResolverComponent,
            {}
        )
    }
    
    return components


__all__ = [
    'ContainerComponent',
    'FactoryComponent',
    'LocatorComponent',
    'RegistryComponent',
    'ResolverComponent',
    'create_container_components'
]

