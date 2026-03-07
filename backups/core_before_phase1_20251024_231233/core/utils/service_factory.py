"""
RQA2025系统服务工厂
Service Factory for RQA2025 System

实现服务工厂模式，统一管理服务的创建和生命周期
"""

from typing import Dict, Type, Any, Optional, List
from ...unified_exceptions import handle_infrastructure_exceptions, InfrastructureException
import logging

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    服务工厂

    统一管理服务的创建和生命周期，提供依赖注入和配置管理功能
    """

    def __init__(self):
        self._service_registry: Dict[str, Type] = {}
        self._service_instances: Dict[str, Any] = {}
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._service_dependencies: Dict[str, List[str]] = {}
        self._shutdown_order: List[str] = []

    @handle_infrastructure_exceptions
    def register_service(self, service_name: str, service_class: Type,
                         config: Optional[Dict[str, Any]] = None,
                         dependencies: Optional[List[str]] = None) -> None:
        """
        注册服务

        Args:
            service_name: 服务名称
            service_class: 服务类
            config: 默认配置
            dependencies: 依赖的服务列表
        """
        if not isinstance(service_name, str) or not service_name:
            raise ValueError("服务名称必须是非空字符串")

        if not isinstance(service_class, type):
            raise ValueError("服务类必须是有效的类类型")

        self._service_registry[service_name] = service_class
        if config:
            self._service_configs[service_name] = config
        if dependencies:
            self._service_dependencies[service_name] = dependencies

        logger.info(f"服务 '{service_name}' 已注册到服务工厂")

    @handle_infrastructure_exceptions
    def create_service(self, service_name: str,
                       config: Optional[Dict[str, Any]] = None,
                       force_new: bool = False) -> Any:
        """
        创建服务实例

        Args:
            service_name: 服务名称
            config: 运行时配置
            force_new: 是否强制创建新实例

        Returns:
            服务实例
        """
        if service_name not in self._service_registry:
            raise InfrastructureException(f"未注册的服务: {service_name}")

        # 如果已有实例且不强制创建新实例，则返回现有实例
        if not force_new and service_name in self._service_instances:
            return self._service_instances[service_name]

        service_class = self._service_registry[service_name]

        # 合并配置
        final_config = {}
        if service_name in self._service_configs:
            final_config.update(self._service_configs[service_name])
        if config:
            final_config.update(config)

        # 解析依赖
        dependencies = self._resolve_dependencies(service_name)

        # 创建实例
        try:
            instance = service_class()

            # 注入依赖
            if dependencies:
                self._inject_dependencies(instance, dependencies)

            # 初始化服务
            if hasattr(instance, 'initialize'):
                success = instance.initialize(final_config)
                if not success:
                    raise InfrastructureException(f"服务初始化失败: {service_name}")

            self._service_instances[service_name] = instance

            # 添加到关闭顺序（逆序）
            if service_name not in self._shutdown_order:
                self._shutdown_order.insert(0, service_name)

            logger.info(f"服务 '{service_name}' 实例创建成功")
            return instance

        except Exception as e:
            logger.error(f"服务 '{service_name}' 创建失败: {e}")
            raise InfrastructureException(f"服务创建失败: {service_name}") from e

    def _resolve_dependencies(self, service_name: str) -> Dict[str, Any]:
        """
        解析服务依赖

        Args:
            service_name: 服务名称

        Returns:
            依赖实例字典
        """
        dependencies = {}
        if service_name in self._service_dependencies:
            for dep_name in self._service_dependencies[service_name]:
                if dep_name not in self._service_instances:
                    # 递归创建依赖服务
                    self.create_service(dep_name)
                dependencies[dep_name] = self._service_instances[dep_name]
        return dependencies

    def _inject_dependencies(self, instance: Any, dependencies: Dict[str, Any]) -> None:
        """
        注入依赖到服务实例

        Args:
            instance: 服务实例
            dependencies: 依赖字典
        """
        if hasattr(instance, 'set_dependencies'):
            instance.set_dependencies(dependencies)
        else:
            # 尝试通过属性注入
            for dep_name, dep_instance in dependencies.items():
                attr_name = f"_{dep_name}"
                if hasattr(instance, attr_name):
                    setattr(instance, attr_name, dep_instance)

    def get_service(self, service_name: str) -> Optional[Any]:
        """
        获取服务实例

        Args:
            service_name: 服务名称

        Returns:
            服务实例或None
        """
        return self._service_instances.get(service_name)

    def has_service(self, service_name: str) -> bool:
        """
        检查服务是否已创建

        Args:
            service_name: 服务名称

        Returns:
            是否已创建
        """
        return service_name in self._service_instances

    def list_services(self) -> List[str]:
        """
        列出所有已注册的服务

        Returns:
            服务名称列表
        """
        return list(self._service_registry.keys())

    def list_instances(self) -> List[str]:
        """
        列出所有已创建的服务实例

        Returns:
            服务实例名称列表
        """
        return list(self._service_instances.keys())

    @handle_infrastructure_exceptions
    def shutdown_service(self, service_name: str) -> None:
        """
        关闭指定服务

        Args:
            service_name: 服务名称
        """
        if service_name in self._service_instances:
            instance = self._service_instances[service_name]
            if hasattr(instance, 'shutdown'):
                try:
                    instance.shutdown()
                    logger.info(f"服务 '{service_name}' 已关闭")
                except Exception as e:
                    logger.error(f"服务 '{service_name}' 关闭失败: {e}")

            del self._service_instances[service_name]

            if service_name in self._shutdown_order:
                self._shutdown_order.remove(service_name)

    def shutdown_all(self) -> None:
        """关闭所有服务（按依赖顺序）"""
        logger.info("开始关闭所有服务...")

        # 按注册顺序的逆序关闭
        for service_name in self._shutdown_order.copy():
            self.shutdown_service(service_name)

        self._shutdown_order.clear()
        logger.info("所有服务已关闭")

    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        获取服务信息

        Args:
            service_name: 服务名称

        Returns:
            服务信息字典
        """
        if service_name not in self._service_registry:
            return None

        info = {
            'name': service_name,
            'class': self._service_registry[service_name].__name__,
            'module': self._service_registry[service_name].__module__,
            'has_instance': service_name in self._service_instances,
            'config': self._service_configs.get(service_name, {}),
            'dependencies': self._service_dependencies.get(service_name, [])
        }

        if service_name in self._service_instances:
            instance = self._service_instances[service_name]
            info.update({
                'version': getattr(instance, 'service_version', 'unknown'),
                'status': 'active'
            })

        return info

    def get_factory_status(self) -> Dict[str, Any]:
        """
        获取工厂状态

        Returns:
            工厂状态信息
        """
        return {
            'registered_services': len(self._service_registry),
            'active_instances': len(self._service_instances),
            'shutdown_order': self._shutdown_order.copy(),
            'service_list': self.list_services(),
            'instance_list': self.list_instances()
        }


# 全局服务工厂实例
global_service_factory = ServiceFactory()


class ServiceLocator:
    """
    服务定位器

    提供服务定位功能，支持按类型或名称查找服务
    """

    def __init__(self, factory: ServiceFactory):
        self.factory = factory
        self._service_cache: Dict[Type, Any] = {}

    def get_service_by_type(self, service_type: Type) -> Optional[Any]:
        """
        按类型获取服务

        Args:
            service_type: 服务类型

        Returns:
            服务实例
        """
        if service_type in self._service_cache:
            return self._service_cache[service_type]

        # 查找匹配类型的服务
        for service_name, instance in self.factory._service_instances.items():
            if isinstance(instance, service_type):
                self._service_cache[service_type] = instance
                return instance

        return None

    def get_service_by_interface(self, interface_type: Type) -> List[Any]:
        """
        按接口获取服务列表

        Args:
            interface_type: 接口类型

        Returns:
            服务实例列表
        """
        services = []
        for instance in self.factory._service_instances.values():
            if isinstance(instance, interface_type):
                services.append(instance)
        return services

    def clear_cache(self) -> None:
        """清空缓存"""
        self._service_cache.clear()


# 全局服务定位器
global_service_locator = ServiceLocator(global_service_factory)


# 便捷函数
def register_service(service_name: str, service_class: Type,
                     config: Optional[Dict[str, Any]] = None,
                     dependencies: Optional[List[str]] = None) -> None:
    """
    注册服务到全局工厂

    Args:
        service_name: 服务名称
        service_class: 服务类
        config: 默认配置
        dependencies: 依赖服务列表
    """
    global_service_factory.register_service(service_name, service_class, config, dependencies)


def get_service(service_name: str) -> Optional[Any]:
    """
    从全局工厂获取服务

    Args:
        service_name: 服务名称

    Returns:
        服务实例
    """
    return global_service_factory.get_service(service_name)


def create_service(service_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    从全局工厂创建服务

    Args:
        service_name: 服务名称
        config: 运行时配置

    Returns:
        服务实例
    """
    return global_service_factory.create_service(service_name, config)


def shutdown_services() -> None:
    """关闭所有全局服务"""
    global_service_factory.shutdown_all()
