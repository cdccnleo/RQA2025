"""
RQA2025系统服务工厂

实现服务工厂模式，统一管理服务的创建和生命周期。

作者: 系统架构师
创建时间: 2025-01-28
版本: 2.0.0
"""

from typing import Dict, Type, Any, Optional, List, Callable
from ...unified_exceptions import handle_infrastructure_exceptions, InfrastructureException
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    service_class: Type
    config: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    singleton: bool = True
    auto_start: bool = True


class ServiceFactory:
    """
    服务工厂

    统一管理服务的创建和生命周期，提供依赖注入和配置管理功能
    """

    def __init__(self) -> None:
        self._service_registry: Dict[str, Type] = {}
        self._service_instances: Dict[str, Any] = {}
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._service_dependencies: Dict[str, List[str]] = {}
        self._shutdown_order: List[str] = []
        self._service_configs_data: Dict[str, ServiceConfig] = {}

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

        # 保存完整的配置数据
        self._service_configs_data[service_name] = ServiceConfig(
            name=service_name,
            service_class=service_class,
            config=config,
            dependencies=dependencies
        )

    def register_service_with_config(self, service_config: ServiceConfig) -> None:
        """使用服务配置对象注册服务

        Args:
            service_config: 服务配置对象
        """
        self.register_service(
            service_name=service_config.name,
            service_class=service_config.service_class,
            config=service_config.config,
            dependencies=service_config.dependencies
        )

    @handle_infrastructure_exceptions
    def create_service(self, service_name: str,
                       config: Optional[Dict[str, Any]] = None,
                       force_new: bool = False) -> Any:
        """
        创建服务实例 - 重构版：拆分职责

        Args:
            service_name: 服务名称
            config: 运行时配置
            force_new: 是否强制创建新实例

        Returns:
            服务实例
        """
        # 步骤1: 验证服务是否注册
        self._validate_service_registered(service_name)

        # 步骤2: 检查是否返回现有实例
        if not force_new and service_name in self._service_instances:
            return self._service_instances[service_name]

        # 步骤3: 准备服务和依赖
        service_class = self._service_registry[service_name]
        final_config = self._merge_service_config(service_name, config)
        dependencies = self._resolve_dependencies(service_name)

        # 步骤4: 创建和初始化实例
        return self._create_and_initialize_service(service_name, service_class, final_config, dependencies)

    def _validate_service_registered(self, service_name: str):
        """验证服务是否已注册 - 职责：验证服务注册"""
        if service_name not in self._service_registry:
            raise InfrastructureException(f"未注册的服务: {service_name}")

    def _merge_service_config(self, service_name: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """合并服务配置 - 职责：合并默认配置和运行时配置"""
        final_config = {}
        if service_name in self._service_configs:
            final_config.update(self._service_configs[service_name])
        if config:
            final_config.update(config)
        return final_config

    def _create_and_initialize_service(self, service_name: str, service_class: type,
                                       final_config: Dict[str, Any], dependencies: Dict[str, Any]) -> Any:
        """创建并初始化服务实例 - 职责：创建实例并完成初始化"""
        try:
            # 创建实例
            instance = service_class()

            # 注入依赖
            if dependencies:
                self._inject_dependencies(instance, dependencies)

            # 初始化服务
            self._initialize_service_instance(instance, service_name, final_config)

            # 注册实例
            self._register_service_instance(service_name, instance)

            logger.info(f"服务 '{service_name}' 实例创建成功")
            return instance

        except Exception as e:
            logger.error(f"服务 '{service_name}' 创建失败: {e}")
            raise InfrastructureException(f"服务创建失败: {service_name}") from e

    def _initialize_service_instance(self, instance: Any, service_name: str, config: Dict[str, Any]):
        """初始化服务实例 - 职责：调用服务的初始化方法"""
        if hasattr(instance, 'initialize'):
            success = instance.initialize(config)
            if not success:
                raise InfrastructureException(f"服务初始化失败: {service_name}")

    def _register_service_instance(self, service_name: str, instance: Any):
        """注册服务实例 - 职责：将实例添加到实例字典和关闭顺序"""
        self._service_instances[service_name] = instance
        if service_name not in self._shutdown_order:
            self._shutdown_order.insert(0, service_name)

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
