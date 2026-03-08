"""
RQA2025系统服务工厂

实现服务工厂模式，统一管理服务的创建和生命周期。

作者: 系统架构师
创建时间: 2025-01-28
版本: 2.0.0
"""

from typing import Dict, Type, Any, Optional, List, Callable
from ..unified_exceptions import handle_infrastructure_exceptions, InfrastructureException
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class CircularDependencyError(InfrastructureException):
    """循环依赖错误"""


class ServiceCreationError(InfrastructureException):
    """服务创建错误"""

    def __init__(self, message: str, dependency_chain: List[str]):
        super().__init__(f"检测到循环依赖: {' -> '.join(dependency_chain)}")
        self.dependency_chain = dependency_chain


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
                         dependencies: Optional[List[str]] = None,
                         singleton: bool = True) -> None:
        """
        注册服务

        Args:
            service_name: 服务名称
            service_class: 服务类
            config: 默认配置
            dependencies: 依赖的服务列表
            singleton: 是否为单例模式
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
            dependencies=dependencies,
            singleton=singleton
        )

    @handle_infrastructure_exceptions
    def start_service(self, service_name: str) -> bool:
        """
        启动服务

        Args:
            service_name: 服务名称

        Returns:
            启动是否成功
        """
        if service_name not in self._service_instances:
            self.create_service(service_name)

        service = self._service_instances[service_name]

        # 调用生命周期钩子
        if hasattr(service, 'pre_start'):
            try:
                service.pre_start()
            except Exception as e:
                logger.warning(f"服务 {service_name} pre_start 钩子执行失败: {e}")

        # 启动服务
        result = True
        if hasattr(service, 'start'):
            result = service.start()

        # 调用后启动钩子
        if hasattr(service, 'post_start'):
            try:
                service.post_start()
            except Exception as e:
                logger.warning(f"服务 {service_name} post_start 钩子执行失败: {e}")

        return result

    @handle_infrastructure_exceptions
    def stop_service(self, service_name: str) -> bool:
        """
        停止服务

        Args:
            service_name: 服务名称

        Returns:
            停止是否成功
        """
        if service_name not in self._service_instances:
            return True

        service = self._service_instances[service_name]

        # 调用生命周期钩子
        if hasattr(service, 'pre_stop'):
            try:
                service.pre_stop()
            except Exception as e:
                logger.warning(f"服务 {service_name} pre_stop 钩子执行失败: {e}")

        # 停止服务
        result = True
        if hasattr(service, 'stop'):
            result = service.stop()

        # 调用后停止钩子
        if hasattr(service, 'post_stop'):
            try:
                service.post_stop()
            except Exception as e:
                logger.warning(f"服务 {service_name} post_stop 钩子执行失败: {e}")

        return result

    @handle_infrastructure_exceptions
    def start_all_services(self) -> Dict[str, bool]:
        """
        启动所有已注册的服务

        Returns:
            服务启动结果字典
        """
        results = {}
        for service_name in self._service_registry.keys():
            try:
                results[service_name] = self.start_service(service_name)
            except Exception as e:
                logger.error(f"启动服务 '{service_name}' 失败: {e}")
                results[service_name] = False
        return results

    @handle_infrastructure_exceptions
    def stop_all_services(self) -> Dict[str, bool]:
        """
        停止所有已创建的服务

        Returns:
            服务停止结果字典
        """
        results = {}
        for service_name in self._service_instances.keys():
            try:
                results[service_name] = self.stop_service(service_name)
            except Exception as e:
                logger.error(f"停止服务 '{service_name}' 失败: {e}")
                results[service_name] = False
        return results

    def get_service(self, service_name: str) -> Any:
        """
        获取服务实例

        Args:
            service_name: 服务名称

        Returns:
            服务实例

        Raises:
            SystemException: 如果服务未注册
        """
        if service_name not in self._service_registry:
            from src.core.foundation.exceptions.unified_exceptions import SystemException
            raise SystemException(f"服务 '{service_name}' 未注册")
        if service_name not in self._service_instances:
            self.create_service(service_name)
        return self._service_instances[service_name]

    def has_service(self, service_name: str) -> bool:
        """
        检查服务是否已注册

        Args:
            service_name: 服务名称

        Returns:
            是否已注册
        """
        return service_name in self._service_registry

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

    def register_service_config(self, config_data) -> None:
        """
        使用配置数据注册服务

        Args:
            config_data: 服务配置数据（字典或ServiceConfig对象）
        """
        if isinstance(config_data, ServiceConfig):
            service_config = config_data
        elif isinstance(config_data, dict):
            service_config = ServiceConfig(**config_data)
        else:
            raise TypeError("config_data must be a dict or ServiceConfig instance")

        self.register_service_with_config(service_config)

    def bulk_register_services(self, services_config) -> None:
        """
        批量注册服务

        Args:
            services_config: 服务配置，可以是字典或列表格式
                            字典格式: {"service_name": {"class": ..., "config": ..., "dependencies": ...}}
                            列表格式: [{"name": "...", "class": ..., "config": ..., "dependencies": ...}]
        """
        if isinstance(services_config, dict):
            # 字典格式处理
            for service_name, service_config in services_config.items():
                try:
                    # 转换为ServiceConfig格式
                    config_dict = {
                        'name': service_name,
                        'service_class': service_config.get('class'),
                        'config': service_config.get('config', {}),
                        'dependencies': service_config.get('dependencies', []),
                        'singleton': service_config.get('singleton', True),
                        'auto_start': service_config.get('auto_start', True)
                    }
                    self.register_service_config(config_dict)
                except Exception as e:
                    logger.error(f"批量注册服务失败: {service_name} - {e}")
                    raise
        elif isinstance(services_config, list):
            # 列表格式处理
            for config_data in services_config:
                try:
                    self.register_service_config(config_data)
                except Exception as e:
                    # 处理不同类型的config_data
                    if isinstance(config_data, dict):
                        service_name = config_data.get('name', 'unknown')
                    elif hasattr(config_data, 'name'):
                        service_name = getattr(config_data, 'name', 'unknown')
                    else:
                        service_name = str(config_data)[:50]  # 限制长度避免日志过长
                    logger.error(f"批量注册服务失败: {service_name} - {e}")
                    raise
        else:
            raise TypeError("services_config must be a dict or list")

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

        # 步骤2: 检查是否返回现有实例 (仅单例模式)
        if not force_new and service_name in self._service_instances:
            service_config = self._service_configs_data.get(service_name)
            if service_config and service_config.singleton:
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
            # 创建实例 - 传递配置参数和依赖
            if dependencies:
                # 如果有依赖，尝试通过构造函数参数注入
                instance = self._create_instance_with_dependencies(service_class, final_config, dependencies)
            else:
                if final_config:
                    instance = service_class(final_config)
                else:
                    instance = service_class()

            # 注入依赖（如果构造函数注入失败）
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

    def _resolve_dependencies(self, service_name: str, visited: Optional[set] = None) -> Dict[str, Any]:
        """
        解析服务依赖

        Args:
            service_name: 服务名称
            visited: 已访问的服务集合（用于检测循环依赖）

        Returns:
            依赖实例字典

        Raises:
            CircularDependencyError: 检测到循环依赖
        """
        if visited is None:
            visited = set()

        if service_name in visited:
            raise CircularDependencyError(
                f"检测到循环依赖: {service_name}",
                list(visited) + [service_name]
            )

        visited.add(service_name)

        dependencies = {}
        if service_name in self._service_dependencies:
            for dep_name in self._service_dependencies[service_name]:
                # 检测循环依赖
                if dep_name in visited:
                    raise CircularDependencyError(
                        f"检测到循环依赖: {dep_name}",
                        list(visited) + [dep_name]
                    )

                if dep_name not in self._service_instances:
                    # 递归创建依赖服务
                    self._resolve_dependencies(dep_name, visited.copy())
                    self.create_service(dep_name)
                dependencies[dep_name] = self._service_instances[dep_name]

        return dependencies

    def _create_instance_with_dependencies(self, service_class: type, config: Dict[str, Any],
                                         dependencies: Dict[str, Any]) -> Any:
        """
        通过构造函数参数创建实例并注入依赖

        Args:
            service_class: 服务类
            config: 配置参数
            dependencies: 依赖字典

        Returns:
            服务实例
        """
        import inspect

        # 获取构造函数签名
        init_signature = inspect.signature(service_class.__init__)
        init_params = list(init_signature.parameters.keys())[1:]  # 跳过 'self'

        # 准备构造函数参数
        kwargs = {}

        # 添加配置参数
        if config:
            for key, value in config.items():
                if key in init_params:
                    kwargs[key] = value

        # 添加依赖参数（按参数名匹配）
        for dep_name, dep_instance in dependencies.items():
            if dep_name in init_params:
                kwargs[dep_name] = dep_instance
            elif f"{dep_name}_service" in init_params:
                # 特殊处理：database -> database_service
                kwargs[f"{dep_name}_service"] = dep_instance

        # 创建实例
        if kwargs:
            return service_class(**kwargs)
        elif config:
            return service_class(config)
        else:
            return service_class()

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
                # 也尝试直接设置属性
                elif hasattr(instance, dep_name):
                    setattr(instance, dep_name, dep_instance)

    def get_service(self, service_name: str) -> Optional[Any]:
        """
        获取服务实例（便捷方法）

        Args:
            service_name: 服务名称

        Returns:
            服务实例或None
        """
        try:
            return self._service_instances.get(service_name) or self.create_service(service_name)
        except Exception:
            return None

    def has_service(self, service_name: str) -> bool:
        """
        检查服务是否已注册

        Args:
            service_name: 服务名称

        Returns:
            是否已注册
        """
        return service_name in self._service_registry

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

    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        获取服务状态

        Args:
            service_name: 服务名称

        Returns:
            服务状态信息
        """
        if service_name not in self._service_registry:
            return None

        status = {
            'name': service_name,
            'registered': True,
            'created': service_name in self._service_instances,
            'config': self._service_configs.get(service_name, {}),
            'dependencies': self._service_dependencies.get(service_name, [])
        }

        if service_name in self._service_instances:
            instance = self._service_instances[service_name]
            status.update({
                'status': 'active',
                'has_start_method': hasattr(instance, 'start'),
                'has_stop_method': hasattr(instance, 'stop'),
                'initialized': getattr(instance, 'initialized', True)
            })

        return status

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

    @handle_infrastructure_exceptions
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        获取服务健康状态

        Args:
            service_name: 服务名称

        Returns:
            服务健康状态信息
        """
        if service_name not in self._service_registry:
            return {
                'status': 'not_registered',
                'service_name': service_name,
                'error': 'Service not registered'
            }

        health_info = {
            'service_name': service_name,
            'status': 'unknown',
            'registered': True,
            'created': service_name in self._service_instances,
            'last_check': datetime.now().isoformat()
        }

        if service_name in self._service_instances:
            instance = self._service_instances[service_name]

            # 检查服务是否有健康检查方法
            if hasattr(instance, 'health_check'):
                try:
                    health_result = instance.health_check()
                    health_info.update(health_result)
                    health_info['status'] = 'healthy' if health_result.get('healthy', True) else 'unhealthy'
                except Exception as e:
                    health_info.update({
                        'status': 'error',
                        'error': str(e),
                        'healthy': False
                    })
            else:
                # 基本健康检查：检查实例是否存在且没有错误状态
                health_info.update({
                    'status': 'healthy',
                    'healthy': True,
                    'method': 'basic_check'
                })

        return health_info

    @handle_infrastructure_exceptions
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        获取服务指标信息

        Args:
            service_name: 服务名称

        Returns:
            服务指标信息
        """
        if service_name not in self._service_registry:
            return {
                'service_name': service_name,
                'error': 'Service not registered'
            }

        metrics_info = {
            'service_name': service_name,
            'registered': True,
            'created': service_name in self._service_instances,
            'collection_time': datetime.now().isoformat()
        }

        if service_name in self._service_instances:
            instance = self._service_instances[service_name]

            # 收集基本指标
            basic_metrics = {
                'instance_type': type(instance).__name__,
                'has_start_method': hasattr(instance, 'start'),
                'has_stop_method': hasattr(instance, 'stop'),
                'initialized': getattr(instance, 'initialized', True)
            }

            # 如果服务有指标收集方法，使用它
            if hasattr(instance, 'get_metrics'):
                try:
                    service_metrics = instance.get_metrics()
                    basic_metrics.update(service_metrics)
                except Exception as e:
                    basic_metrics['metrics_error'] = str(e)

            metrics_info['metrics'] = basic_metrics

        return metrics_info

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取工厂统计信息

        Returns:
            工厂统计信息
        """
        total_registered = len(self._service_registry)
        total_created = len(self._service_instances)
        total_configured = len(self._service_configs)

        # 计算依赖关系统计
        services_with_deps = sum(1 for deps in self._service_dependencies.values() if deps)
        total_dependencies = sum(len(deps) for deps in self._service_dependencies.values())

        return {
            'total_services': total_registered,
            'active_services': total_created,
            'total_instances': total_created,
            'total_configured_services': total_configured,
            'services_with_dependencies': services_with_deps,
            'total_dependencies': total_dependencies,
            'average_dependencies_per_service': total_dependencies / total_registered if total_registered > 0 else 0,
            'singleton_services': sum(1 for config in self._service_configs_data.values() if config.singleton),
            'transient_services': sum(1 for config in self._service_configs_data.values() if not config.singleton),
            'shutdown_order_length': len(self._shutdown_order),
            'service_types': len(set(type(instance).__name__ for instance in self._service_instances.values())),
            'collection_time': datetime.now().isoformat()
        }

    def cleanup(self) -> None:
        """
        清理工厂资源
        """
        logger.info("开始清理服务工厂资源...")

        # 停止所有服务
        self.shutdown_all()

        # 清空所有状态
        self._service_registry.clear()
        self._service_instances.clear()
        self._service_configs.clear()
        self._service_dependencies.clear()
        self._shutdown_order.clear()
        self._service_configs_data.clear()

        logger.info("服务工厂资源清理完成")


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
