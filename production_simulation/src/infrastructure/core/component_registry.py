"""
component_registry 模块

提供 component_registry 相关功能和接口。
"""

import logging

from typing import Dict, Any, Callable, Optional, Type, TypeVar
"""
基础设施层组件注册表 - Phase 6.1组织架构重构

集中管理基础设施层所有组件的注册和依赖注入，
减少模块间的直接耦合，提高代码的可维护性和可扩展性。
"""

T = TypeVar('T')

logger = logging.getLogger(__name__)


class InfrastructureComponentRegistry:
    """
    基础设施层组件注册表

    职责：
    - 组件注册和发现
    - 依赖注入管理
    - 延迟加载支持
    - 组件生命周期管理
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}

    def register_component(self, name: str, factory: Callable, singleton: bool = False) -> None:
        """
        注册组件

        Args:
            name: 组件名称
            factory: 组件工厂函数
            singleton: 是否为单例模式
        """
        self._registry[name] = factory
        if singleton and name in self._instances:
            del self._instances[name]  # 清除旧实例，允许重新创建

        logger.debug(f"组件 '{name}' 已注册 (singleton: {singleton})")

    def get_component(self, name: str) -> Any:
        """
        获取组件实例

        Args:
            name: 组件名称

        Returns:
            组件实例

        Raises:
            KeyError: 如果组件未注册
        """
        if name not in self._registry:
            raise KeyError(f"组件 '{name}' 未注册")

        # 如果是单例且已创建，直接返回
        if name in self._singletons:
            return self._singletons[name]

        # 创建新实例
        try:
            instance = self._registry[name]()
            self._instances[name] = instance

            # 如果是单例，保存到单例缓存
            if hasattr(instance, '_is_singleton') and instance._is_singleton:
                self._singletons[name] = instance

            logger.debug(f"组件 '{name}' 实例已创建")
            return instance

        except Exception as e:
            logger.error(f"创建组件 '{name}' 失败: {e}")
            raise

    def has_component(self, name: str) -> bool:
        """检查组件是否已注册"""
        return name in self._registry

    def list_components(self) -> list:
        """列出所有已注册的组件"""
        return list(self._registry.keys())

    def unregister_component(self, name: str) -> bool:
        """
        注销组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功注销
        """
        if name in self._registry:
            del self._registry[name]
            if name in self._instances:
                del self._instances[name]
            if name in self._singletons:
                del self._singletons[name]
            logger.debug(f"组件 '{name}' 已注销")
            return True
        return False

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除组件实例缓存

        Args:
            name: 组件名称，如果为None则清除所有缓存
        """
        if name is None:
            self._instances.clear()
            self._singletons.clear()
            logger.debug("所有组件缓存已清除")
        else:
            if name in self._instances:
                del self._instances[name]
            if name in self._singletons:
                del self._singletons[name]
            logger.debug(f"组件 '{name}' 缓存已清除")


class ComponentFactory:
    """
    组件工厂基类

    为基础设施层组件提供统一的创建接口
    """

    def __init__(self, registry: InfrastructureComponentRegistry):
        self.registry = registry

    def create_component(self, component_type: Type[T], *args, **kwargs) -> T:
        """
        创建组件

        Args:
            component_type: 组件类型
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            创建的组件实例
        """
        return component_type(*args, **kwargs)

    def register_factory(self, name: str, factory_func: Callable) -> None:
        """
        注册工厂函数

        Args:
            name: 组件名称
            factory_func: 工厂函数
        """
        self.registry.register_component(name, factory_func)


# 全局组件注册表实例
_global_registry = InfrastructureComponentRegistry()


def get_global_registry() -> InfrastructureComponentRegistry:
    """获取全局组件注册表"""
    return _global_registry


def register_infrastructure_component(name: str, factory: Callable, singleton: bool = False) -> None:
    """
    注册基础设施层组件到全局注册表

    Args:
        name: 组件名称
        factory: 组件工厂函数
        singleton: 是否为单例
    """
    _global_registry.register_component(name, factory, singleton)


def get_infrastructure_component(name: str) -> Any:
    """
    从全局注册表获取基础设施层组件

    Args:
        name: 组件名称

    Returns:
        组件实例
    """
    return _global_registry.get_component(name)
