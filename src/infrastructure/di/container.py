from typing import Type, TypeVar, Any, Callable, Dict, Union
import inspect

T = TypeVar('T')

class Container:
    """依赖注入容器"""

    _singletons: Dict[str, Any] = {}
    _transients: Dict[str, Callable[..., Any]] = {}
    _instances: Dict[str, Any] = {}

    @classmethod
    def register_singleton(cls, name: str, instance: Any) -> None:
        """注册单例实例
        Args:
            name: 服务名称
            instance: 单例实例
        """
        cls._singletons[name] = instance

    @classmethod
    def register_transient(cls, name: str, factory: Callable[..., Any]) -> None:
        """注册瞬态服务工厂
        Args:
            name: 服务名称
            factory: 服务工厂函数
        """
        cls._transients[name] = factory

    @classmethod
    def resolve(cls, name: str) -> Any:
        """解析依赖项
        Args:
            name: 服务名称
        Returns:
            解析后的服务实例
        """
        if name in cls._singletons:
            return cls._singletons[name]

        if name in cls._transients:
            factory = cls._transients[name]
            return cls._inject_dependencies(factory)

        raise ValueError(f"未注册的服务: {name}")

    @classmethod
    def resolve_type(cls, service_type: Type[T]) -> T:
        """通过类型解析依赖项
        Args:
            service_type: 服务类型
        Returns:
            解析后的服务实例
        """
        for name, instance in cls._singletons.items():
            if isinstance(instance, service_type):
                return instance

        for name, factory in cls._transients.items():
            instance = cls._inject_dependencies(factory)
            if isinstance(instance, service_type):
                return instance

        raise ValueError(f"未注册的服务类型: {service_type.__name__}")

    @classmethod
    def _inject_dependencies(cls, factory: Callable[..., Any]) -> Any:
        """注入依赖项到工厂函数
        Args:
            factory: 工厂函数
        Returns:
            实例化的对象
        """
        sig = inspect.signature(factory)
        kwargs = {}

        for param in sig.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError(f"工厂函数参数缺少类型注解: {param.name}")

            try:
                kwargs[param.name] = cls.resolve_type(param.annotation)
            except ValueError as e:
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"无法解析依赖项: {param.annotation.__name__}"
                    ) from e

        return factory(**kwargs)

    @classmethod
    def clear(cls):
        """清除所有注册的服务"""
        cls._singletons.clear()
        cls._transients.clear()
        cls._instances.clear()
