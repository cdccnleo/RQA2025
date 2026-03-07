
from .config_manager_complete import UnifiedConfigManager
from typing import Dict, Any, Optional, Type, Callable
import logging
#!/usr/bin/env python3
"""
统一配置工厂核心 (拆分自factory.py)

包含UnifiedConfigFactory核心工厂逻辑
"""

from ..interfaces.unified_interface import (
    IConfigManagerComponent as IConfigManager,
    IConfigManagerFactoryComponent as IConfigManagerFactory
)

logger = logging.getLogger(__name__)


class ConfigManagerRegistry:
    """配置管理器注册表"""

    def __init__(self):
        self._manager_types: Dict[str, Type[IConfigManager]] = {}
        self._provider_factories: Dict[str, Callable] = {}

    def register_manager(self, name: str, manager_class: Type[IConfigManager]) -> None:
        """注册管理器类型"""
        if not isinstance(manager_class, type):
            raise ValueError(f"Manager class must be a class, got {type(manager_class)}")
        self._manager_types[name] = manager_class
        logger.info(f"Registered config manager: {name}")

    def register_provider(self, provider_name: str, provider_factory: Callable) -> None:
        """注册提供商工厂"""
        self._provider_factories[provider_name] = provider_factory
        logger.info(f"Registered provider factory: {provider_name}")

    def unregister_manager(self, name: str) -> bool:
        """取消注册管理器类型"""
        if name in self._manager_types:
            del self._manager_types[name]
            logger.info(f"Unregistered config manager: {name}")
            return True
        return False

    def get_manager_class(self, name: str) -> Optional[Type[IConfigManager]]:
        """获取管理器类"""
        return self._manager_types.get(name)

    def get_provider_factory(self, provider_name: str) -> Optional[Callable]:
        """获取提供商工厂"""
        return self._provider_factories.get(provider_name)

    def get_available_managers(self) -> Dict[str, Type[IConfigManager]]:
        """获取所有可用的管理器"""
        return self._manager_types.copy()

    def has_manager(self, name: str) -> bool:
        """检查是否存在指定的管理器"""
        return name in self._manager_types


class ConfigManagerCache:
    """配置管理器缓存"""

    def __init__(self):
        self._cache: Dict[str, IConfigManager] = {}
        self._stats = {
            'created': 0,
            'cached': 0,
            'destroyed': 0
        }

    def get(self, cache_key: str) -> Optional[IConfigManager]:
        """从缓存获取管理器"""
        manager = self._cache.get(cache_key)
        if manager:
            self._stats['cached'] += 1
        return manager

    def put(self, cache_key: str, manager: IConfigManager) -> None:
        """将管理器放入缓存"""
        self._cache[cache_key] = manager
        self._stats['created'] += 1

    def remove(self, cache_key: str) -> bool:
        """从缓存移除管理器"""
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._stats['destroyed'] += 1
            return True
        return False

    def get_all(self) -> Dict[str, IConfigManager]:
        """获取所有缓存的管理器"""
        return self._cache.copy()

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return self._stats.copy()


class ConfigManagerFactory:
    """配置管理器工厂"""

    def __init__(self, registry: ConfigManagerRegistry):
        self.registry = registry

    def create_manager(self, manager_type: str, **kwargs) -> IConfigManager:
        """创建配置管理器"""
        manager_class = self.registry.get_manager_class(manager_type)
        if not manager_class:
            raise ValueError(f"Unknown manager type: {manager_type}")

        try:
            manager = manager_class(**kwargs)
            logger.info(f"Created config manager: {manager_type}")
            return manager
        except Exception as e:
            logger.error(f"Failed to create config manager {manager_type}: {e}")
            raise

    def create_config_manager(self, manager_type: str = "unified",
                              config_path: Optional[str] = None,
                              auto_reload: bool = False,
                              validation_enabled: bool = True,
                              **kwargs) -> IConfigManager:
        """创建配置管理器（详细参数版本）"""
        # 合并参数
        manager_kwargs = {
            'config_path': config_path,
            'auto_reload': auto_reload,
            'validation_enabled': validation_enabled,
            **kwargs
        }

        return self.create_manager(manager_type, **manager_kwargs)


class ConfigFactoryStats:
    """配置工厂统计信息"""

    def __init__(self):
        self._stats = {
            'total_created': 0,
            'total_cached': 0,
            'total_destroyed': 0,
            'active_managers': 0,
            'registered_types': 0
        }

    def increment_created(self):
        """增加创建计数"""
        self._stats['total_created'] += 1
        self._stats['active_managers'] += 1

    def increment_cached(self):
        """增加缓存命中计数"""
        self._stats['total_cached'] += 1

    def increment_destroyed(self):
        """增加销毁计数"""
        self._stats['total_destroyed'] += 1
        self._stats['active_managers'] -= 1

    def set_registered_types(self, count: int):
        """设置注册类型数量"""
        self._stats['registered_types'] = count

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_created': self._stats['total_created'],
            'total_cached': self._stats['total_cached'],
            'total_destroyed': self._stats['total_destroyed'],
            'active_managers': self._stats['active_managers'],
            'registered_types': self._stats['registered_types'],
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        total_requests = self._stats['total_created'] + self._stats['total_cached']
        if total_requests == 0:
            return 0.0
        return self._stats['total_cached'] / total_requests


class UnifiedConfigFactory(IConfigManagerFactory):
    """
    统一配置工厂

    整合了所有配置工厂的功能：
    - 配置管理器注册和创建
    - 实例缓存管理
    - 生命周期管理
    - 类型验证
    - 性能监控
    """

    def __init__(self):
        """初始化统一配置工厂"""
        # 初始化组件
        self.registry = ConfigManagerRegistry()
        self.cache = ConfigManagerCache()
        self.factory = ConfigManagerFactory(self.registry)
        self.stats = ConfigFactoryStats()

        # 兼容性属性
        self._manager_classes = self.registry._manager_types
        self._manager_instances = self.cache._cache
        self._managers = self._manager_classes
        self._provider_factories = self.registry._provider_factories
        self._stats = {
            'created_count': 0,
            'cached_hits': 0,
            'errors': 0
        }
        # 注册默认管理器
        self._register_default_managers()

    def _register_default_managers(self):
        """注册默认的配置管理器类型"""
        try:
            self.register_manager("unified", UnifiedConfigManager)
            self.register_manager("default", UnifiedConfigManager)
            logger.info("Default config managers registered successfully")
        except Exception as e:
            logger.error(f"Failed to register default managers: {e}")

    # ==================== 管理器注册和管理 ====================

    def register_manager(self, name: str, manager_class: Type[IConfigManager]) -> None:
        """
        注册配置管理器类型

        Args:
            name: 管理器类型名称
            manager_class: 管理器类

        Raises:
            ValueError: 当管理器类不符合要求时
        """
        self.registry.register_manager(name, manager_class)
        self.stats.set_registered_types(len(self.registry.get_available_managers()))

    def register_provider(self, provider_name: str, provider_factory: Callable) -> None:
        """
        注册配置提供者工厂

        Args:
        provider_name: 提供者名称
        provider_factory: 提供者工厂函数
        """
        if not callable(provider_factory):
            raise ValueError("Provider factory must be callable")

        self._provider_factories[provider_name] = provider_factory
        logger.info(f"Registered config provider: {provider_name}")

    def unregister_manager(self, name: str) -> bool:
        """
        注销配置管理器类型

        Args:
        name: 管理器类型名称

        Returns:
        bool: 是否成功注销
        """
        if name in self._manager_classes:
            del self._manager_classes[name]
            logger.info(f"Unregistered config manager: {name}")
            return True
        return False

    def get_available_managers(self) -> Dict[str, Type[IConfigManager]]:
        """
        获取所有可用的配置管理器类型

        Returns:
        Dict[str, Type[IConfigManager]]: 管理器类型字典
        """
        return self._manager_classes.copy()

    # ==================== 管理器创建 ====================

    def create_manager(self, manager_type: str = "unified", **kwargs) -> IConfigManager:
        """
        创建配置管理器实例

        Args:
            manager_type: 管理器类型
            **kwargs: 创建参数

        Returns:
            IConfigManager: 配置管理器实例

        Raises:
            ValueError: 当管理器类型不存在时
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(manager_type, kwargs)

        # 检查缓存
        cached_manager = self.cache.get(cache_key)
        if cached_manager:
            self.stats.increment_cached()
            self._stats['cached_hits'] += 1
            logger.debug(f"Returning cached manager instance: {cache_key}")
            return cached_manager

        # 检查是否为注册的管理器类型
        if self.registry.has_manager(manager_type):
            try:
                manager = self.factory.create_manager(manager_type, **kwargs)
            except Exception as e:
                logger.error(f"Failed to create config manager {manager_type}: {e}")
                self._stats['errors'] += 1
                raise
        # 检查是否为提供者工厂
        elif self.registry.get_provider_factory(manager_type):
            provider_factory = self.registry.get_provider_factory(manager_type)
            try:
                manager = provider_factory(**kwargs)
            except Exception as e:
                logger.error(f"Failed to create provider {manager_type}: {e}")
                self._stats['errors'] += 1
                raise
        else:
            available_types = list(self.registry.get_available_managers().keys())
            self._stats['errors'] += 1
            raise ValueError(
                f"Unknown manager type: {manager_type}. Available types: {available_types}")

        # 放入缓存
        self.cache.put(cache_key, manager)
        self.stats.increment_created()
        self._stats['created_count'] += 1

        logger.info(f"Created new manager instance: {manager_type} -> {cache_key}")
        return manager

    def create_config_manager(self,
                              name: str = "default",
                              config: Optional[Dict[str, Any]] = None,
                              manager_type: str = "unified") -> IConfigManager:
        """
        创建配置管理器（增强版）

        Args:
        name: 管理器名称（用于缓存）
        config: 配置参数
        manager_type: 管理器类型

        Returns:
        IConfigManager: 配置管理器实例
        """
        # 准备配置参数
        if config is None:
            config = self._get_default_config()
        else:
            default_config = self._get_default_config()
            default_config.update(config)
            config = default_config

        # 为缓存添加名称标识
        config['_instance_name'] = name

        return self.create_manager(manager_type, config=config)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "auto_reload": True,
            "validation_enabled": True,
            "encryption_enabled": False,
            "backup_enabled": True,
            "max_backup_files": 5,
            "performance_monitoring": True,
            "error_handling": True
        }

# ==================== 实例管理 ====================

    def get_manager(self, cache_key: str) -> Optional[IConfigManager]:
        """
        获取已缓存的管理器实例

        Args:
        cache_key: 缓存键

        Returns:
        Optional[IConfigManager]: 管理器实例，如果不存在则返回None
        """
        manager = self._manager_instances.get(cache_key)
        if manager is not None:
            self._stats['cached_hits'] += 1
            return manager

    def destroy_manager(self, cache_key: str) -> bool:
        """
        销毁缓存的管理器实例

        Args:
        cache_key: 缓存键

        Returns:
        bool: 是否成功销毁
        """
        if cache_key in self._manager_instances:
            manager = self._manager_instances[cache_key]
            try:
                # 尝试多种清理方法
                cleanup_method = getattr(manager, 'cleanup', None)
                if cleanup_method and callable(cleanup_method):
                    cleanup_method()
                else:
                    close_method = getattr(manager, 'close', None)
                    if close_method and callable(close_method):
                        close_method()
                    else:
                        shutdown_method = getattr(manager, 'shutdown', None)
                        if shutdown_method and callable(shutdown_method):
                            shutdown_method()
            except Exception as e:
                logger.warning(f"Error during manager cleanup: {e}")

            del self._manager_instances[cache_key]
            logger.info(f"Destroyed manager instance: {cache_key}")
            return True
        return False

    def get_all_managers(self) -> Dict[str, IConfigManager]:
        """
        获取所有缓存的管理器实例

        Returns:
        Dict[str, IConfigManager]: 管理器实例字典
        """
        return self._manager_instances.copy()

    def cleanup_all(self):
        """清理所有管理器实例"""
        cache_keys = list(self._manager_instances.keys())
        destroyed_count = 0
        for cache_key in cache_keys:
            if self.destroy_manager(cache_key):
                destroyed_count += 1

        logger.info(f"Cleaned up {destroyed_count}/{len(cache_keys)} manager instances")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取工厂统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        cache_stats = self.cache.get_stats()
        factory_stats = self.stats.get_stats()

        return {
            'registered_managers': len(self.registry.get_available_managers()),
            'cached_instances': len(self.cache.get_all()),
            'registered_providers': len(self.registry._provider_factories),
            'available_types': list(self.registry.get_available_managers().keys()),
            'created_count': self._stats['created_count'],
            'cached_hits': self._stats['cached_hits'],
            'errors': self._stats['errors'],
            **factory_stats,
            **cache_stats
        }

# ==================== 内部方法 ====================

    def _generate_cache_key(self, manager_type: str, kwargs: Dict[str, Any]) -> str:
        """
        生成缓存键

        Args:
        manager_type: 管理器类型
        kwargs: 创建参数

        Returns:
        str: 缓存键
        """
        # 使用关键参数生成缓存键
        key_parts = [manager_type]

        # 添加重要的识别参数
        important_keys = ['config_path', 'env', 'config', 'config_file', 'type', 'name', 'path']
        for key in important_keys:
            if key in kwargs:
                value = kwargs[key]
                # 对复杂对象进行简化
                if isinstance(value, dict):
                    # 检查是否有实例名称
                    if '_instance_name' in value:
                        key_parts.append(f"name={value['_instance_name']}")
                    elif 'name' in value:
                        key_parts.append(f"name={value['name']}")
                    elif 'config_file' in value:
                        key_parts.append(f"config_file={value['config_file']}")
                    else:
                        # 只对其他参数进行哈希
                        other_items = {k: v for k, v in value.items() if k not in [
                            'config_file', 'name', '_instance_name']}
                        if other_items:
                            key_parts.append(f"{key}={hash(str(sorted(other_items.items())))}")
                else:
                    key_parts.append(f"{key}={value}")

        # 如果没有找到识别参数，使用默认名称
        if len(key_parts) == 1 and kwargs:
            key_parts.append("name=default")

        return "|".join(key_parts)




