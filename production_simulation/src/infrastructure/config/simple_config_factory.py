
from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
from typing import Dict, Any
#!/usr/bin/env python3
"""
简单配置工厂
提供简单的配置管理器创建功能
"""


class SimpleConfigFactory:
    """简单配置工厂"""

    def __init__(self):
        self._instances: Dict[str, UnifiedConfigManager] = {}

    def create_manager(self, name: str = "default", config: Dict[str, Any] = None) -> UnifiedConfigManager:
        """创建配置管理器"""
        if name in self._instances:
            return self._instances[name]

        manager = UnifiedConfigManager(config or {})
        self._instances[name] = manager
        return manager

    def get_manager(self, name: str = "default") -> UnifiedConfigManager:
        """获取配置管理器"""
        return self._instances.get(name)

    def remove_manager(self, name: str) -> bool:
        """移除配置管理器"""
        if name in self._instances:
            del self._instances[name]
            return True
        return False

    def list_managers(self) -> list:
        """列出所有管理器"""
        return list(self._instances.keys())

    def clear_all(self):
        """清空所有管理器"""
        self._instances.clear()


# 全局简单工厂实例
_simple_factory = None


def get_simple_factory() -> SimpleConfigFactory:
    """获取全局简单工厂实例"""
    global _simple_factory
    if _simple_factory is None:
        _simple_factory = SimpleConfigFactory()
    return _simple_factory


def create_simple_manager(name: str = "default", config: Dict[str, Any] = None) -> UnifiedConfigManager:
    """创建简单配置管理器（便捷函数）"""
    factory = get_simple_factory()
    return factory.create_manager(name, config)


def get_simple_manager(name: str = "default") -> UnifiedConfigManager:
    """获取简单配置管理器（便捷函数）"""
    factory = get_simple_factory()
    return factory.get_manager(name)




