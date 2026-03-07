
# 全局工厂函数别名

from .config_factory_compat import ConfigFactory
from .config_factory_core import UnifiedConfigFactory
from .config_factory_utils import get_config_factory, create_config_manager
# 工厂模块别名 - 为向后兼容性


def reset_global_factory():
    """重置全局工厂"""


def get_available_config_types():
    """获取可用的配置类型"""
    return ["unified"]


def get_factory_stats():
    """获取工厂统计信息"""
    return {}


def get_config_manager(name: str):
    """获取配置管理器"""
    factory = get_config_factory()
    return factory.get_manager(name)


def register_config_manager(name: str, manager_class):
    """注册配置管理器"""
    factory = get_config_factory()
    factory.register_manager(name, manager_class)


__all__ = [
    'UnifiedConfigFactory',
    'ConfigFactory',
    'get_config_factory',
    'create_config_manager',
    'reset_global_factory',
    'get_available_config_types',
    'get_factory_stats',
    'get_config_manager',
    'register_config_manager'
]




