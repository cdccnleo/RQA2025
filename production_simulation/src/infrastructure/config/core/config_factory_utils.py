
from .config_factory_core import UnifiedConfigFactory
from typing import Dict, Any, Optional, Type, List
import logging
from ..interfaces.unified_interface import IConfigManager
#!/usr/bin/env python3
"""
配置工厂工具函数 (拆分自factory.py)

包含全局工厂实例和便捷函数
"""

logger = logging.getLogger(__name__)

# ==================== 全局工厂实例 ====================

# 全局工厂实例
_global_factory: Optional[UnifiedConfigFactory] = None


def get_config_factory() -> UnifiedConfigFactory:
    """
    获取全局配置工厂实例

    Returns:
    UnifiedConfigFactory: 配置工厂实例
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = UnifiedConfigFactory()
    return _global_factory


def reset_global_factory():
    """重置全局工厂实例"""
    global _global_factory
    _global_factory = None

# ==================== 便捷函数 ====================


def create_config_manager(manager_type: str = "unified", **kwargs) -> IConfigManager:
    """
    便捷的配置管理器创建函数

    Args:
    manager_type: 管理器类型
    **kwargs: 创建参数

    Returns:
    IConfigManager: 配置管理器实例
    """
    factory = get_config_factory()
    return factory.create_manager(manager_type, **kwargs)


def get_available_config_types() -> List[str]:
    """
    获取所有可用的配置管理器类型

    Returns:
    List[str]: 类型列表
    """
    factory = get_config_factory()
    return factory.get_stats()['available_types']


def get_factory_stats() -> Dict[str, Any]:
    """
    获取工厂统计信息

    Returns:
    Dict[str, Any]: 统计信息
    """
    factory = get_config_factory()
    return factory.get_stats()

# ==================== 向后兼容性别名 ====================


# 为向后兼容性添加别名
ConfigManagerFactory = UnifiedConfigFactory

# 便捷函数别名


def get_config_manager(manager_type: str = "unified", **kwargs) -> IConfigManager:
    """获取配置管理器实例（便捷函数）"""
    return create_config_manager(manager_type, **kwargs)


def register_config_manager(name: str, manager_class: Type[IConfigManager]) -> None:
    """注册配置管理器（便捷函数）"""
    factory = get_config_factory()
    factory.register_manager(name, manager_class)




