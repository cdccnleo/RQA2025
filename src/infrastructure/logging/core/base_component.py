
from abc import ABC
from typing import Any, Dict, Optional
"""
基础设施层 - 基础组件类

提供通用的组件初始化和配置管理功能，消除重复代码。
"""


class BaseComponent(ABC):
    """
    基础组件类

    提供标准的组件初始化、配置管理和生命周期管理。
    所有组件都应该继承此类以获得一致的行为。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础组件

        Args:
            config: 组件配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._component_name = self.__class__.__name__

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置

        Args:
            new_config: 新配置字典
        """
        self.config.update(new_config)

    def is_initialized(self) -> bool:
        """检查组件是否已初始化"""
        return self._initialized

    def get_component_name(self) -> str:
        """获取组件名称"""
        return self._component_name

    def validate_config(self) -> bool:
        """
        验证配置

        Returns:
            配置是否有效
        """
        # 子类可以重写此方法进行配置验证
        return True
