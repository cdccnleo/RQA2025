"""
health_components 统一组件工厂

统一以下模块的ComponentFactory实现:
- health\core\checker_components.py
- health\core\health_components.py
- health\monitors\monitor_components.py
- health\monitors\probe_components.py
- health\services\alert_components.py
- health\services\status_components.py
"""

from src.infrastructure.interfaces import BaseComponentFactory
from typing import Dict, Any
import logging


class HealthComponentsFactory(BaseComponentFactory):
    """health_components 组件工厂"""

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._initialize_supported_types()

    def _initialize_supported_types(self):
        """初始化支持的组件类型"""
        # 这里可以根据具体需求初始化支持的类型

    def create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件

        Args:
            component_type: 组件类型
            config: 组件配置

        Returns:
            创建的组件实例

        Raises:
            ValueError: 当组件类型不支持或配置无效时
        """
        if not self.validate_config(component_type, config):
            raise ValueError("Invalid config for component type: " + component_type)

        # 统一的组件创建逻辑
        # 这里可以根据component_type调用相应的创建方法

        # 缓存已创建的组件
        cache_key = component_type + ":" + str(hash(str(config)))
        if cache_key in self._components:
            return self._components[cache_key]

        # 创建新组件
        component = self._create_component_instance(component_type, config)

        # 缓存组件
        self._components[cache_key] = component

        self._logger.info("Created " + component_type + " component")
        return component

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件实例 - 子类必须实现"""
        raise NotImplementedError("Subclasses must implement _create_component_instance")
