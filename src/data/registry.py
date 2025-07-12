"""
数据加载器注册中心
"""
from typing import Dict, Optional, Type
import logging

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class DataRegistry:
    """
    数据加载器注册中心，负责管理所有数据加载器实例
    """
    def __init__(self):
        """初始化注册中心"""
        self._loaders: Dict[str, BaseDataLoader] = {}
        self._loader_classes: Dict[str, Type[BaseDataLoader]] = {}
        logger.info("DataRegistry initialized")

    def register(self, name: str, loader: BaseDataLoader) -> None:
        """
        注册数据加载器实例

        Args:
            name: 加载器名称
            loader: 加载器实例

        Raises:
            ValueError: 如果名称已被注册则抛出
        """
        if name in self._loaders:
            raise ValueError(f"Loader with name '{name}' already registered")

        self._loaders[name] = loader
        logger.info(f"Registered loader instance: {name}")

    def register_class(self, name: str, loader_class: Type[BaseDataLoader]) -> None:
        """
        注册数据加载器类

        Args:
            name: 加载器名称
            loader_class: 加载器类

        Raises:
            ValueError: 如果名称已被注册则抛出
        """
        if name in self._loader_classes:
            raise ValueError(f"Loader class with name '{name}' already registered")

        self._loader_classes[name] = loader_class
        logger.info(f"Registered loader class: {name}")

    def get_loader(self, name: str) -> Optional[BaseDataLoader]:
        """
        获取数据加载器实例

        Args:
            name: 加载器名称

        Returns:
            Optional[BaseDataLoader]: 加载器实例，如果不存在则返回None
        """
        return self._loaders.get(name)

    def create_loader(self, name: str, config: dict) -> BaseDataLoader:
        """
        创建并注册数据加载器实例

        Args:
            name: 加载器名称
            config: 加载器配置

        Returns:
            BaseDataLoader: 创建的加载器实例

        Raises:
            ValueError: 如果加载器类未注册则抛出
        """
        if name not in self._loader_classes:
            raise ValueError(f"Loader class '{name}' not registered")

        loader_class = self._loader_classes[name]
        loader = loader_class(config)
        self._loaders[name] = loader
        logger.info(f"Created and registered loader instance: {name}")
        return loader

    def list_registered_loaders(self) -> list:
        """
        列出所有已注册的加载器名称

        Returns:
            list: 已注册的加载器名称列表
        """
        return list(self._loaders.keys())

    def list_registered_loader_classes(self) -> list:
        """
        列出所有已注册的加载器类名称

        Returns:
            list: 已注册的加载器类名称列表
        """
        return list(self._loader_classes.keys())

    def is_registered(self, name: str) -> bool:
        """检查加载器名称是否已注册
        
        Args:
            name: 加载器名称
            
        Returns:
            bool: 是否已注册
        """
        return name in self._loaders or name in self._loader_classes
