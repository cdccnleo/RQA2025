import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征插件基类

from src.infrastructure.logging.core.unified_logger import get_unified_logger
定义特征插件的基类和元数据结构。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class PluginType(Enum):

    """插件类型枚举"""
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    SELECTOR = "selector"
    STANDARDIZER = "standardizer"
    ENGINEER = "engineer"
    CUSTOM = "custom"


class PluginStatus(Enum):

    """插件状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class PluginMetadata:

    """插件元数据"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    min_api_version: str = "1.0.0"
    max_api_version: str = "2.0.0"
    status: PluginStatus = PluginStatus.INACTIVE
    load_time: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "config_schema": self.config_schema,
            "min_api_version": self.min_api_version,
            "max_api_version": self.max_api_version,
            "status": self.status.value,
            "load_time": self.load_time,
            "error_message": self.error_message
        }


class BaseFeaturePlugin(ABC):

    """特征插件基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化插件

        Args:
            config: 插件配置
        """
        self.config = config or {}
        # 为每个插件实例提供独立的日志记录器，防止未定义属性导致初始化失败
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metadata = self._get_metadata()
        self._validate_config()

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """获取插件元数据"""

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        处理数据

        Args:
            data: 输入数据
            **kwargs: 额外参数

        Returns:
            处理后的数据
        """

    def initialize(self) -> bool:
        """
        初始化插件

        Returns:
            初始化是否成功
        """
        try:
            self.logger.info(f"初始化插件: {self.metadata.name}")
            self._initialize_plugin()
            self.metadata.status = PluginStatus.ACTIVE
            return True
        except Exception as e:
            self.logger.error(f"插件初始化失败: {e}")
            self.metadata.status = PluginStatus.ERROR
            self.metadata.error_message = str(e)
            return False

    def cleanup(self) -> bool:
        """
        清理插件资源

        Returns:
            清理是否成功
        """
        try:
            self.logger.info(f"清理插件: {self.metadata.name}")
            self._cleanup_plugin()
            self.metadata.status = PluginStatus.INACTIVE
            return True
        except Exception as e:
            self.logger.error(f"插件清理失败: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        获取插件信息

        Returns:
            插件信息字典
        """
        return {
            "metadata": self.metadata.to_dict(),
            "config": self.config,
            "capabilities": self._get_capabilities()
        }

    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            数据是否有效
        """
        try:
            return self._validate_input(data)
        except Exception as e:
            self.logger.error(f"输入验证失败: {e}")
            return False

    def _validate_config(self):
        """验证配置"""
        if self.metadata.config_schema:
            # TODO: 实现配置验证逻辑
            pass

    def _initialize_plugin(self):
        """初始化插件（子类可重写）"""

    def _cleanup_plugin(self):
        """清理插件（子类可重写）"""

    def _get_capabilities(self) -> Dict[str, Any]:
        """获取插件能力（子类可重写）"""
        return {}

    def _validate_input(self, data: Any) -> bool:
        """验证输入数据（子类可重写）"""
        return True

    def __str__(self):

        return f"{self.metadata.name} v{self.metadata.version}"

    def __repr__(self):

        return f"<{self.__class__.__name__}: {self}>"
