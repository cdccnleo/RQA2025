"""
数据适配器基类定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from .interfaces import IDataModel

logger = logging.getLogger(__name__)


class AdapterError(Exception):
    """数据适配器异常基类"""
    pass


class BaseDataAdapter(ABC):
    """
    数据适配器基类，定义了所有数据适配器必须实现的方法
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据适配器

        Args:
            config: 适配器配置信息
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效

        Raises:
            AdapterError: 配置无效时抛出
        """
        pass

    @abstractmethod
    def transform(self, raw_data: Any) -> IDataModel:
        """
        数据转换方法，将原始数据转换为标准数据模型

        Args:
            raw_data: 原始数据

        Returns:
            IDataModel: 转换后的标准数据模型

        Raises:
            AdapterError: 转换失败时抛出
        """
        pass

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与数据源的连接

        Returns:
            bool: 连接是否成功

        Raises:
            AdapterError: 连接失败时抛出
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与数据源的连接
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接数据源

        Returns:
            bool: 是否已连接
        """
        pass
