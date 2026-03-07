"""
数据适配器基类定义
"""
# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

from abc import ABC, abstractmethod
from src.infrastructure.logging import get_infrastructure_logger
from typing import Dict, Any

try:
    from ..interfaces.IDataModel import IDataModel
except ImportError:
    # 如果接口不存在，使用占位符
    IDataModel = Any


logger = get_infrastructure_logger('__name__')


class AdapterError(Exception):

    """数据适配器异常基类"""


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

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与数据源的连接

        Returns:
            bool: 连接是否成功

        Raises:
            AdapterError: 连接失败时抛出
        """

    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与数据源的连接
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接数据源

        Returns:
            bool: 是否已连接
        """
