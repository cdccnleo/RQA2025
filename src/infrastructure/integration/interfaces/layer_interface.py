"""
层接口

此模块提供了层接口的核心功能。
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# 注意: ICoreLayerComponent已迁移到interfaces.py中的ILayerComponent
# 此处保留向后兼容性，建议使用interfaces.ILayerComponent


class ICoreLayerComponentCompat(ABC):

    """核心层组件接口 (兼容性版本)

    此接口已迁移到interfaces.py中的ILayerComponent
    新代码请使用: from .interfaces import ILayerComponent
    """

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据"""

    @abstractmethod
    def validate(self) -> bool:
        """验证配置"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""


# 向后兼容性别名
ICoreLayerComponent = ICoreLayerComponentCompat


class LayerInterface(ICoreLayerComponent):

    """层接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化层接口

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.initialized = False
        logger.info("初始化层接口")

    def initialize(self) -> bool:
        """初始化组件

        Returns:
            初始化是否成功
        """
        try:
            self.initialized = True
            logger.info("层接口初始化成功")
            return True
        except Exception as e:
            logger.error(f"层接口初始化失败: {e}")
            return False

    def process(self, data: Any) -> Any:
        """处理数据

        Args:
            data: 输入数据

        Returns:
            处理后的数据
        """
        logger.info(f"处理数据: {type(data)}")
        return data

    def validate(self) -> bool:
        """验证配置

        Returns:
            验证结果
        """
        logger.info("验证配置")
        return True

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            状态信息字典
        """
        return {
            "initialized": self.initialized,
            "config": self.config,
            "status": "active"
        }


# 导出主要类
__all__ = ['ICoreLayerComponent', 'LayerInterface']
