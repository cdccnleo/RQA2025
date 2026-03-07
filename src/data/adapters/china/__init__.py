"""
RQA2025 中国市场数据适配器 - 通用适配器接口层

职责定位：
1. 提供符合统一适配器规范的通用适配器接口
2. 定义中国市场适配器基类和标准接口
3. 为 china/ 目录的业务实现层提供基础接口

架构层次：
- adapters/china/  → 通用适配器接口层（本目录）
- china/           → 中国市场业务逻辑实现层
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseChinaAdapter(ABC):
    """
    中国市场数据适配器基类
    
    这是所有中国市场适配器的统一基类，定义了标准的适配器接口。
    具体的业务实现应该在 china/ 目录中继承此类。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器

        Args:
            config: 适配器配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def connect(self) -> bool:
        """
        连接数据源

        Returns:
            bool: 连接是否成功
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            bool: 断开是否成功
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取数据

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 数据字典
        """
        raise NotImplementedError

    def is_connected(self) -> bool:
        """
        检查连接状态

        Returns:
            bool: 是否已连接
        """
        # 默认实现，子类可以重写
        return hasattr(self, '_is_connected') and self._is_connected


# 导出接口
__all__ = [
    'BaseChinaAdapter',
]
