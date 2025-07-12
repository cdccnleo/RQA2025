"""
数据层核心接口定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class IDataModel(ABC):
    """
    数据模型接口，定义了所有数据模型必须实现的方法
    """
    @abstractmethod
    def validate(self) -> bool:
        """
        数据有效性验证

        Returns:
            bool: 数据是否有效
        """
        pass

    @abstractmethod
    def get_frequency(self) -> str:
        """
        获取数据频率

        Returns:
            str: 数据频率，如 "1d", "1h", "5min" 等
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取元数据信息

        Returns:
            Dict[str, Any]: 包含数据来源、处理时间、版本等元数据信息
        """
        pass
