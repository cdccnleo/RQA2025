from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

@dataclass
class DataModel:
    """统一数据模型"""
    raw_data: Dict
    metadata: Dict
    validation_status: bool = False

class BaseDataAdapter(ABC):
    """所有数据适配器必须实现的基类"""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """必须明确定义类型标识"""
        pass

    @abstractmethod
    def load_data(self, config: Dict) -> DataModel:
        """加载数据并返回统一数据模型"""
        pass

    @abstractmethod
    def validate(self, data: DataModel) -> bool:
        """验证数据有效性"""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(type={self.adapter_type})"
