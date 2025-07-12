"""数据适配器抽象基类

定义所有数据适配器应实现的通用接口
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from src.data.core.models import DataModel
import logging

logger = logging.getLogger(__name__)

class BaseDataAdapter(ABC):
    """数据适配器抽象基类"""
    
    @abstractmethod
    def load(self, **params) -> DataModel:
        """
        加载数据
        
        Args:
            params: 适配器特定参数
            
        Returns:
            DataModel: 统一的数据模型
        """
        pass
    
    @abstractmethod 
    def validate(self, data: DataModel) -> bool:
        """
        验证数据完整性
        
        Args:
            data: 要验证的数据模型
            
        Returns:
            bool: 验证结果
        """
        pass
        
    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """
        适配器类型标识
        
        Returns:
            str: 适配器类型名称
        """
        pass
