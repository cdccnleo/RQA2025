"""
Process Components模块

业务流程组件定义
"""

from typing import Dict, Any
from abc import ABC, abstractmethod


class ProcessComponent(ABC):
    """流程组件基类"""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        self._initialized = True
        return True
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行组件逻辑"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'name': self.name,
            'initialized': self._initialized
        }


__all__ = ['ProcessComponent']

