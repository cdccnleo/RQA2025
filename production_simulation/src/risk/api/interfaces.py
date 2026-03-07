#!/usr/bin/env python3
"""
# risk层接口定义
定义risk层所有组件的标准接口
"""


from abc import ABC, abstractmethod
from typing import Dict, Any


class IRiskComponent(ABC):

    """risk组件基础接口"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""

    __all__ = ['IRiskComponent']
