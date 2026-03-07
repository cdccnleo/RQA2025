
import time

from .interfaces import IErrorComponent
from typing import Any, Dict, List, Optional
"""基础设施层 - 错误处理层 基础实现"""


class BaseErrorComponent(IErrorComponent):
    """错误处理层 基础组件实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础错误组件

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._error_history: List[Dict[str, Any]] = []
        self._max_history = self.config.get('max_history', 1000)

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理错误"""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'handled': False
        }

        # 记录错误历史
        self._error_history.append(error_info)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        return error_info

    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self._error_history.copy()

    def clear_history(self) -> None:
        """清空错误历史"""
        self._error_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_errors = len(self._error_history)
        error_types = {}
        for error in self._error_history:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'max_history': self._max_history,
            'current_history_size': len(self._error_history)
        }
