"""
error_classifier 模块

专门负责错误分类和上下文创建的组件。
"""

import logging
from typing import Dict, Any, Optional, List

from ..core.interfaces import ErrorSeverity, ErrorCategory, ErrorContext

logger = logging.getLogger(__name__)


class ErrorClassifier:
    """错误分类器 - 专门负责错误分类和上下文创建"""

    def __init__(self):
        pass

    def determine_error_type(self, error: Exception) -> str:
        """确定错误类型名称"""
        error_type_name = type(error).__name__

        if error_type_name in ['ConnectionError', 'TimeoutError']:
            return error_type_name
        elif error_type_name == 'OSError':
            # 根据错误消息判断是IOError还是OSError
            error_str = str(error).lower()
            if '文件' in error_str or 'io' in error_str or '磁盘' in error_str:
                return 'IOError'
            else:
                return 'OSError'
        else:
            return error_type_name

    def classify_severity(self, error: Exception) -> ErrorSeverity:
        """分类错误严重程度"""
        error_type = type(error).__name__

        if 'Critical' in error_type or 'SystemExit' in error_type:
            return ErrorSeverity.CRITICAL
        elif 'Database' in error_type or 'Network' in error_type or 'Timeout' in error_type:
            return ErrorSeverity.ERROR
        elif 'Validation' in error_type or 'KeyError' in error_type:
            return ErrorSeverity.WARNING
        elif 'OS' in error_type:
            return ErrorSeverity.ERROR
        else:
            return ErrorSeverity.INFO

    def classify_category(self, error: Exception) -> ErrorCategory:
        """分类错误类别"""
        error_type = type(error).__name__

        if 'Connection' in error_type or 'Network' in error_type or 'Timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'Database' in error_type or 'SQL' in error_type:
            return ErrorCategory.DATABASE
        elif 'Async' in error_type or 'Coroutine' in error_type or 'Future' in error_type or 'OS' in error_type:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN

    def create_error_context(self, error: Exception, context: Optional[Dict[str, Any]], 
                           boundary_results: List[Dict[str, Any]]) -> ErrorContext:
        """创建错误上下文"""
        return ErrorContext(
            error=error,
            severity=self.classify_severity(error),
            category=self.classify_category(error),
            context=context,
            boundary_check=boundary_results
        )
