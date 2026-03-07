"""
connection_error_handler 模块

专门处理连接相关错误的处理器。
"""

import logging
from typing import Dict, Any, Optional, Type
from ..core.interfaces import IErrorHandler, ErrorSeverity, ErrorCategory, ErrorContext

logger = logging.getLogger(__name__)


class ConnectionErrorHandler:
    """连接错误处理器 - 专门处理网络和连接相关错误"""

    def __init__(self):
        self.max_retries = 5
        self.base_delay = 1.0

    def handle_connection_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理连接错误"""
        return {
            'action': 'retry',
            'retry_config': {
                'max_attempts': self.max_retries,
                'base_delay': self.base_delay,
                'strategy': 'exponential'
            },
            'message': f'连接错误，将重试: {error}',
            'handled': False,
            'error_type': 'ConnectionError',
            'context': context
        }

    def handle_timeout_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理超时错误"""
        return {
            'action': 'retry',
            'retry_config': {
                'max_attempts': 3,
                'base_delay': 2.0,
                'strategy': 'exponential'
            },
            'message': f'超时错误，将重试: {error}',
            'handled': False,
            'error_type': 'TimeoutError',
            'context': context
        }
