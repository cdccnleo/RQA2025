"""
io_error_handler 模块

专门处理IO和系统相关错误的处理器。
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IOErrorHandler:
    """IO错误处理器 - 专门处理IO和系统相关错误"""

    def __init__(self):
        self.max_retries = 3
        self.base_delay = 0.5

    def handle_io_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理IO错误"""
        return {
            'action': 'retry',
            'retry_config': {
                'max_attempts': self.max_retries,
                'base_delay': self.base_delay,
                'strategy': 'fixed'
            },
            'message': f'IO错误，将重试: {error}',
            'handled': False,
            'error_type': 'IOError',
            'context': context
        }

    def handle_os_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理OS错误"""
        error_str = str(error).lower()
        if 'io' in error_str or '文件' in error_str or '磁盘' in error_str or type(error).__name__ == 'IOError':
            error_type = 'IOError'
            message = f'IO错误，将重试: {error}'
        elif type(error).__name__ == 'OSError':
            error_type = 'OSError'
            message = f'系统错误，将重试: {error}'
        else:
            error_type = type(error).__name__
            message = f'错误，将重试: {error}'

        return {
            'action': 'retry',
            'retry_config': {
                'max_attempts': self.max_retries,
                'base_delay': self.base_delay,
                'strategy': 'fixed'
            },
            'message': message,
            'handled': False,
            'error_type': error_type,
            'context': context
        }
