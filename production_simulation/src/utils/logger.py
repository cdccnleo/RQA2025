"""
日志工具模块
"""

import logging

def get_logger(name=None):
    """获取logger"""
    return logging.getLogger(name or __name__)

class Logger:
    """Logger类"""
    def __init__(self, name=None):
        self.logger = get_logger(name)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

__all__ = ['get_logger', 'Logger']

