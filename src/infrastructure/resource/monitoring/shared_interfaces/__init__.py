"""
共享接口模块

提供监控系统各组件共享的接口定义
"""

from .interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler

__all__ = ['ILogger', 'IErrorHandler', 'StandardLogger', 'BaseErrorHandler']
