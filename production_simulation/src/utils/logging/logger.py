#!/usr/bin/env python3
"""
统一日志记录器工具

为整个RQA2025项目提供统一的日志记录功能
"""

import logging
import os
import logging.handlers
from typing import Optional

# 全局日志记录器实例
_logger: Optional[logging.Logger] = None
_initialized = False


def _setup_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        config: 日志配置字典

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    # 设置日志级别
    level = (config or {}).get('level', 'INFO')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'RQA2025', config: Optional[dict] = None) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称
        config: 日志配置字典

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    global _logger, _initialized

    if not _initialized:
        _logger = _setup_logger(name, config)
        _initialized = True

    return _logger if _logger else _setup_logger(name, config)


def get_component_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    获取组件专用日志记录器

    Args:
        name: 基础名称
        component: 组件名称

    Returns:
        logging.Logger: 组件日志记录器
    """
    if component:
        full_name = f"{name}.{component}"
    else:
        full_name = name

    return get_logger(full_name)

# 向后兼容函数


def configure_logging(level: str = 'INFO'):
    """配置全局日志"""
    global _initialized
    if not _initialized:
        get_logger('RQA2025', {'level': level})
        _initialized = True


def set_log_level(level: str):
    """设置日志级别"""
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def add_file_handler(log_file: str):
    """添加文件处理器"""
    logger = get_logger()
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create file handler: {e}")

# 工厂类


class LoggerFactory:
    """日志记录器工厂"""

    @staticmethod
    def create_logger(name: str, level: str = 'INFO') -> logging.Logger:
        """创建日志记录器"""
        return get_logger(name, {'level': level})

    @staticmethod
    def get_default_logger() -> logging.Logger:
        """获取默认日志记录器"""
        return get_logger()


__all__ = [
    'get_logger',
    'get_component_logger',
    'configure_logging',
    'set_log_level',
    'add_file_handler',
    'LoggerFactory'
]
