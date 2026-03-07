"""
logger 模块

提供 logger 相关功能和接口。
"""

import logging.handlers

import threading

from pathlib import Path
from typing import Optional, Dict, Any

from ..core.exceptions import (
    handle_logging_exception
)
"""
基础设施层 - 日志系统组件

logger 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理模块 - 基础设施层核心日志服务

提供统一的日志记录功能，支持：
1. 基础日志记录
2. 异步日志处理
3. 日志级别控制
4. 格式化配置
5. 文件和控制台输出
"""

# 全局日志配置
_logging_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': 'logs',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# 线程锁确保线程安全
_logging_lock = threading.Lock()
_initialized = False


@handle_logging_exception("logging_configuration")
def configure_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    全局配置日志系统

    Args:
        config: 日志配置字典，如果为None则使用默认配置

    Raises:
        LogConfigurationError: 当配置失败时抛出
    """
    with _logging_lock:
        # 更新配置
        updated_config = _update_config(config)

        # 创建日志目录
        log_dir = _ensure_log_directory(updated_config)

        # 配置根日志记录器
        root_logger = _configure_root_logger(updated_config)

        # 添加处理器
        _add_handlers(root_logger, log_dir, updated_config)

        # 确保日志文件存在
        _ensure_log_file_exists(log_dir)

        global _initialized
        _initialized = True


def _update_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """更新配置"""
    if config:
        _logging_config.update(config)
    return _logging_config


def _ensure_log_directory(config: Dict[str, Any]) -> Path:
    """确保日志目录存在"""
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _configure_root_logger(config: Dict[str, Any]) -> logging.Logger:
    """配置根日志记录器"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config['level'].upper()))

    # 清除现有处理器
    root_logger.handlers.clear()

    return root_logger


def _add_handlers(root_logger: logging.Logger, log_dir: Path, config: Dict[str, Any]) -> None:
    """添加日志处理器"""
    # 创建格式化器
    formatter = logging.Formatter(
        config['format'],
        datefmt=config['date_format']
    )

    # 添加控制台处理器
    _add_console_handler(root_logger, formatter)

    # 添加文件处理器
    _add_file_handler(root_logger, log_dir, config, formatter)


def _add_console_handler(root_logger: logging.Logger, formatter: logging.Formatter) -> None:
    """添加控制台处理器"""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def _add_file_handler(root_logger: logging.Logger, log_dir: Path,
                      config: Dict[str, Any], formatter: logging.Formatter) -> None:
    """添加文件处理器"""
    log_file_path = log_dir / 'app.log'
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=config['max_bytes'],
        backupCount=config['backup_count'],
        encoding='utf-8'
    )

    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _ensure_log_file_exists(log_dir: Path) -> None:
    """确保日志文件存在"""
    log_file_path = log_dir / 'app.log'
    if not log_file_path.exists():
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write('')


def reset_logging() -> None:
    """重置日志配置，用于测试"""
    global _logging_config, _initialized

    with _logging_lock:
        # 重置为默认配置
        _logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'log_dir': 'logs',
            'max_bytes': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }

        _initialized = False

        # 清除根日志记录器的处理器
        root_logger = logging.getLogger()
        root_logger.handlers.clear()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """获取配置好的日志记录器"

    Args:
        name: 日志器名称，通常使用__name__
        level: 日志级别，如果为None则使用全局配置

    Returns:
        配置好的日志记录器实例
    """
    # 确保日志系统已初始化
    if not _initialized:
        configure_logging()

    logger = logging.getLogger(name)

    # 设置日志级别
    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger


def get_component_logger(component: str, category: str = "business") -> logging.Logger:
    """获取组件专用日志记录器"

    Args:
        component: 组件名称
        category: 日志分类

    Returns:
        组件日志记录器
    """
    logger_name = f"{category}.{component}"
    return get_logger(logger_name)


def set_log_level(name: str, level: str) -> None:
    """设置指定日志记录器的级别"

    Args:
        name: 日志记录器名称
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))


def add_file_handler(logger: Optional[logging.Logger], name: str,

                     filename: str,
                     level: Optional[str] = None,
                     formatter: Optional[logging.Formatter] = None):
    """为指定日志记录器添加文件处理器"

    Args:
        logger: 日志记录器实例，如果为None则根据name获取
        name: 日志记录器名称
        filename: 日志文件路径
        level: 日志级别，如果为None则使用记录器的级别
        formatter: 格式化器，如果为None则使用默认格式化器
    """
    if logger is None:
        logger = logging.getLogger(name)

    # 确保目录存在
    log_path = Path(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建文件处理器
    file_handler = logging.FileHandler(filename, encoding='utf-8')

    if level:
        file_handler.setLevel(getattr(logging, level.upper()))

    if formatter:
        file_handler.setFormatter(formatter)
    else:
        # 使用默认格式化器

        default_formatter = logging.Formatter(
            _logging_config['format'],
            datefmt=_logging_config['date_format']
        )

        file_handler.setFormatter(default_formatter)

    logger.addHandler(file_handler)

    # 确保日志文件被创建
    if not log_path.exists():
        with open(log_path, 'w', encoding='utf - 8') as f:
            f.write('')


class LoggerFactory:

    """日志记录器工厂类"""

    def __init__(self):
        """初始化工厂"""
        self._loggers = {}

    @staticmethod
    def create_logger(name: str,

                      level: str = "INFO",
                      add_file: bool = False,
                      filename: Optional[str] = None):
        """创建日志记录器"

        Args:
            name: 日志记录器名称
            level: 日志级别
            add_file: 是否添加文件处理器
            filename: 日志文件路径，如果为None则使用默认路径

        Returns:
            日志记录器实例
        """
        logger = get_logger(name, level)

        if add_file:
            if filename is None:
                log_dir = Path(_logging_config['log_dir'])
                filename = str(log_dir / f"{name}.log")

            add_file_handler(logger, name, filename, level)

        return logger

    def get_or_create_logger(self, name: str, level: str = "INFO", add_file: bool = False, filename: Optional[str] = None):
        """获取或创建日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别
            add_file: 是否添加文件处理器
            filename: 日志文件路径

        Returns:
            日志记录器实例
        """
        key = (name, level, add_file, filename)

        if key not in self._loggers:
            self._loggers[key] = self.create_logger(name, level, add_file, filename)

        return self._loggers[key]

# 便捷的日志记录函数


def debug(message: str, logger_name: str = "default") -> None:
    """记录调试信息"""
    logger = get_logger(logger_name)
    logger.debug(message)


def info(message: str, logger_name: str = "default") -> None:
    """记录信息"""
    logger = get_logger(logger_name)
    logger.info(message)


def warning(message: str, logger_name: str = "default") -> None:
    """记录警告"""
    logger = get_logger(logger_name)
    logger.warning(message)


def error(message: str, logger_name: str = "default") -> None:
    """记录错误"""
    logger = get_logger(logger_name)
    logger.error(message)


def critical(message: str, logger_name: str = "default") -> None:
    """记录严重错误"""
    logger = get_logger(logger_name)
    logger.critical(message)
