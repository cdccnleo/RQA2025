"""日志配置工具

提供项目范围内的日志配置功能。

函数:
- setup_logging: 配置项目日志系统
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir='logs', log_file='app.log', level=logging.INFO):
    """配置项目日志系统

    参数:
        log_dir: 日志目录路径
        log_file: 日志文件名
        level: 日志级别

    返回:
        None
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 避免重复配置
    if root_logger.handlers:
        return

    # 创建文件处理器
    log_path = Path(log_dir) / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

__all__ = ['setup_logging']
