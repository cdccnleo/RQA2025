"""
logger 模块

提供 logger 相关功能和接口。
"""

import logging
import logging.handlers
import os


from typing import Optional
"""
RQA2025 Logger Helper Functions

Logging utilities for infrastructure components.
"""


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file is specified
    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(os.getenv("LOG_MAX_SIZE", "10485760")),  # 10MB
                backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5"))
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to create file handler: {e}")

    return logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup global logging configuration

    Args:
        level: Logging level
        log_file: Log file path
    """
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set root logger level
    logging.root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )

            file_handler.setFormatter(formatter)
            logging.root.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Failed to setup file logging: {e}")


def get_unified_logger(name: str) -> logging.Logger:
    """
    Get unified logger (alias for get_logger)

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return get_logger(name)
