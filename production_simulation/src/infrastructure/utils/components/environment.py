"""
environment 模块

提供 environment 相关功能和接口。
"""

import logging
import os


from typing import Dict, Any
"""
RQA2025 Environment Helper Functions

Environment detection and configuration utilities.
"""

logger = logging.getLogger(__name__)


def is_production() -> bool:
    """
    Check if running in production environment

    Returns:
        True if in production, False otherwise
    """
    env = os.getenv("ENVIRONMENT", "").lower()
    return env in ["production", "prod"]


def is_development() -> bool:
    """
    Check if running in development environment

    Returns:
        True if in development, False otherwise
    """
    env = os.getenv("ENVIRONMENT", "").lower()
    return env in ["", "development", "dev"]


def is_testing() -> bool:
    """
    Check if running in testing environment.

    Returns:
        True if in testing, False otherwise.
    """
    env = os.getenv("ENVIRONMENT", "").lower()
    return env in ["testing", "test"]


def get_environment() -> str:
    """
    Get current environment name

    Returns:
        Environment name (production, development, testing)
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env in ["production", "prod"]:
        return "production"
    elif env in ["testing", "test"]:
        return "testing"
    else:
        return "development"


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get configuration value from environment variables

    Args:
        key: Configuration key

        default: Default value if not found

    Returns:
        Configuration value
    """
    return os.getenv(key, default)


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment

    Returns:
        Database configuration dictionary
    """
    try:
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "name": os.getenv("DB_NAME", "rqa2025"),
            "user": os.getenv("DB_USER", "rqa2025"),
            "password": os.getenv("DB_PASSWORD", ""),
            "ssl_mode": os.getenv("DB_SSL_MODE", "prefer"),
        }
    except ValueError as e:
        logger.error(f"数据库配置解析失败: {e}，使用默认配置")
        return {
            "host": "localhost",
            "port": 5432,
            "name": "rqa2025",
            "user": "rqa2025",
            "password": "",
            "ssl_mode": "prefer",
        }
    except Exception as e:
        logger.error(f"获取数据库配置失败: {e}")
        return {
            "host": "localhost",
            "port": 5432,
            "name": "rqa2025",
            "user": "rqa2025",
            "password": "",
            "ssl_mode": "prefer",
        }


def get_redis_config() -> Dict[str, Any]:
    """
    Get Redis configuration from environment

    Returns:
        Redis configuration dictionary
    """
    try:
        return {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", ""),
            "decode_responses": True,
        }
    except ValueError as e:
        logger.error(f"Redis配置解析失败: {e}，使用默认配置")
        return {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "",
            "decode_responses": True,
        }
    except Exception as e:
        logger.error(f"获取Redis配置失败: {e}")
        return {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "",
            "decode_responses": True,
        }


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration

    Returns:
        Logging configuration dictionary
    """
    return {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        "file": os.getenv("LOG_FILE", "logs / rqa2025.log"),
        "max_file_size": int(os.getenv("LOG_MAX_SIZE", "10485760")),  # 10MB
        "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
    }
