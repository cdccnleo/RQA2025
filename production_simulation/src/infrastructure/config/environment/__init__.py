"""
__init__ 模块

提供 __init__ 相关功能和接口。
"""

import sys
import os

from enum import Enum
from typing import Optional
#!/usr/bin/env python3
"""
配置环境相关模块

提供环境检测和管理功能
"""


class EnvironmentType(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


def get_current_environment() -> EnvironmentType:
    """
    获取当前配置环境

    Returns:
        EnvironmentType: 当前环境
    """
    env_value = os.getenv('CONFIG_ENV', os.getenv('ENV', 'development'))
    env = (env_value or 'development').lower()

    env_mapping = {
        'dev': EnvironmentType.DEVELOPMENT,
        'development': EnvironmentType.DEVELOPMENT,
        'test': EnvironmentType.TESTING,
        'testing': EnvironmentType.TESTING,
        'stage': EnvironmentType.STAGING,
        'staging': EnvironmentType.STAGING,
        'prod': EnvironmentType.PRODUCTION,
        'production': EnvironmentType.PRODUCTION,
    }

    return env_mapping.get(env, EnvironmentType.DEVELOPMENT)


def is_production() -> bool:
    """
    检查是否为生产环境

    Returns:
        bool: 是否为生产环境
    """
    return get_current_environment() == EnvironmentType.PRODUCTION


def is_development() -> bool:
    """
    检查是否为开发环境

    Returns:
        bool: 是否为开发环境（非生产环境都被视为开发环境）
    """
    return get_current_environment() != EnvironmentType.PRODUCTION


def is_testing() -> bool:
    """
    检查是否为测试环境

    Returns:
        bool: 是否为测试环境
    """
    # 检查PYTEST_CURRENT_TEST环境变量（优先）
    if os.environ.get('PYTEST_CURRENT_TEST') is not None:
        return True
    
    # 检查当前环境是否为测试环境
    return get_current_environment() == EnvironmentType.TESTING


def is_staging() -> bool:
    """
    检查是否为预发布环境

    Returns:
        bool: 是否为预发布环境
    """
    return get_current_environment() == EnvironmentType.STAGING


def get_environment_config_path() -> Optional[str]:
    """
    获取环境特定的配置文件路径

    Returns:
        Optional[str]: 配置文件路径，如果不存在则返回None
    """
    env = get_current_environment()
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 尝试多种可能的配置文件路径
    possible_paths = [
        os.path.join(base_dir, f"config.{env.name.lower()}.json"),
        os.path.join(base_dir, f"config.{env.name.lower()}.yaml"),
        os.path.join(base_dir, f"config.{env.name.lower()}.yml"),
        os.path.join(base_dir, f"config.{env.name.lower()}.toml"),
        os.path.join(base_dir, "config", f"{env.name.lower()}.json"),
        os.path.join(base_dir, "config", f"{env.name.lower()}.yaml"),
        os.path.join(base_dir, "config", f"{env.name.lower()}.yml"),
        os.path.join(base_dir, "config", f"{env.name.lower()}.toml"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


class ConfigEnvironment:
    """配置环境管理器"""

    def __init__(self):
        self._env_cache = {}
        self._env_vars = {}

    def get_environment(self) -> str:
        """获取当前环境"""
        return os.getenv('ENV', 'development')

    def is_production(self) -> bool:
        """检查是否在生产环境"""
        return self.get_environment().lower() == 'production'

    def is_development(self) -> bool:
        """检查是否在开发环境"""
        return not self.is_production()

    def is_testing(self) -> bool:
        """检查是否在测试环境"""
        return os.environ.get('PYTEST_CURRENT_TEST') is not None

    def get_env_var(self, key: str, default: str = "") -> str:
        """获取环境变量"""
        if key not in self._env_cache:
            self._env_cache[key] = os.getenv(key, default)
        return self._env_cache[key]

    def set_env_var(self, key: str, value: str) -> bool:
        """设置环境变量"""
        try:
            os.environ[key] = value
            self._env_cache[key] = value
            return True
        except Exception:
            return False

    def get_config_for_environment(self, base_config: dict) -> dict:
        """根据环境获取配置"""
        env = self.get_environment()
        config = base_config.copy()

        # 环境特定的配置覆盖
        env_config = config.get(env, {})
        config.update(env_config)

        return config

    def get_environment_info(self) -> dict:
        """获取环境信息"""
        return {
            'environment': self.get_environment(),
            'is_production': self.is_production(),
            'is_development': self.is_development(),
            'is_testing': self.is_testing(),
            'python_version': os.sys.version,
            'platform': os.sys.platform
        }




