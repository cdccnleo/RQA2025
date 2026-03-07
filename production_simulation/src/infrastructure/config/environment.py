
import os
"""
基础设施层 - 配置管理组件

environment 模块

配置管理相关的文件
提供配置管理相关的功能实现。
"""


def is_production():
    """检查是否在生产环境"""
    env_value = os.getenv('ENV', 'development')
    return (env_value or '').lower() == 'production'


def is_development():
    """检查是否在开发环境"""
    env_value = os.getenv('ENV', 'development')
    return (env_value or '').lower() != 'production'


def is_testing():
    """检查是否在测试环境"""
    return os.environ.get('PYTEST_CURRENT_TEST') is not None






