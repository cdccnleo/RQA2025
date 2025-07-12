import os


def is_production():
    """检查是否在生产环境"""
    return os.getenv('ENV', 'development').lower() == 'production'


def is_development():
    """检查是否在开发环境"""
    return not is_production()


def is_testing():
    """检查是否在测试环境"""
    return 'PYTEST_CURRENT_TEST' in os.environ
