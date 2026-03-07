
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


class ConfigEnvironment:
    """配置环境管理器

    提供环境检测、环境配置管理、环境变量缓存等功能。
    """

    def __init__(self):
        """初始化环境管理器"""
        self._env_cache = {}
        self._env_vars = {}

    def get_environment(self):
        """获取当前环境"""
        env_value = os.getenv('ENV', 'development')
        return env_value

    def is_production(self):
        """检查是否在生产环境"""
        env = self.get_environment().lower()
        return env == 'production'

    def is_development(self):
        """检查是否在开发环境"""
        env = self.get_environment().lower()
        return env == 'development'

    def is_testing(self):
        """检查是否在测试环境"""
        return os.environ.get('PYTEST_CURRENT_TEST') is not None

    def get_environment_info(self):
        """获取环境详细信息"""
        env = self.get_environment()
        return {
            'environment': env,
            'is_production': self.is_production(),
            'is_development': self.is_development(),
            'is_testing': self.is_testing()
        }

    def set_env_var(self, name, value):
        """设置环境变量"""
        try:
            os.environ[name] = str(value)
            self._env_vars[name] = str(value)
            return True
        except Exception:
            return False

    def get_env_var(self, name, default=None):
        """获取环境变量"""
        return os.environ.get(name, default)

    def get_config_for_environment(self, base_config):
        """获取环境特定的配置"""
        import copy
        return copy.copy(base_config)


