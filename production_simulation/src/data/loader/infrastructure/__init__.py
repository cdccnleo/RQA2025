"""数据加载器基础设施模块"""

# 配置管理


class Config:

    """简单的配置管理类"""

    def __init__(self):

        self._config = {}

    def get(self, key: str, default=None):
        """获取配置值"""
        return self._config.get(key, default)

    def set(self, key: str, value):
        """设置配置值"""
        self._config[key] = value


# 全局配置实例
config = Config()
