import logging


logger = logging.getLogger(__name__)


class ConfigManagerCore:
    """配置管理器核心"""

    def __init__(self):
        self.configs = {}

    def get_config(self, key):
        return self.configs.get(key)

    def set_config(self, key, value):
        self.configs[key] = value
