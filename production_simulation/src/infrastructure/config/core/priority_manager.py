
from enum import Enum


class ConfigPriority(Enum):
    """配置优先级枚举"""
    DEFAULT = 1
    FILE = 2
    ENVIRONMENT = 3
    REMOTE = 4
    OVERRIDE = 5


class PriorityManager:
    """优先级管理器"""

    def __init__(self):
        self.priorities = {}

    def set_priority(self, item, priority):
        self.priorities[item] = priority

    def get_priority(self, item):
        return self.priorities.get(item, 0)


class ConfigPriorityManager(PriorityManager):
    """配置优先级管理器"""

    def __init__(self):
        super().__init__()
        self.config_priorities = {}

    def set_config_priority(self, config_key, priority):
        """设置配置项优先级"""
        self.config_priorities[config_key] = priority

    def get_config_priority(self, config_key):
        """获取配置项优先级"""
        return self.config_priorities.get(config_key, 0)