
from enum import Enum
"""配置作用域定义"""


class ConfigScope(Enum):
    """配置作用域枚举"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    APPLICATION = "application"




