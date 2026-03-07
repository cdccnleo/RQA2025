from enum import Enum
"""
storagetype 模块

提供 storagetype 相关功能和接口。
"""

"""配置文件存储相关类"""


class StorageType(Enum):
    """存储类型枚举"""
    FILE = "file"
    MEMORY = "memory"
    DISTRIBUTED = "distributed"
    DATABASE = "database"
    REDIS = "redis"




