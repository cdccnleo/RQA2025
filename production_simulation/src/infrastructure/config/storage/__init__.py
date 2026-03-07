
from .config_storage import *
from .types.iconfigstorage import IConfigStorage
from .types.configscope import ConfigScope
from .types.consistencylevel import ConsistencyLevel
from .types.distributedstoragetype import DistributedStorageType
from .types.storagetype import StorageType
from .types.configitem import ConfigItem
#!/usr/bin/env python3
"""
配置存储模块

提供统一的配置存储功能
"""

__all__ = [
    # 存储接口
    'IConfigStorage',

    # 存储实现
    'FileConfigStorage',
    'MemoryConfigStorage',
    'DistributedConfigStorage',
    'ConfigStorage',

    # 存储枚举
    'ConfigScope',
    'StorageType',
    'DistributedStorageType',
    'ConsistencyLevel',

    # 存储配置
    'StorageConfig',
    'ConfigItem',

    # 工厂函数
    'create_file_storage',
    'create_memory_storage',
    'create_distributed_storage',
    'create_storage',
]




