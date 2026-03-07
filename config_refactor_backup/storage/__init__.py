"""
配置存储模块

注意：存储功能已整合到core/config_storage.py中
这里提供向后兼容的导入接口
"""

# 从core模块导入所有存储相关的类和函数
from infrastructure.config.core.config_storage import (
    # 枚举类
    ConfigScope,
    StorageType,
    DistributedStorageType,
    ConsistencyLevel,

    # 数据类
    ConfigItem,
    StorageConfig,

    # 接口
    IConfigStorage,

    # 实现类
    FileConfigStorage,
    MemoryConfigStorage,
    DistributedConfigStorage,
    ConfigStorage,
    UnifiedConfigStorageFactory,

    # 工厂函数
    create_file_storage,
    create_memory_storage,
    create_distributed_storage
)

# 提供向后兼容的别名
FileStorage = FileConfigStorage

__all__ = [
    # 枚举类
    'ConfigScope',
    'StorageType',
    'DistributedStorageType',
    'ConsistencyLevel',

    # 数据类
    'ConfigItem',
    'StorageConfig',

    # 接口
    'IConfigStorage',

    # 实现类
    'FileConfigStorage',
    'MemoryConfigStorage',
    'DistributedConfigStorage',
    'ConfigStorage',
    'UnifiedConfigStorageFactory',

    # 工厂函数
    'create_file_storage',
    'create_memory_storage',
    'create_distributed_storage',

    # 向后兼容别名
    'FileStorage'
]
