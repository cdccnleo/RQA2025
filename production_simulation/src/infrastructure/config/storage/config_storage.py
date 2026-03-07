
from typing import Optional, Dict, Any
from .types.storagetype import StorageType
from .types.storageconfig import StorageConfig
from .types.fileconfigstorage import FileConfigStorage
from .types.memoryconfigstorage import MemoryConfigStorage
from .types.distributedconfigstorage import DistributedConfigStorage


class ConfigStorage:
    """配置存储 - 兼容性类"""

    def __init__(self, config=None):
        self.data = {}
        if config:
            self.data.update(config)

        # 兼容性属性
        self.storage = self.data  # 旧API兼容性
        self._configs = self.data  # 内部配置存储

    def get(self, key):
        """获取配置值"""
        return self.data.get(key)

    def set(self, key, value):
        """设置配置值"""
        self.data[key] = value
        return True

    # 兼容性方法 - 旧API支持
    def set_config(self, key, value):
        """设置配置 (兼容性方法)"""
        return self.set(key, value)

    def get_config(self, key):
        """获取配置 (兼容性方法)"""
        return self.get(key)

    def list_configs(self):
        """列出所有配置键 (兼容性方法)"""
        return list(self.data.keys())


# 工厂函数
def create_file_storage(config_path: str = None, path: str = None, **kwargs) -> FileConfigStorage:
    """创建文件配置存储"""
    actual_path = config_path or path
    if not actual_path:
        raise ValueError("必须提供config_path或path参数")
    config = StorageConfig(
        type=StorageType.FILE,
        path=actual_path,
        **kwargs
    )
    return FileConfigStorage(config)


def create_memory_storage(**kwargs) -> MemoryConfigStorage:
    """创建内存配置存储"""
    config = StorageConfig(
        type=StorageType.MEMORY,
        **kwargs
    )
    return MemoryConfigStorage(config)


def create_distributed_storage(storage_type_or_nodes=None, nodes=None, storage_type=None, **kwargs) -> DistributedConfigStorage:
    """创建分布式配置存储

    支持多种调用方式:
    1. create_distributed_storage("consul") - storage_type作为第一个参数
    2. create_distributed_storage(nodes=["host1", "host2"], storage_type="redis")
    3. create_distributed_storage(storage_type="redis") - 关键字参数
    """
    from .types.distributedstoragetype import DistributedStorageType

    # 智能参数解析
    actual_storage_type = None
    actual_nodes = None

    # 检查关键字参数
    if storage_type:
        actual_storage_type = storage_type
    elif 'storage_type' in kwargs:
        actual_storage_type = kwargs.pop('storage_type')

    # 检查位置参数
    if isinstance(storage_type_or_nodes, str) and not actual_storage_type:
        # 第一个参数是storage_type字符串
        actual_storage_type = storage_type_or_nodes
    elif isinstance(storage_type_or_nodes, list):
        # 第一个参数是nodes列表
        actual_nodes = storage_type_or_nodes
    elif storage_type_or_nodes is None:
        # 没有位置参数，使用默认值
        pass

    # 设置默认值
    if not actual_storage_type:
        actual_storage_type = 'redis'
    if not actual_nodes:
        actual_nodes = nodes or ["localhost:6379"]

    # 设置分布式存储类型
    distributed_type = None
    if actual_storage_type:
        storage_type_lower = str(actual_storage_type).lower()
        if storage_type_lower == "redis":
            distributed_type = DistributedStorageType.REDIS
        elif storage_type_lower == "consul":
            distributed_type = DistributedStorageType.CONSUL
        elif storage_type_lower == "etcd":
            distributed_type = DistributedStorageType.ETCD
        elif storage_type_lower == "zookeeper":
            distributed_type = DistributedStorageType.ZOOKEEPER
        else:
            # 如果无法识别，使用Redis作为默认
            distributed_type = DistributedStorageType.REDIS

    # 确保有nodes
    if not actual_nodes:
        actual_nodes = ["localhost:6379"]

    config = StorageConfig(
        type=StorageType.DISTRIBUTED,
        distributed_type=distributed_type,
        nodes=actual_nodes,
        **kwargs
    )
    return DistributedConfigStorage(config)


def create_storage(storage_type, **kwargs):
    """通用存储创建函数"""
    # 支持字符串或枚举类型
    if isinstance(storage_type, str):
        try:
            storage_type = StorageType(storage_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    elif isinstance(storage_type, StorageType):
        pass
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

    # 处理参数映射：dist_type -> storage_type (for backward compatibility)
    if 'dist_type' in kwargs and storage_type == StorageType.DISTRIBUTED:
        kwargs['storage_type'] = kwargs.pop('dist_type')

    if storage_type == StorageType.FILE:
        return create_file_storage(**kwargs)
    elif storage_type == StorageType.MEMORY:
        return create_memory_storage(**kwargs)
    elif storage_type == StorageType.DISTRIBUTED:
        # 为分布式存储移除storage_type参数，因为它会传递给create_distributed_storage
        distributed_kwargs = kwargs.copy()
        distributed_kwargs.pop('storage_type', None)
        return create_distributed_storage(**distributed_kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
