
"""配置文件存储相关类"""

from enum import Enum


class DistributedStorageType(Enum):
    """分布式存储类型"""
    REDIS = "redis"
    ETCD = "etcd"
    CONSUL = "consul"
    ZOOKEEPER = "zookeeper"




