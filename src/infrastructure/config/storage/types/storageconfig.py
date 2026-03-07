
"""配置文件存储相关类"""

from dataclasses import dataclass
from typing import Optional
from .consistencylevel import ConsistencyLevel
from .distributedstoragetype import DistributedStorageType
from .storagetype import StorageType


@dataclass
class StorageConfig:
    """存储配置"""
    type: StorageType = StorageType.FILE
    path: Optional[str] = None
    distributed_type: Optional[DistributedStorageType] = None
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    backup_enabled: bool = True
    max_backups: int = 10
    nodes: Optional[list] = None




