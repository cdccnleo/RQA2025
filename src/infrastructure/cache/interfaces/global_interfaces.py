
# from .interfaces import ICacheStrategy, ICacheManager, IConfigStorage
# 其他接口暂时注释，避免导入错误
# 缓存相关接口

# from .data_structures import CacheEvictionStrategy  # 暂时注释
from enum import Enum
from typing import Protocol, Dict, Any
"""
全局接口定义文件

统一导入所有测试中需要的接口，避免NameError
"""

# 导入处理已在上方完成


class ICacheStrategy(Protocol):
    """缓存策略接口"""

    def should_evict(self, key: str, value: Any, cache_size: int) -> bool: ...

    def on_access(self, key: str, value: Any) -> None: ...

    def on_evict(self, key: str, value: Any) -> None: ...

    def on_get(self, cache: Dict[str, Any], key: str, entry: Any, config: Any) -> None: ...


class PartitionStrategy(Enum):
    DATE = "date"
    HASH = "hash"
    CUSTOM = "custom"
    RANGE = "range"


class RepairStrategy(Enum):
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    REMOVE_OUTLIERS = "remove_outliers"
    DROP = "drop"
    LOG_TRANSFORM = "log_transform"
    INTERPOLATE = "interpolate"


# 导出所有接口
__all__ = [
    'ICacheStrategy',
    'PartitionStrategy',
    'RepairStrategy'
]
