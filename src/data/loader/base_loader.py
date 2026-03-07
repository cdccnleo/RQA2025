"""
数据加载器基础模块（向后兼容适配层）

在此前的重构过程中，真实的加载器抽象已经迁移至
`src/data/core/base_loader.py`。为了兼容 legacy 代码与单测，
此模块需要对外暴露统一的接口定义（LoaderConfig、BaseDataLoader 等）。

该文件因此主要起"桥接层"作用：从 core 模块导入实际实现并重新导出，
确保历史引用路径 `src.data.loader.base_loader` 依旧可用。

同时，为了兼容 data_management 模块的代码，也导出了 DataLoaderConfig 和 LoadResult。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

import pandas as pd

# 从 core 模块导入核心类
from ..core.base_loader import (
    LoaderConfig as CoreLoaderConfig,
    BaseDataLoader as CoreBaseDataLoader,
    DataLoader,
    MockDataLoader,
    DataLoaderRegistry,
    loader_registry,
)

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """
    数据加载器配置类（兼容 data_management 模块）
    
    Attributes:
        source_type: 数据源类型 (postgresql, file, api等)
        connection_string: 连接字符串
        timeout: 加载超时时间（秒）
        retry_count: 重试次数
        cache_enabled: 是否启用缓存
        cache_ttl: 缓存过期时间（秒）
        batch_size: 批量加载大小
    """
    source_type: str = "postgresql"
    connection_string: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5分钟
    batch_size: int = 1000
    
    def __post_init__(self):
        """配置验证"""
        if self.timeout <= 0:
            raise ValueError("timeout 必须大于0")
        if self.retry_count < 0:
            raise ValueError("retry_count 不能为负数")


@dataclass
class LoadResult:
    """
    数据加载结果（兼容 data_management 模块）
    
    Attributes:
        data: 加载的数据
        success: 是否成功
        message: 结果消息
        row_count: 行数
        load_time_ms: 加载耗时（毫秒）
        metadata: 元数据
    """
    data: Optional[pd.DataFrame] = None
    success: bool = False
    message: str = ""
    row_count: int = 0
    load_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# 向后兼容旧命名
BaseLoader = CoreBaseDataLoader

# 确保 LoaderConfig 可以直接导入
LoaderConfig = CoreLoaderConfig
BaseDataLoader = CoreBaseDataLoader

# 导出所有类
__all__ = [
    # 核心类（从 core 模块导入）
    "LoaderConfig",
    "BaseLoader",
    "BaseDataLoader",
    "DataLoader",
    "MockDataLoader",
    "DataLoaderRegistry",
    "loader_registry",
    # 兼容 data_management 模块的类
    "DataLoaderConfig",
    "LoadResult",
]
