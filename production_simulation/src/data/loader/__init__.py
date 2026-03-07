"""
数据加载器模块对外导出

保留 legacy API，并同步导出核心基类/工厂对象。
"""

from .base_loader import (  # noqa: F401
    LoaderConfig,
    BaseLoader,
    BaseDataLoader,
    DataLoader,
    MockDataLoader,
    DataLoaderRegistry,
    loader_registry,
)

__all__ = [
    "LoaderConfig",
    "BaseLoader",
    "BaseDataLoader",
    "DataLoader",
    "MockDataLoader",
    "DataLoaderRegistry",
    "loader_registry",
]
