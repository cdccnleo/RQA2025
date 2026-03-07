"""
数据加载器模块对外导出

保留 legacy API，并同步导出核心基类/工厂对象。
同时兼容 data_management 模块的代码。
"""

from .base_loader import (  # noqa: F401
    # 核心类
    LoaderConfig,
    BaseLoader,
    BaseDataLoader,
    DataLoader,
    MockDataLoader,
    DataLoaderRegistry,
    loader_registry,
    # 兼容 data_management 模块的类
    DataLoaderConfig,
    LoadResult,
)

# PostgreSQL 数据加载器
try:
    from .postgresql_loader import (
        PostgreSQLDataLoader,
        get_postgresql_loader,
        close_postgresql_loader,
    )
    POSTGRESQL_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"PostgreSQL 数据加载器导入失败: {e}")
    POSTGRESQL_LOADER_AVAILABLE = False
    PostgreSQLDataLoader = None
    get_postgresql_loader = None
    close_postgresql_loader = None

__all__ = [
    # 核心类
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
    # PostgreSQL 数据加载器
    "PostgreSQLDataLoader",
    "get_postgresql_loader",
    "close_postgresql_loader",
    "POSTGRESQL_LOADER_AVAILABLE",
]
