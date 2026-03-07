"""
数据加载器基础模块（向后兼容适配层）

在此前的重构过程中，真实的加载器抽象已经迁移至
`src/data/core/base_loader.py`。为了兼容 legacy 代码与单测，
此模块需要对外暴露统一的接口定义（LoaderConfig、BaseDataLoader 等）。

该文件因此主要起“桥接层”作用：从 core 模块导入实际实现并重新导出，
确保历史引用路径 `src.data.loader.base_loader` 依旧可用。
"""

from ..core.base_loader import (  # noqa: F401 - re-export for compatibility
    LoaderConfig,
    BaseDataLoader,
    DataLoader,
    MockDataLoader,
    DataLoaderRegistry,
    loader_registry,
)

BaseLoader = BaseDataLoader  # 向后兼容旧命名

__all__ = [
    "LoaderConfig",
    "BaseLoader",
    "BaseDataLoader",
    "DataLoader",
    "MockDataLoader",
    "DataLoaderRegistry",
    "loader_registry",
]

