
from interfaces import IConfigStorage
from typing import Dict, Optional
"""
基础设施层 - 工具组件组件

registry 模块

通用工具组件
提供工具组件相关的功能实现。
"""


class StorageRegistry:
    def __init__(self):
        self._storages: Dict[str, IConfigStorage] = {}

    def register_storage(self, name: str, storage: IConfigStorage):
        self._storages[name] = storage

    def get_storage(self, name: str) -> Optional[IConfigStorage]:
        return self._storages.get(name)


# 全局注册表实例
_storage_registry = StorageRegistry()


def get_storage_registry() -> StorageRegistry:
    return _storage_registry




