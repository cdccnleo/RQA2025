"""
版本管理核心模块

提供基础的版本类、比较器和接口定义。
"""

from .version import Version, VersionComparator
from .interfaces import (
    VersionProvider,
    VersionComparatorInterface,
    VersionStorage,
    VersionManagerInterface,
    VersionPolicyInterface,
    DataVersionManagerInterface,
    ConfigVersionManagerInterface
)

__all__ = [
    "Version",
    "VersionComparator",
    "VersionProvider",
    "VersionComparatorInterface",
    "VersionStorage",
    "VersionManagerInterface",
    "VersionPolicyInterface",
    "DataVersionManagerInterface",
    "ConfigVersionManagerInterface",
]
