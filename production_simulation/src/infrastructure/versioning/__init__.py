"""
版本管理子系统

提供统一的版本管理功能，包括：
- 数据版本管理
- 配置版本管理
- 版本比较和合并
- 版本历史追踪
- 版本策略管理

版本管理架构：
├── core/        # 核心版本类和比较器
├── proxy/       # 版本代理和缓存层
├── manager/     # 版本管理器和策略
├── data/        # 数据版本管理
├── config/      # 配置版本管理
├── api/         # 版本管理API接口
└── tests/       # 测试套件
"""

from .core.version import Version, VersionComparator
from .proxy.proxy import VersionProxy
from .manager.manager import VersionManager
from .manager.policy import VersionPolicy
from .data.data_version_manager import DataVersionManager, VersionInfo

__version__ = "2.0.0"
__all__ = [
    "Version",
    "VersionComparator",
    "VersionProxy",
    "VersionManager",
    "VersionPolicy",
    "DataVersionManager",
    "VersionInfo",
]
