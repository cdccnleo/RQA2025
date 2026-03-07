"""
基础设施层版本管理模块 - 向后兼容层

此模块提供向后兼容性，保证现有代码能够正常工作。
所有功能已迁移到 versioning 子系统。

推荐新代码使用:
from versioning import Version, VersionComparator, VersionProxy, VersionManager, VersionPolicy
"""

# 导入新的版本管理模块以保持向后兼容性
from .versioning.proxy.proxy import get_default_version_proxy
import warnings

# 发出弃用警告
warnings.warn(
    "src.infrastructure.version 模块已被弃用。请使用 src.infrastructure.versioning 替代。",
    DeprecationWarning,
    stacklevel=2
)

# 重新导出所有类以保持向后兼容性

# 保持原有的全局函数


def get_default_version_proxy():
    """获取默认版本代理实例"""
    return get_default_version_proxy()
