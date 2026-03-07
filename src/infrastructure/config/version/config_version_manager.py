
from .components.configdiff import ConfigDiff
from .components.configversion import ConfigVersion
from .components.configversionmanager import ConfigVersionManager
"""配置版本管理模块

已重构为模块化结构，保持向后兼容性。
"""

# 导入所有版本管理组件
__all__ = [
    "ConfigVersion",
    "ConfigDiff",
    "ConfigVersionManager",
]

# 向后兼容性别名
ConfigVersionAlias = ConfigVersion




