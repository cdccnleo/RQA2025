"""
配置接口模块

注意：接口定义已移动到core/unified_interface.py中
这里提供向后兼容的导入
"""

from infrastructure.config.core.unified_interface import IConfigStorage, IConfigManagerComponent, ConfigScope, ConfigItem

# 向后兼容的别名
IConfigManager = IConfigManagerComponent

__all__ = [
    'IConfigStorage',
    'IConfigManager',
    'IConfigManagerComponent',
    'ConfigScope',
    'ConfigItem'
]
