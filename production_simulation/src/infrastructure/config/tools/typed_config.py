"""
typed_config 模块

提供 typed_config 相关功能和接口。
"""

"""
类型安全的配置工具模块

提供配置管理的高级工具和扩展功能
核心类型定义统一到 core/typed_config.py 中
"""

from ..core.typed_config import TypedConfigBase

# 注意: 所有核心类已统一到 core/typed_config.py 中
# 工具特定的扩展功能可以在这里添加

# 为兼容性创建别名
TypedConfig = TypedConfigBase




