"""
业务流程配置管理子模块

包含：
- config.py: 流程配置管理器
- enums.py: 业务流程相关枚举定义
"""

# 只导出枚举，避免循环导入
from .enums import BusinessProcessState, EventType

__all__ = [
    'BusinessProcessState',
    'EventType'
]

