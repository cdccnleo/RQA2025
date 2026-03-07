from enum import Enum
"""
consistencylevel 模块

提供 consistencylevel 相关功能和接口。
"""

"""配置文件存储相关类"""


class ConsistencyLevel(Enum):
    """一致性级别"""
    STRONG = "strong"      # 强一致性
    EVENTUAL = "eventual"  # 最终一致性
    CAUSAL = "causal"      # 因果一致性




