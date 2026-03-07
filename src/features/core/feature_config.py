"""
特征配置模块
提供特征工程相关的配置管理
"""

from enum import Enum

# 从core层导入配置类，避免重复定义
from .config import FeatureConfig, FeatureProcessingConfig, FeatureRegistrationConfig

# 从core层导入FeatureType，避免重复定义
from .config import FeatureType


class ProcessingMode(Enum):

    """处理模式枚举"""
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"


# 导出配置类
__all__ = [
    'FeatureType',
    'ProcessingMode',
    'FeatureConfig',
    'FeatureProcessingConfig',
    'FeatureRegistrationConfig'
]
