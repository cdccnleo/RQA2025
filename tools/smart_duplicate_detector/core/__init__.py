"""
智能重复代码检测工具

提供高级的代码克隆检测、重构建议生成和自动修复功能。
"""

from .code_fragment import CodeFragment, FragmentType
from .similarity_metrics import SimilarityMetrics
from .detection_result import DetectionResult, CloneGroup
from .config import SmartDuplicateConfig

__all__ = [
    'CodeFragment',
    'FragmentType',
    'SimilarityMetrics',
    'DetectionResult',
    'CloneGroup',
    'SmartDuplicateConfig'
]
