"""
智能代码分析器核心模块

提供配置管理和核心数据结构。
"""

from .config import SmartAnalysisConfig
from .analysis_result import AnalysisResult, CodeMetrics, RefactoringSuggestion
from .quality_metrics import QualityMetricsCalculator
from .refactoring_plan import RefactoringPlan, RefactoringAction

__all__ = [
    'SmartAnalysisConfig',
    'AnalysisResult',
    'CodeMetrics',
    'RefactoringSuggestion',
    'QualityMetricsCalculator',
    'RefactoringPlan',
    'RefactoringAction'
]
