"""
优化器配置模块

提供智能业务流程优化器的所有配置类
采用参数对象模式，替代长参数列表
"""

from .optimizer_configs import (
    OptimizationStrategy,
    OptimizerConfig,
    AnalysisConfig,
    DecisionConfig,
    ExecutionConfig,
    RecommendationConfig,
    MonitoringConfig,
    create_optimizer_config
)

# DecisionStrategy 从 models 导入
from ..models import DecisionStrategy

__all__ = [
    # 枚举
    'OptimizationStrategy',
    'DecisionStrategy',

    # 配置类
    'OptimizerConfig',
    'AnalysisConfig',
    'DecisionConfig',
    'ExecutionConfig',
    'RecommendationConfig',
    'MonitoringConfig',

    # 便捷函数
    'create_optimizer_config'
]
