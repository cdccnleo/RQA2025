# 策略决策层模块
"""
策略决策层模块
Strategy Decision Layer Module

基于业务流程驱动架构，提供策略的统一创建、管理和优化功能。
"""

from .base_strategy import BaseStrategy
from .factory import StrategyFactory, get_strategy_factory
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy

# 兼容性导入
from strategy.interfaces.strategy_interfaces import StrategyConfig
from .multi_strategy_integration import (
    MultiStrategyIntegration,
    IntegrationConfig,
    IntegrationResult,
    StrategyInfo,
    PerformanceMonitor,
    WeightOptimizer,
    RiskManager
)
from .performance_evaluation import (
    StrategyPerformanceEvaluator,
    PerformanceMetrics,
    EvaluationConfig,
    EvaluationResult,
    ReturnCalculator,
    RiskAnalyzer,
    BenchmarkComparator,
    PerformanceAttributor
)
from .optimization.advanced_optimizer import (
    AdvancedStrategyOptimizer,
    OptimizationConfig,
    OptimizationResult,
    ParameterSpace,
    BayesianOptimizer,
    GeneticOptimizer,
    ParticleSwarmOptimizer,
    GridSearchOptimizer
)

__all__ = [
    # 基础策略
    'BaseStrategy',
    'StrategyConfig',
    'StrategyFactory',
    'get_strategy_factory',
    'MomentumStrategy',
    'MeanReversionStrategy',

    # 多策略集成
    'MultiStrategyIntegration',
    'IntegrationConfig',
    'IntegrationResult',
    'StrategyInfo',
    'PerformanceMonitor',
    'WeightOptimizer',
    'RiskManager',

    # 性能评估
    'StrategyPerformanceEvaluator',
    'PerformanceMetrics',
    'EvaluationConfig',
    'EvaluationResult',
    'ReturnCalculator',
    'RiskAnalyzer',
    'BenchmarkComparator',
    'PerformanceAttributor',

    # 参数优化
    'AdvancedStrategyOptimizer',
    'OptimizationConfig',
    'OptimizationResult',
    'ParameterSpace',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'ParticleSwarmOptimizer',
    'GridSearchOptimizer'
]
