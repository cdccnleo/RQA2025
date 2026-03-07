"""
RQA2025 策略层接口定义

策略层提供完整的量化策略开发、回测和优化能力，
通过标准化的接口定义确保策略服务的可扩展性和一致性。
"""

from .strategy_interfaces import (
    # 枚举
    StrategyStatus,
    SignalType,
    StrategyType,

    # 数据结构
    StrategyConfig,
    Signal,
    StrategyPerformance,
    BacktestResult,
    OptimizationConfig,
    OptimizationResult,

    # 核心接口
    IStrategyEngine,
    IStrategyManager,
    IBacktestEngine,
    IStrategyOptimizer,
    ISignalGenerator,
    IStrategyServiceProvider,
)

__all__ = [
    # 枚举
    'StrategyStatus',
    'SignalType',
    'StrategyType',

    # 数据结构
    'StrategyConfig',
    'Signal',
    'StrategyPerformance',
    'BacktestResult',
    'OptimizationConfig',
    'OptimizationResult',

    # 核心接口
    'IStrategyEngine',
    'IStrategyManager',
    'IBacktestEngine',
    'IStrategyOptimizer',
    'ISignalGenerator',
    'IStrategyServiceProvider',
]
