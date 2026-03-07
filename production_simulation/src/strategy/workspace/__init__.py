"""
策略工作台模块

提供量化策略的全生命周期管理，包括：
- 可视化策略构建
- 参数优化工具
- 模拟交易环境
- 绩效分析面板
"""

from .visual_editor import VisualStrategyEditor, StrategyNode, NodeType
from .strategy_generator import AutomaticStrategyGenerator, StrategyConfig, StrategyTemplate, MarketType
from .optimizer import StrategyOptimizer, OptimizationMethod, OptimizationConfig
from .simulator import StrategySimulator, SimulationMode, SimulationConfig
from .analyzer import StrategyAnalyzer
from .store import StrategyStore

__all__ = [
    'VisualStrategyEditor',
    'StrategyNode',
    'NodeType',
    'AutomaticStrategyGenerator',
    'StrategyConfig',
    'StrategyTemplate',
    'MarketType',
    'StrategyOptimizer',
    'OptimizationMethod',
    'OptimizationConfig',
    'StrategySimulator',
    'SimulationMode',
    'SimulationConfig',
    'StrategyAnalyzer',
    'StrategyStore'
]
