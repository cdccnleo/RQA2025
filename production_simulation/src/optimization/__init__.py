"""
优化模块
"""

# 从core导入优化引擎
try:
    from .core.optimization_engine import OptimizationEngine
except ImportError:
    class OptimizationEngine:
        pass

# 从core导入性能优化器
try:
    from .core.performance_optimizer import PerformanceOptimizer
except ImportError:
    class PerformanceOptimizer:
        pass

# 从strategy导入策略优化器
try:
    from .strategy.strategy_optimizer import StrategyOptimizer
except ImportError:
    class StrategyOptimizer:
        pass

# 从portfolio导入投资组合优化器
try:
    from .portfolio.portfolio_optimizer import PortfolioOptimizer
except ImportError:
    class PortfolioOptimizer:
        pass

# 从system导入系统优化器
try:
    from .system.memory_optimizer import MemoryOptimizer
except ImportError:
    class MemoryOptimizer:
        pass

try:
    from .system.cpu_optimizer import CPUOptimizer
except ImportError:
    class CPUOptimizer:
        pass

__all__ = [
    'OptimizationEngine',
    'PerformanceOptimizer',
    'StrategyOptimizer',
    'PortfolioOptimizer',
    'MemoryOptimizer',
    'CPUOptimizer'
]

