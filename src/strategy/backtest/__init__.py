"""
策略回测模块

提供策略回测、优化和分析功能
"""

# 使用延迟导入，避免循环导入问题
def __getattr__(name):
    if name == 'BacktestEngine':
        from .backtest_engine import BacktestEngine
        return BacktestEngine
    elif name == 'ParameterOptimizer':
        from .parameter_optimizer import ParameterOptimizer
        return ParameterOptimizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['BacktestEngine', 'ParameterOptimizer']
