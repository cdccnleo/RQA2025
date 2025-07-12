"""RQA2025回测系统 - 策略验证与评估框架

核心组件:
- engine: 回测引擎核心
- visualizer: 结果可视化
- analyzer: 绩效分析工具
- optimizer: 参数优化器
- data_loader: 回测数据加载

使用示例:
    from src.backtest import BacktestEngine
    from src.backtest.visualizer import Plotter

    # 运行回测
    engine = BacktestEngine()
    results = engine.run(strategy, data)

    # 可视化结果
    plotter = Plotter()
    plotter.plot_performance(results)

主要功能:
- 多频率回测支持(Tick/分钟/日)
- 多维度绩效分析
- 参数优化与敏感性测试
- A股特有规则模拟(涨跌停/T+1)

版本历史:
- v1.0 (2024-05-15): 初始版本
- v1.1 (2024-06-10): 添加参数优化器
"""

from .engine import BacktestEngine
from .visualizer import Plotter
from .analyzer import PerformanceAnalyzer
from .optimizer import ParameterOptimizer

__all__ = [
    'BacktestEngine',
    'Plotter',
    'PerformanceAnalyzer',
    'ParameterOptimizer',
    # 子模块
    'engine',
    'visualizer',
    'analyzer',
    'optimizer',
    'data_loader'
]
