"""RQA2025交易模块 - 包含交易策略、执行和风控系统

主要组件:
- execution: 订单执行
- portfolio: 组合管理
- settlement: 结算系统
- strategies: 交易策略(包含优化工具)

使用示例:
    from src.trading import StrategyManager
    from src.trading.execution import ExecutionEngine

    # 初始化策略
    strategy = StrategyManager(model)

    # 执行交易
    executor = ExecutionEngine()
    executor.execute(strategy.generate_signals())

版本历史:
- v1.0 (2024-02-10): 初始版本
- v1.1 (2024-03-20): 添加策略优化功能
"""

from .execution import *
from .portfolio import *
from .settlement import *
from .strategies.optimization import *

__all__ = [
    'execution',
    'portfolio', 
    'settlement',
    'strategies'
]
