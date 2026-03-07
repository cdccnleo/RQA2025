"""
交易执行器别名模块
提供向后兼容的导入路径

实际实现在 execution/executor.py 中
"""

try:
    from .execution.executor import TradingExecutor, OrderType, OrderStatus
except ImportError:
    # 提供基础实现
    from enum import Enum
    
    class OrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"
    
    class OrderStatus(Enum):
        PENDING = "pending"
        FILLED = "filled"
    
    class TradingExecutor:
        pass

__all__ = ['TradingExecutor', 'OrderType', 'OrderStatus']

