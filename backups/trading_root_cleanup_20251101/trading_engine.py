"""
交易引擎模块（别名模块）
提供向后兼容的导入路径

实际实现在 core/trading_engine.py 中
"""

try:
    from .core.trading_engine import TradingEngine
    from .interfaces.trading_interfaces import OrderType, OrderStatus, OrderSide
except ImportError:
    # 提供基础实现
    class TradingEngine:
        pass
    
    # 基础枚举
    from enum import Enum
    class OrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"
    
    class OrderStatus(Enum):
        PENDING = "pending"
        FILLED = "filled"
    
    class OrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

# OrderDirection在__init__.py中定义，直接从那里导入
from enum import Enum
class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"

__all__ = ['TradingEngine', 'OrderType', 'OrderStatus', 'OrderSide', 'OrderDirection']

