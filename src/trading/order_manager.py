"""
订单管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 execution/order_manager.py 中
"""

from enum import Enum


class OrderDirection(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


try:
    from .execution.order_manager import OrderManager
except ImportError:
    # 提供基础实现
    class OrderManager:
        pass

__all__ = ['OrderManager', 'OrderDirection', 'OrderType']

