"""
交易执行层 (Trading Execution Layer)

提供完整的交易执行、风险管理、信号生成、投资组合管理功能
"""

from enum import Enum
from .interfaces.trading_interfaces import (
    OrderType, OrderStatus, OrderSide
)
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# 首先导入常量（必须在导入OrderManager之前，因为OrderManager的默认参数需要用到ORDER_CACHE_SIZE）
try:
    from .core.constants import (
        ORDER_CACHE_SIZE,
        POSITION_CACHE_SIZE,
        CACHE_TTL_SECONDS,
        DEFAULT_ORDER_TIMEOUT,
        MAX_ORDERS_PER_SECOND
    )
except ImportError:
    # 提供默认值
    ORDER_CACHE_SIZE = 10000
    POSITION_CACHE_SIZE = 1000
    CACHE_TTL_SECONDS = 3600
    DEFAULT_ORDER_TIMEOUT = 300
    MAX_ORDERS_PER_SECOND = 100

# 导入实际实现类
try:
    from .core.trading_engine import TradingEngine
    logger.info("Successfully imported TradingEngine from core.trading_engine")
except ImportError as e:
    logger.warning(f"Failed to import TradingEngine from core.trading_engine: {e}")
    # 提供基础实现

    class TradingEngine:
        def __init__(self, risk_config=None):
            self.name = "TradingEngine"
            self.risk_config = risk_config or {}

try:
    from .execution.execution_engine import ExecutionEngine
    logger.info("Successfully imported ExecutionEngine from execution.execution_engine")
except ImportError as e:
    logger.warning(f"Failed to import ExecutionEngine from execution.execution_engine: {e}")

    class ExecutionEngine:
        def __init__(self):
            self.name = "ExecutionEngine"

try:
    from .execution.order_manager import OrderManager
    logger.info("Successfully imported OrderManager from execution.order_manager")
except ImportError as e:
    logger.warning(f"Failed to import OrderManager from execution.order_manager: {e}")

    class OrderManager:
        def __init__(self):
            self.name = "OrderManager"

try:
    from .signal.signal_generator import SignalGenerator
    logger.info("Successfully imported SignalGenerator from signal.signal_generator")
except ImportError as e:
    logger.warning(f"Failed to import SignalGenerator from signal.signal_generator: {e}")

    class SignalGenerator:
        def __init__(self):
            self.name = "SignalGenerator"

try:
    from .interfaces.risk.risk import ChinaRiskController
    logger.info("Successfully imported ChinaRiskController from interfaces.risk.risk")
except ImportError as e:
    logger.warning(f"Failed to import ChinaRiskController from interfaces.risk.risk: {e}")

    class ChinaRiskController:
        def __init__(self):
            self.name = "ChinaRiskController"

# 导入枚举和常量

# 定义OrderDirection枚举（如果不存在）


class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"


# 创建SimpleSignalGenerator别名
SimpleSignalGenerator = SignalGenerator

__all__ = [
    'TradingEngine', 'OrderType', 'OrderDirection', 'OrderStatus', 'OrderSide',
    'OrderManager', 'ExecutionEngine',
    'ChinaRiskController',
    'SignalGenerator', 'SimpleSignalGenerator',
    # 常量
    'ORDER_CACHE_SIZE',
    'POSITION_CACHE_SIZE',
    'CACHE_TTL_SECONDS',
    'DEFAULT_ORDER_TIMEOUT',
    'MAX_ORDERS_PER_SECOND'
]
