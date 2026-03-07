"""
RQA2025 交易层接口定义

交易层提供完整的交易执行、订单管理和投资组合管理能力，
通过标准化的接口定义确保交易服务的可扩展性和一致性。
"""

from .trading_interfaces import (
    # 枚举
    OrderType,
    OrderStatus,
    OrderSide,
    ExecutionMode,

    # 数据结构
    OrderRequest,
    OrderResponse,
    TradeExecution,
    Position,
    PortfolioSummary,
    TradingMetrics,

    # 核心接口
    IExecutionEngine,
    IOrderManager,
    IPortfolioManager,
    ITradingMonitor,
    IMarketConnector,
    IBrokerAdapter,
    ITradingServiceProvider,
)

__all__ = [
    # 枚举
    'OrderType',
    'OrderStatus',
    'OrderSide',
    'ExecutionMode',

    # 数据结构
    'OrderRequest',
    'OrderResponse',
    'TradeExecution',
    'Position',
    'PortfolioSummary',
    'TradingMetrics',

    # 核心接口
    'IExecutionEngine',
    'IOrderManager',
    'IPortfolioManager',
    'ITradingMonitor',
    'IMarketConnector',
    'IBrokerAdapter',
    'ITradingServiceProvider',
]
