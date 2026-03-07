# RQA2025 Strategy Module
"""
量化策略模块

提供完整的量化策略开发、测试、部署和监控功能。
包括策略开发、回测、实时交易、风险管理等核心功能。
"""

__version__ = "1.0.0"

# 核心服务
from .core.strategy_service import UnifiedStrategyService

# 基础策略类和接口
from .strategies.base_strategy import (
    IStrategy, BaseStrategy, MarketData, StrategySignal,
    StrategyOrder, StrategyPosition, StrategyResult, StrategyType, StrategyStatus
)

# 策略工厂
from .strategies.factory import StrategyFactory

# AI智能组件
from .intelligence import (
    SmartStockFilter, get_smart_stock_filter, MarketState
)

__all__ = [
    # 核心服务
    'UnifiedStrategyService',

    # 基础策略类和接口
    'IStrategy', 'BaseStrategy', 'MarketData', 'StrategySignal',
    'StrategyOrder', 'StrategyPosition', 'StrategyType', 'StrategyStatus',

    # 策略工厂
    'StrategyFactory',

    # AI智能组件
    'SmartStockFilter', 'get_smart_stock_filter', 'MarketState',
]
