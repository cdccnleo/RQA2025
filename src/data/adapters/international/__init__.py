#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国际数据源适配器模块

提供国际市场数据获取能力，支持：
- 美股、港股、日股、英股等股票市场
- 期货、外汇、加密货币等市场
- 实时行情和历史数据
"""

from .base_international_adapter import (
    InternationalDataAdapter,
    MarketDataRequest,
    RealtimeQuote,
    AdapterStatus,
    MarketType,
    DataFrequency,
    RateLimitError,
    DataSourceError
)

from .yahoo_finance_adapter import (
    YahooFinanceAdapter,
    get_yahoo_finance_adapter
)

from .alpha_vantage_adapter import (
    AlphaVantageAdapter,
    get_alpha_vantage_adapter
)

__all__ = [
    # 基类和数据模型
    'InternationalDataAdapter',
    'MarketDataRequest',
    'RealtimeQuote',
    'AdapterStatus',
    'MarketType',
    'DataFrequency',
    'RateLimitError',
    'DataSourceError',
    
    # 适配器实现
    'YahooFinanceAdapter',
    'get_yahoo_finance_adapter',
    'AlphaVantageAdapter',
    'get_alpha_vantage_adapter',
]
