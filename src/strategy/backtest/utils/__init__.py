import time
"""回测工具模块 - 从基础设施层导入统一工具服务"""

# 从基础设施层导入统一的工具服务
from src.infrastructure.utils.helpers.logger import (
    get_logger,
    get_component_logger,
    LoggerFactory,
    configure_logging,
    set_log_level,
    add_file_handler
)

from src.infrastructure.utils.helpers.date_utils import (
    convert_timezone,
    get_trading_days,
    is_market_open,
    TradingDateUtils
)

# 导入回测专用工具函数
from .backtest_utils import (
    BacktestUtils,
    StrategyValidationResult
)

import logging
from typing import Optional

# 回测专用日志记录器


def get_backtest_logger(name: str = "backtest", level: Optional[str] = None) -> logging.Logger:
    """获取回测专用日志记录器"""
    return get_component_logger(name, "backtest")


__all__ = [
    # 日志相关
    'get_logger',
    'get_backtest_logger',
    'get_component_logger',
    'LoggerFactory',
    'configure_logging',
    'set_log_level',
    'add_file_handler',
    # 日期相关
    'convert_timezone',
    'parse_date_range',
    'get_trading_days',
    'is_market_open',
    'DateUtils',
    # 回测专用工具
    'BacktestUtils',
    'StrategyValidationResult'
]
