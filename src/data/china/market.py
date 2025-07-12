"""A股市场规则模块

包含中国A股市场的特定交易规则和限制
"""

from dataclasses import dataclass
from datetime import time
from typing import Dict, Any


@dataclass
class TradingHours:
    """交易时间段"""
    pre_open: time
    open: time
    close: time
    after_hours: time  # 科创板盘后交易时间

class ChinaMarketRules:
    """A股市场规则"""

    # 主板交易时间
    MAIN_BOARD_HOURS = TradingHours(
        pre_open=time(9, 15),
        open=time(9, 30),
        close=time(15, 0),
        after_hours=time(15, 30)
    )

    # 科创板交易时间
    STAR_MARKET_HOURS = TradingHours(
        pre_open=time(9, 15),
        open=time(9, 30),
        close=time(15, 0),
        after_hours=time(15, 30)
    )

    @staticmethod
    def get_price_limit(symbol: str) -> float:
        """获取股票的涨跌停限制比例

        Args:
            symbol: 股票代码

        Returns:
            涨跌停限制比例(如0.1表示10%)
        """
        if symbol.startswith(('688', '300')):
            return 0.2  # 科创板和创业板20%
        return 0.1  # 主板10%

    @staticmethod
    def is_t1_restricted(symbol: str) -> bool:
        """检查股票是否受T+1限制

        Args:
            symbol: 股票代码

        Returns:
            bool: 是否受T+1限制
        """
        return True  # A股所有股票都受T+1限制

    @staticmethod
    def get_star_market_rules(symbol: str) -> Dict[str, Any]:
        """获取科创板的特殊规则

        Args:
            symbol: 股票代码

        Returns:
            包含特殊规则的字典
        """
        if not symbol.startswith('688'):
            return {}

        return {
            'after_hours_trading': True,  # 盘后固定价格交易
            'price_limits': 0.2,  # 20%涨跌幅
            'listing_requirements': {
                'profit': False,  # 不要求盈利
                'market_cap': 1_000_000_000  # 10亿市值要求
            }
        }
