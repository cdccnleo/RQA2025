"""移动端数据模型

此模块包含移动端交易的数据模型定义。
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class WatchlistItem:
    """自选股项目"""

    symbol: str
    name: str
    current_price: float
    change_percent: float
    volume: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    added_at: datetime = None

