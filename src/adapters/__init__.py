
"""
RQA2025适配器模块

提供各种交易平台和数据源的适配器实现
"""

from .base.base_adapter import SecureConfigManager, BaseAdapter, DataAdapter, MockAdapter
from .miniqmt import MiniQMTAdapter, MiniQMTTradeAdapter
from .market.market_adapters import (
    MarketAdapter, AStockAdapter, HStockAdapter, USStockAdapter,
    FuturesAdapter, MarketType, AssetClass
)
from .professional.professional_data_adapters import ProfessionalDataAdapter
from .qmt.qmt_adapter import QMTAdapter
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# 导出所有适配器类和组件
__all__ = [
    'SecureConfigManager',
    'BaseAdapter',
    'DataAdapter',
    'MockAdapter',
    'MiniQMTAdapter',
    'MiniQMTTradeAdapter',
    'MarketAdapter',
    'AStockAdapter',
    'HStockAdapter',
    'USStockAdapter',
    'FuturesAdapter',
    'MarketType',
    'AssetClass',
    'ProfessionalDataAdapter',
    'QMTAdapter'
]
