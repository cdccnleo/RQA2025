"""特殊股票处理模块

包含中国A股市场中特殊股票的处理逻辑，如ST/*ST股票、科创板股票等
"""

from typing import Dict, List, Optional
import pandas as pd

class SpecialStockHandler:
    """特殊股票处理器"""

    def __init__(self, config: Optional[Dict] = None):
        """初始化处理器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    def is_special_stock(self, symbol: str) -> bool:
        """检查股票是否为特殊股票(ST/*ST等)

        Args:
            symbol: 股票代码

        Returns:
            bool: 是否为特殊股票
        """
        return symbol.startswith(('ST', '*ST'))

    def get_special_rules(self, symbol: str) -> Dict:
        """获取股票的特殊交易规则

        Args:
            symbol: 股票代码

        Returns:
            包含特殊规则的字典
        """
        rules = {}
        if symbol.startswith('ST'):
            rules['price_limit'] = 0.05  # ST股票5%涨跌幅限制
            rules['disclosure'] = 'enhanced'  # 加强信息披露
        elif symbol.startswith('688'):
            rules['price_limit'] = 0.2  # 科创板20%涨跌幅
            rules['after_hours'] = True  # 盘后固定价格交易
        return rules

    def filter_special_stocks(self, stocks: List[str]) -> Dict[str, List[str]]:
        """筛选特殊股票

        Args:
            stocks: 股票代码列表

        Returns:
            按类型分类的特殊股票字典
        """
        result = {
            'ST': [],
            'STAR': [],
            'other': []
        }

        for symbol in stocks:
            if symbol.startswith('ST'):
                result['ST'].append(symbol)
            elif symbol.startswith('688'):
                result['STAR'].append(symbol)
            else:
                result['other'].append(symbol)

        return result
