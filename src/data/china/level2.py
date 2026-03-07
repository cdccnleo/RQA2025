"""A股Level2行情处理模块

包含中国A股Level2行情数据的处理逻辑
"""

from typing import Dict, Optional
import pandas as pd


class ChinaLevel2Processor:

    """A股Level2行情处理器"""

    def __init__(self, config: Optional[Dict] = None):
        """初始化处理器

        Args:
            config: 配置字典
        """
        self.config = (config.copy() if config else {})

    def sequence(self, order_book: Dict) -> pd.DataFrame:
        """处理Level2订单簿数据

        Args:
            order_book: 原始订单簿数据

        Returns:
            处理后的订单簿DataFrame
        """
        if not order_book or not isinstance(order_book, dict):
            return pd.DataFrame()

        # 检查必要字段
        required_fields = ['symbol', 'timestamp']
        if not all(field in order_book for field in required_fields):
            return pd.DataFrame()

        # 创建基础DataFrame
        data = []
        if 'bids' in order_book and order_book['bids']:
            for bid in order_book['bids']:
                if isinstance(bid, dict) and 'price' in bid and 'volume' in bid:
                    data.append({
                        'symbol': order_book['symbol'],
                        'timestamp': order_book['timestamp'],
                        'side': 'bid',
                        'price': bid['price'],
                        'volume': bid['volume']
                    })

        if 'asks' in order_book and order_book['asks']:
            for ask in order_book['asks']:
                if isinstance(ask, dict) and 'price' in ask and 'volume' in ask:
                    data.append({
                        'symbol': order_book['symbol'],
                        'timestamp': order_book['timestamp'],
                        'side': 'ask',
                        'price': ask['price'],
                        'volume': ask['volume']
                    })

        return pd.DataFrame(data)

    def process_tick(self, tick_data: Dict) -> pd.DataFrame:
        """处理Level2逐笔数据

        Args:
            tick_data: 原始逐笔数据

        Returns:
            处理后的逐笔DataFrame
        """
        if not tick_data or not isinstance(tick_data, dict):
            return pd.DataFrame()

        # 检查必要字段
        required_fields = ['symbol', 'timestamp']
        if not all(field in tick_data for field in required_fields):
            return pd.DataFrame()

        # 创建DataFrame
        data = [{
            'symbol': tick_data.get('symbol', ''),
            'timestamp': tick_data.get('timestamp', ''),
            'price': tick_data.get('price', 0.0),
            'volume': tick_data.get('volume', 0),
            'side': tick_data.get('side', 'unknown'),
            'order_id': tick_data.get('order_id', '')
        }]

        return pd.DataFrame(data)

    def calculate_market_depth(self, order_book: Dict) -> Dict:
        """计算市场深度指标

        Args:
            order_book: 订单簿数据

        Returns:
            包含深度指标的字典
        """
        if not order_book or not isinstance(order_book, dict):
            return {}

        # 检查必要字段
        if 'bids' not in order_book or 'asks' not in order_book:
            return {}

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # 计算基础指标
        total_bid_volume = sum(bid.get('volume', 0) for bid in bids if isinstance(bid, dict))
        total_ask_volume = sum(ask.get('volume', 0) for ask in asks if isinstance(ask, dict))

        # 计算价格范围
        bid_prices = [bid.get('price', 0)
                      for bid in bids if isinstance(bid, dict) and bid.get('price', 0) > 0]
        ask_prices = [ask.get('price', 0)
                      for ask in asks if isinstance(ask, dict) and ask.get('price', 0) > 0]

        min_bid_price = min(bid_prices) if bid_prices else 0
        max_ask_price = max(ask_prices) if ask_prices else 0

        return {
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'bid_ask_spread': max_ask_price - min_bid_price if min_bid_price > 0 and max_ask_price > 0 else 0,
            'bid_levels': len(bids),
            'ask_levels': len(asks),
            'min_bid_price': min_bid_price,
            'max_ask_price': max_ask_price
        }
