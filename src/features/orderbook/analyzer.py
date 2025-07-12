#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
订单簿分析模块
负责Level2订单簿数据的实时分析
"""

from typing import Dict, List, Tuple
import numpy as np
from ..manager import FeatureManager

class OrderbookAnalyzer:
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.orderbook_cache = {}  # {symbol: {'bids': [], 'asks': []}}

    def update_orderbook(self, symbol: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        """更新订单簿数据

        Args:
            symbol: 标的代码
            bids: 买盘 [(价格, 数量), ...]
            asks: 卖盘 [(价格, 数量), ...]
        """
        self.orderbook_cache[symbol] = {
            'bids': sorted(bids, key=lambda x: -x[0]),  # 买盘按价格降序
            'asks': sorted(asks, key=lambda x: x[0])    # 卖盘按价格升序
        }

    def calculate_metrics(self, symbol: str) -> Dict[str, float]:
        """计算订单簿指标

        Args:
            symbol: 标的代码

        Returns:
            订单簿指标字典
        """
        if symbol not in self.orderbook_cache:
            return {}

        orderbook = self.orderbook_cache[symbol]
        metrics = {}

        # 计算买卖价差
        if orderbook['asks'] and orderbook['bids']:
            metrics['spread'] = orderbook['asks'][0][0] - orderbook['bids'][0][0]

        # 计算订单簿不平衡度
        metrics['imbalance'] = self._calculate_imbalance(orderbook)

        # 计算订单簿深度
        metrics['depth'] = self._calculate_depth(orderbook)

        return metrics

    def _calculate_imbalance(self, orderbook: Dict) -> float:
        """计算订单簿不平衡度"""
        bid_vol = sum(vol for _, vol in orderbook['bids'][:5])
        ask_vol = sum(vol for _, vol in orderbook['asks'][:5])

        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def _calculate_depth(self, orderbook: Dict) -> float:
        """计算订单簿深度"""
        depth_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5%深度
        mid_price = (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2
        results = {}

        for level in depth_levels:
            price_range = mid_price * level
            bid_depth = sum(vol for price, vol in orderbook['bids'] if price >= mid_price - price_range)
            ask_depth = sum(vol for price, vol in orderbook['asks'] if price <= mid_price + price_range)
            results[f'depth_{int(level*100)}pct'] = (bid_depth + ask_depth) / 2

        return results

    def register_features(self):
        """向特征管理器注册订单簿特征"""
        self.feature_manager.register(
            name='orderbook_imbalance',
            calculator=lambda sym: self.calculate_metrics(sym)['imbalance'],
            description='订单簿买卖不平衡度[-1,1]'
        )
        self.feature_manager.register(
            name='orderbook_spread',
            calculator=lambda sym: self.calculate_metrics(sym)['spread'],
            description='买卖价差'
        )
