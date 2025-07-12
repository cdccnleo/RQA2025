#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Level2数据分析
处理A股Level2行情数据
"""

from typing import Dict, List, Tuple
from .analyzer import OrderbookAnalyzer
from ..manager import FeatureManager

class Level2Processor:
    def __init__(self, feature_manager: FeatureManager):
        self.analyzer = OrderbookAnalyzer(feature_manager)
        self.symbol_map = {}  # 代码映射表

    def process_snapshot(self, symbol: str, data: Dict) -> None:
        """处理Level2快照数据

        Args:
            symbol: 标的代码
            data: Level2快照数据
        """
        # 标准化代码
        normalized_symbol = self._normalize_symbol(symbol)

        # 解析订单簿数据
        bids = self._parse_orderbook(data.get('Bid', []))
        asks = self._parse_orderbook(data.get('Ask', []))

        # 更新分析器
        self.analyzer.update_orderbook(normalized_symbol, bids, asks)

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化标的代码

        Args:
            symbol: 原始代码

        Returns:
            标准化后的代码
        """
        if symbol not in self.symbol_map:
            # 添加市场后缀
            if symbol.startswith(('6', '5')):
                self.symbol_map[symbol] = f"{symbol}.SH"
            elif symbol.startswith(('0', '3')):
                self.symbol_map[symbol] = f"{symbol}.SZ"
            else:
                self.symbol_map[symbol] = symbol
        return self.symbol_map[symbol]

    def _parse_orderbook(self, raw_data: List[Dict]) -> List[Tuple[float, float]]:
        """解析原始订单簿数据

        Args:
            raw_data: 原始订单簿数据

        Returns:
            [(价格, 数量), ...]
        """
        return [
            (float(item['Price']), float(item['Volume']))
            for item in raw_data
            if 'Price' in item and 'Volume' in item
        ]

    def get_features(self, symbol: str) -> Dict[str, float]:
        """获取特征数据

        Args:
            symbol: 标的代码

        Returns:
            特征字典
        """
        normalized_symbol = self._normalize_symbol(symbol)
        return self.analyzer.calculate_metrics(normalized_symbol)
