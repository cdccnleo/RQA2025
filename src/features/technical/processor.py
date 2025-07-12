#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标统一处理器
负责所有技术指标的计算和缓存管理
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from ..feature_manager import FeatureManager

@dataclass
class TechnicalIndicator:
    name: str
    value: float
    timestamp: float

class TechnicalProcessor:
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.cache = {}  # 指标缓存 {symbol: {indicator_name: value}}

    def calculate_basic_indicators(self, symbol: str, prices: List[float]) -> Dict[str, float]:
        """计算基础技术指标"""
        indicators = {}

        # 简单移动平均
        indicators['MA5'] = self._calculate_ma(prices, 5)
        indicators['MA10'] = self._calculate_ma(prices, 10)

        # 计算MACD
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        indicators['MACD'] = ema12 - ema26

        # 更新缓存
        self.cache[symbol] = indicators
        return indicators

    def calculate_complex_indicators(self, symbol: str, prices: List[float]) -> Dict[str, float]:
        """计算复杂技术指标"""
        indicators = {}

        # 布林带
        ma20 = self._calculate_ma(prices, 20)
        std = np.std(prices[-20:])
        indicators['BOLL_UP'] = ma20 + 2 * std
        indicators['BOLL_MID'] = ma20
        indicators['BOLL_LOW'] = ma20 - 2 * std

        # RSI
        indicators['RSI14'] = self._calculate_rsi(prices, 14)

        # 更新缓存
        self.cache[symbol].update(indicators)
        return indicators

    def _calculate_ma(self, prices: List[float], window: int) -> float:
        """计算移动平均"""
        if len(prices) < window:
            return sum(prices) / len(prices)
        return sum(prices[-window:]) / window

    def _calculate_ema(self, prices: List[float], window: int) -> float:
        """计算指数移动平均"""
        if len(prices) < window:
            return sum(prices) / len(prices)

        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        return np.dot(prices[-window:], weights)

    def _calculate_rsi(self, prices: List[float], window: int) -> float:
        """计算相对强弱指数"""
        if len(prices) < window + 1:
            return 50.0

        deltas = np.diff(prices)
        seed = deltas[:window]

        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up / down
        return 100 - (100 / (1 + rs))

    def process(self, symbol: str, prices: List[float]) -> List[TechnicalIndicator]:
        """统一处理接口"""
        basic = self.calculate_basic_indicators(symbol, prices)
        complex = self.calculate_complex_indicators(symbol, prices)

        indicators = []
        for name, value in {**basic, **complex}.items():
            indicators.append(TechnicalIndicator(
                name=name,
                value=value,
                timestamp=time.time()
            ))

        return indicators
