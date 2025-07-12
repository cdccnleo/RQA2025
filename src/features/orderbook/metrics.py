#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
订单簿指标计算
包含各种订单簿衍生指标的计算方法
"""

from typing import List, Tuple
import numpy as np

def calculate_vwap(prices: List[float], volumes: List[float]) -> float:
    """计算成交量加权平均价(VWAP)

    Args:
        prices: 价格列表
        volumes: 成交量列表

    Returns:
        成交量加权平均价
    """
    if not prices or not volumes:
        return 0.0
    return np.dot(prices, volumes) / sum(volumes)

def calculate_twap(prices: List[float]) -> float:
    """计算时间加权平均价(TWAP)

    Args:
        prices: 价格列表

    Returns:
        时间加权平均价
    """
    if not prices:
        return 0.0
    return sum(prices) / len(prices)

def calculate_orderbook_imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 5) -> float:
    """计算订单簿不平衡度

    Args:
        bids: 买盘 [(价格, 数量), ...]
        asks: 卖盘 [(价格, 数量), ...]
        levels: 计算层级数

    Returns:
        不平衡度 [-1,1], 正值表示买盘强势
    """
    bid_vol = sum(vol for _, vol in bids[:levels])
    ask_vol = sum(vol for _, vol in asks[:levels])

    if bid_vol + ask_vol == 0:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def calculate_orderbook_skew(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
    """计算订单簿偏度

    Args:
        bids: 买盘
        asks: 卖盘

    Returns:
        偏度值
    """
    if not bids or not asks:
        return 0.0

    mid_price = (asks[0][0] + bids[0][0]) / 2
    bid_prices = np.array([price for price, _ in bids])
    ask_prices = np.array([price for price, _ in asks])

    bid_skew = np.mean((bid_prices - mid_price) ** 3)
    ask_skew = np.mean((ask_prices - mid_price) ** 3)

    return (ask_skew - bid_skew) / (mid_price ** 3)
