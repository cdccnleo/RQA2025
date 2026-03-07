#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level2行情数据适配器

功能：
- 获取Level2行情数据（十档/五档盘口）
- 支持逐笔成交数据
- 支持委托队列数据
- 实时数据流处理

作者: AI Assistant
创建日期: 2026-02-21
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Level2DataType(Enum):
    """Level2数据类型"""
    ORDER_BOOK = "order_book"           # 十档/五档盘口
    TICK_TRADE = "tick_trade"           # 逐笔成交
    ORDER_QUEUE = "order_queue"         # 委托队列
    BROKER_QUEUE = "broker_queue"       # 经纪商队列


@dataclass
class OrderBookLevel:
    """订单簿档位"""
    level: int                          # 档位（1-10）
    bid_price: float                    # 买价
    bid_volume: int                     # 买量
    bid_orders: int                     # 买委托数
    ask_price: float                    # 卖价
    ask_volume: int                     # 卖量
    ask_orders: int                     # 卖委托数


@dataclass
class OrderBook:
    """订单簿"""
    symbol: str
    timestamp: datetime
    levels: List[OrderBookLevel] = field(default_factory=list)
    total_bid_volume: int = 0
    total_ask_volume: int = 0
    weighted_bid_price: float = 0.0
    weighted_ask_price: float = 0.0
    spread: float = 0.0
    mid_price: float = 0.0


@dataclass
class TickTrade:
    """逐笔成交"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    side: str                           # 'buy' or 'sell'
    order_type: str                     # 'market', 'limit', etc.
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None
    broker: Optional[str] = None


@dataclass
class OrderQueueItem:
    """委托队列项"""
    order_id: str
    price: float
    volume: int
    side: str
    timestamp: datetime
    broker: Optional[str] = None


@dataclass
class OrderQueue:
    """委托队列"""
    symbol: str
    price: float
    side: str
    timestamp: datetime
    orders: List[OrderQueueItem] = field(default_factory=list)
    total_volume: int = 0


class Level2MarketDataAdapter:
    """
    Level2行情数据适配器
    
    支持：
    - 十档/五档盘口数据
    - 逐笔成交数据
    - 委托队列数据
    - 实时数据流订阅
    """
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        websocket_url: Optional[str] = None
    ):
        """
        初始化Level2适配器
        
        Args:
            name: 适配器名称
            api_key: API密钥
            websocket_url: WebSocket连接地址
        """
        self.name = name
        self.api_key = api_key
        self.websocket_url = websocket_url
        
        # 连接状态
        self._is_connected = False
        self._websocket = None
        
        # 数据缓存
        self._order_book_cache: Dict[str, OrderBook] = {}
        self._last_trade_cache: Dict[str, TickTrade] = {}
        
        # 订阅管理
        self._subscribers: Dict[str, List[Callable]] = {
            Level2DataType.ORDER_BOOK.value: [],
            Level2DataType.TICK_TRADE.value: [],
            Level2DataType.ORDER_QUEUE.value: []
        }
        
        # 统计
        self._message_count = 0
        self._start_time: Optional[datetime] = None
        
        logger.info(f"Level2行情适配器 {name} 初始化完成")
    
    async def connect(self) -> bool:
        """
        连接Level2数据源
        
        Returns:
            是否连接成功
        """
        try:
            # 这里应该实现实际的连接逻辑
            # 例如连接WebSocket或HTTP API
            self._is_connected = True
            self._start_time = datetime.now()
            
            logger.info(f"Level2行情适配器 {self.name} 连接成功")
            return True
            
        except Exception as e:
            self._is_connected = False
            logger.error(f"Level2行情适配器连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开Level2数据源连接
        
        Returns:
            是否断开成功
        """
        self._is_connected = False
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info(f"Level2行情适配器 {self.name} 已断开")
        return True
    
    async def subscribe_order_book(
        self,
        symbol: str,
        levels: int = 10,
        callback: Optional[Callable[[OrderBook], None]] = None
    ) -> bool:
        """
        订阅订单簿数据
        
        Args:
            symbol: 股票代码
            levels: 档位数量（5或10）
            callback: 数据回调函数
            
        Returns:
            是否订阅成功
        """
        if not self._is_connected:
            logger.error("适配器未连接")
            return False
        
        try:
            # 注册回调
            if callback:
                self._subscribers[Level2DataType.ORDER_BOOK.value].append(callback)
            
            # 这里应该实现实际的订阅逻辑
            logger.info(f"订阅 {symbol} 的{levels}档盘口数据")
            return True
            
        except Exception as e:
            logger.error(f"订阅订单簿失败 {symbol}: {e}")
            return False
    
    async def subscribe_tick_trade(
        self,
        symbol: str,
        callback: Optional[Callable[[TickTrade], None]] = None
    ) -> bool:
        """
        订阅逐笔成交数据
        
        Args:
            symbol: 股票代码
            callback: 数据回调函数
            
        Returns:
            是否订阅成功
        """
        if not self._is_connected:
            logger.error("适配器未连接")
            return False
        
        try:
            # 注册回调
            if callback:
                self._subscribers[Level2DataType.TICK_TRADE.value].append(callback)
            
            logger.info(f"订阅 {symbol} 的逐笔成交数据")
            return True
            
        except Exception as e:
            logger.error(f"订阅逐笔成交失败 {symbol}: {e}")
            return False
    
    async def get_order_book(
        self,
        symbol: str,
        levels: int = 10
    ) -> Optional[OrderBook]:
        """
        获取订单簿快照
        
        Args:
            symbol: 股票代码
            levels: 档位数量
            
        Returns:
            订单簿数据
        """
        if not self._is_connected:
            raise ConnectionError("适配器未连接")
        
        try:
            # 检查缓存
            if symbol in self._order_book_cache:
                return self._order_book_cache[symbol]
            
            # 这里应该实现实际的API调用
            # 模拟数据
            order_book = self._generate_mock_order_book(symbol, levels)
            self._order_book_cache[symbol] = order_book
            
            return order_book
            
        except Exception as e:
            logger.error(f"获取订单簿失败 {symbol}: {e}")
            return None
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[TickTrade]:
        """
        获取最近成交数据
        
        Args:
            symbol: 股票代码
            limit: 返回条数
            
        Returns:
            成交数据列表
        """
        if not self._is_connected:
            raise ConnectionError("适配器未连接")
        
        try:
            # 这里应该实现实际的API调用
            # 模拟数据
            trades = self._generate_mock_trades(symbol, limit)
            return trades
            
        except Exception as e:
            logger.error(f"获取成交数据失败 {symbol}: {e}")
            return []
    
    async def get_order_queue(
        self,
        symbol: str,
        price: float,
        side: str
    ) -> Optional[OrderQueue]:
        """
        获取委托队列
        
        Args:
            symbol: 股票代码
            price: 价格
            side: 方向 ('buy' or 'sell')
            
        Returns:
            委托队列数据
        """
        if not self._is_connected:
            raise ConnectionError("适配器未连接")
        
        try:
            # 这里应该实现实际的API调用
            # 模拟数据
            queue = self._generate_mock_order_queue(symbol, price, side)
            return queue
            
        except Exception as e:
            logger.error(f"获取委托队列失败 {symbol}: {e}")
            return None
    
    def _generate_mock_order_book(self, symbol: str, levels: int) -> OrderBook:
        """生成模拟订单簿数据"""
        base_price = 100.0
        
        order_book_levels = []
        total_bid_volume = 0
        total_ask_volume = 0
        weighted_bid_price = 0.0
        weighted_ask_price = 0.0
        
        for i in range(1, levels + 1):
            bid_price = base_price - i * 0.01
            ask_price = base_price + i * 0.01
            
            bid_volume = np.random.randint(1000, 10000)
            ask_volume = np.random.randint(1000, 10000)
            
            bid_orders = np.random.randint(10, 100)
            ask_orders = np.random.randint(10, 100)
            
            level = OrderBookLevel(
                level=i,
                bid_price=round(bid_price, 2),
                bid_volume=bid_volume,
                bid_orders=bid_orders,
                ask_price=round(ask_price, 2),
                ask_volume=ask_volume,
                ask_orders=ask_orders
            )
            
            order_book_levels.append(level)
            
            total_bid_volume += bid_volume
            total_ask_volume += ask_volume
            weighted_bid_price += bid_price * bid_volume
            weighted_ask_price += ask_price * ask_volume
        
        # 计算加权平均价格
        if total_bid_volume > 0:
            weighted_bid_price /= total_bid_volume
        if total_ask_volume > 0:
            weighted_ask_price /= total_ask_volume
        
        # 计算价差和中价
        best_bid = order_book_levels[0].bid_price
        best_ask = order_book_levels[0].ask_price
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            levels=order_book_levels,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            weighted_bid_price=round(weighted_bid_price, 4),
            weighted_ask_price=round(weighted_ask_price, 4),
            spread=round(spread, 4),
            mid_price=round(mid_price, 4)
        )
    
    def _generate_mock_trades(self, symbol: str, limit: int) -> List[TickTrade]:
        """生成模拟成交数据"""
        trades = []
        base_price = 100.0
        
        for i in range(limit):
            price = base_price + np.random.normal(0, 0.1)
            volume = np.random.randint(100, 10000)
            side = 'buy' if np.random.random() > 0.5 else 'sell'
            
            trade = TickTrade(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(seconds=i),
                price=round(price, 2),
                volume=volume,
                side=side,
                order_type='limit',
                broker=f'Broker_{np.random.randint(1, 10)}'
            )
            
            trades.append(trade)
        
        return trades
    
    def _generate_mock_order_queue(
        self,
        symbol: str,
        price: float,
        side: str
    ) -> OrderQueue:
        """生成模拟委托队列"""
        orders = []
        total_volume = 0
        
        for i in range(20):  # 前20个委托
            volume = np.random.randint(100, 5000)
            order = OrderQueueItem(
                order_id=f'ORDER_{i:06d}',
                price=price,
                volume=volume,
                side=side,
                timestamp=datetime.now() - timedelta(seconds=i * 10),
                broker=f'Broker_{np.random.randint(1, 10)}'
            )
            
            orders.append(order)
            total_volume += volume
        
        return OrderQueue(
            symbol=symbol,
            price=price,
            side=side,
            timestamp=datetime.now(),
            orders=orders,
            total_volume=total_volume
        )
    
    def calculate_order_book_imbalance(self, symbol: str) -> float:
        """
        计算订单簿失衡度
        
        Args:
            symbol: 股票代码
            
        Returns:
            失衡度 (-1.0 to 1.0)
        """
        order_book = self._order_book_cache.get(symbol)
        if not order_book:
            return 0.0
        
        total_volume = order_book.total_bid_volume + order_book.total_ask_volume
        if total_volume == 0:
            return 0.0
        
        # 买方失衡为正，卖方失衡为负
        imbalance = (order_book.total_bid_volume - order_book.total_ask_volume) / total_volume
        
        return round(imbalance, 4)
    
    def calculate_trade_pressure(self, symbol: str, window_seconds: int = 60) -> float:
        """
        计算成交压力
        
        Args:
            symbol: 股票代码
            window_seconds: 时间窗口（秒）
            
        Returns:
            成交压力 (-1.0 to 1.0)
        """
        # 这里应该基于实际成交数据计算
        # 简化实现
        return np.random.uniform(-0.5, 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        uptime = datetime.now() - self._start_time if self._start_time else timedelta(0)
        
        return {
            'is_connected': self._is_connected,
            'uptime_seconds': uptime.total_seconds(),
            'message_count': self._message_count,
            'cached_symbols': len(self._order_book_cache),
            'subscribers': {
                data_type: len(callbacks)
                for data_type, callbacks in self._subscribers.items()
            }
        }


# 全局适配器实例
_level2_adapter: Optional[Level2MarketDataAdapter] = None


async def get_level2_adapter(
    name: str = "Level2Default",
    api_key: Optional[str] = None
) -> Level2MarketDataAdapter:
    """
    获取Level2适配器实例（单例模式）
    
    Args:
        name: 适配器名称
        api_key: API密钥
        
    Returns:
        Level2MarketDataAdapter实例
    """
    global _level2_adapter
    
    if _level2_adapter is None:
        _level2_adapter = Level2MarketDataAdapter(name, api_key)
        await _level2_adapter.connect()
    
    return _level2_adapter
