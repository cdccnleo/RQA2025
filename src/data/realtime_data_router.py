#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数据路由器

功能：
- 多数据源聚合
- 数据去重和合并
- 优先级路由
- 实时数据分发

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    priority: int = 0  # 优先级，数字越小优先级越高
    enabled: bool = True
    weight: float = 1.0
    timeout: float = 5.0  # 超时时间（秒）


@dataclass
class RealtimeMarketData:
    """实时市场数据"""
    symbol: str
    timestamp: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    amount: float = 0.0
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount,
            'source': self.source
        }


class RealtimeDataRouter:
    """
    实时数据路由器
    
    职责：
    1. 多数据源聚合
    2. 数据去重和合并
    3. 优先级路由
    4. 实时数据分发
    """
    
    def __init__(self):
        """初始化实时数据路由器"""
        # 数据源配置
        self._data_sources: Dict[str, DataSourceConfig] = {}
        
        # 数据源处理器
        self._source_handlers: Dict[str, Callable] = {}
        
        # 订阅者回调
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._global_subscribers: List[Callable] = []
        
        # 数据去重缓存 (symbol_timestamp -> data_hash)
        self._dedup_cache: Dict[str, str] = {}
        self._dedup_ttl = 60  # 去重缓存TTL（秒）
        
        # 最新数据缓存
        self._latest_data: Dict[str, RealtimeMarketData] = {}
        
        # 运行状态
        self._is_running = False
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            'total_messages': 0,
            'deduplicated_messages': 0,
            'delivered_messages': 0
        }
        
        logger.info("实时数据路由器初始化完成")
    
    def register_data_source(
        self,
        name: str,
        handler: Callable,
        priority: int = 0,
        weight: float = 1.0
    ) -> bool:
        """
        注册数据源
        
        Args:
            name: 数据源名称
            handler: 数据源处理器
            priority: 优先级
            weight: 权重
            
        Returns:
            是否注册成功
        """
        try:
            with self._lock:
                self._data_sources[name] = DataSourceConfig(
                    name=name,
                    priority=priority,
                    weight=weight
                )
                self._source_handlers[name] = handler
            
            logger.info(f"数据源注册成功: {name}, 优先级: {priority}")
            return True
            
        except Exception as e:
            logger.error(f"数据源注册失败 {name}: {e}")
            return False
    
    def unregister_data_source(self, name: str) -> bool:
        """
        注销数据源
        
        Args:
            name: 数据源名称
            
        Returns:
            是否注销成功
        """
        try:
            with self._lock:
                if name in self._data_sources:
                    del self._data_sources[name]
                if name in self._source_handlers:
                    del self._source_handlers[name]
            
            logger.info(f"数据源注销成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"数据源注销失败 {name}: {e}")
            return False
    
    def subscribe(self, symbol: str, callback: Callable):
        """
        订阅股票实时数据
        
        Args:
            symbol: 股票代码
            callback: 回调函数
        """
        with self._lock:
            if callback not in self._subscribers[symbol]:
                self._subscribers[symbol].append(callback)
        
        logger.info(f"订阅实时数据: {symbol}")
    
    def unsubscribe(self, symbol: str, callback: Callable):
        """
        取消订阅股票实时数据
        
        Args:
            symbol: 股票代码
            callback: 回调函数
        """
        with self._lock:
            if symbol in self._subscribers:
                if callback in self._subscribers[symbol]:
                    self._subscribers[symbol].remove(callback)
        
        logger.info(f"取消订阅实时数据: {symbol}")
    
    def subscribe_all(self, callback: Callable):
        """
        订阅所有股票实时数据
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback not in self._global_subscribers:
                self._global_subscribers.append(callback)
        
        logger.info("订阅所有实时数据")
    
    def unsubscribe_all(self, callback: Callable):
        """
        取消订阅所有股票实时数据
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
        
        logger.info("取消订阅所有实时数据")
    
    def route_data(self, data: RealtimeMarketData) -> bool:
        """
        路由实时数据
        
        Args:
            data: 实时市场数据
            
        Returns:
            是否成功路由
        """
        try:
            # 去重检查
            if self._is_duplicate(data):
                self._stats['deduplicated_messages'] += 1
                return False
            
            # 更新最新数据缓存
            with self._lock:
                self._latest_data[data.symbol] = data
            
            # 分发数据
            self._deliver_data(data)
            
            self._stats['total_messages'] += 1
            return True
            
        except Exception as e:
            logger.error(f"路由数据失败 {data.symbol}: {e}")
            return False
    
    def _is_duplicate(self, data: RealtimeMarketData) -> bool:
        """
        检查数据是否重复
        
        Args:
            data: 实时市场数据
            
        Returns:
            是否重复
        """
        # 生成去重键
        dedup_key = f"{data.symbol}_{data.timestamp.isoformat()}"
        
        # 生成数据哈希（简化版）
        data_hash = f"{data.open}_{data.high}_{data.low}_{data.close}_{data.volume}"
        
        with self._lock:
            # 检查是否已存在
            if dedup_key in self._dedup_cache:
                return self._dedup_cache[dedup_key] == data_hash
            
            # 添加到去重缓存
            self._dedup_cache[dedup_key] = data_hash
            
            # 清理过期缓存（简化版，实际应该使用定时任务）
            if len(self._dedup_cache) > 10000:
                # 保留最近一半
                keys = list(self._dedup_cache.keys())
                for key in keys[:5000]:
                    del self._dedup_cache[key]
        
        return False
    
    def _deliver_data(self, data: RealtimeMarketData):
        """
        分发数据到订阅者
        
        Args:
            data: 实时市场数据
        """
        symbol = data.symbol
        
        # 分发给特定股票的订阅者
        with self._lock:
            subscribers = self._subscribers.get(symbol, []).copy()
            global_subscribers = self._global_subscribers.copy()
        
        # 调用特定股票订阅者
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"调用订阅者回调失败 {symbol}: {e}")
        
        # 调用全局订阅者
        for callback in global_subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"调用全局订阅者回调失败: {e}")
        
        self._stats['delivered_messages'] += 1
    
    def get_latest_data(self, symbol: str) -> Optional[RealtimeMarketData]:
        """
        获取最新数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            最新市场数据
        """
        with self._lock:
            return self._latest_data.get(symbol)
    
    def get_all_latest_data(self) -> Dict[str, RealtimeMarketData]:
        """
        获取所有最新数据
        
        Returns:
            股票代码到最新数据的映射
        """
        with self._lock:
            return self._latest_data.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                **self._stats,
                'subscribed_symbols': len(self._subscribers),
                'global_subscribers': len(self._global_subscribers),
                'data_sources': len(self._data_sources),
                'latest_data_count': len(self._latest_data)
            }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._dedup_cache.clear()
            self._latest_data.clear()
        
        logger.info("缓存已清空")


# 单例实例
_router: Optional[RealtimeDataRouter] = None


def get_realtime_data_router() -> RealtimeDataRouter:
    """
    获取实时数据路由器单例
    
    Returns:
        RealtimeDataRouter实例
    """
    global _router
    if _router is None:
        _router = RealtimeDataRouter()
    return _router
