#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时信号集成服务

功能：
- 实时数据订阅和处理
- 实时信号生成
- WebSocket推送
- 信号缓存和去重

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading

from .realtime_data_router import (
    get_realtime_data_router,
    RealtimeMarketData,
    RealtimeDataRouter
)

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: str  # buy, sell, hold
    confidence: float  # 0-1
    timestamp: datetime
    strategy_id: str
    price: float = 0.0
    volume: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'price': self.price,
            'volume': self.volume,
            'metadata': self.metadata or {}
        }


class RealtimeSignalIntegration:
    """
    实时信号集成服务
    
    职责：
    1. 实时数据订阅和处理
    2. 实时信号生成
    3. WebSocket推送
    4. 信号缓存和去重
    """
    
    def __init__(self):
        """初始化实时信号集成服务"""
        # 实时数据路由器
        self._router = get_realtime_data_router()
        
        # 信号生成器
        self._signal_generators: Dict[str, Callable] = {}
        
        # 信号缓存（用于去重）
        self._signal_cache: Dict[str, datetime] = {}
        self._signal_cache_ttl = timedelta(minutes=5)
        
        # WebSocket推送回调
        self._websocket_callbacks: List[Callable] = []
        
        # 运行状态
        self._is_running = False
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            'signals_generated': 0,
            'signals_pushed': 0,
            'signals_deduplicated': 0
        }
        
        logger.info("实时信号集成服务初始化完成")
    
    def start(self):
        """启动服务"""
        if self._is_running:
            logger.warning("服务已在运行中")
            return
        
        self._is_running = True
        
        # 订阅全局实时数据
        self._router.subscribe_all(self._on_market_data)
        
        logger.info("实时信号集成服务已启动")
    
    def stop(self):
        """停止服务"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # 取消订阅
        self._router.unsubscribe_all(self._on_market_data)
        
        logger.info("实时信号集成服务已停止")
    
    def register_signal_generator(
        self,
        strategy_id: str,
        generator: Callable[[RealtimeMarketData], Optional[TradingSignal]]
    ):
        """
        注册信号生成器
        
        Args:
            strategy_id: 策略ID
            generator: 信号生成函数
        """
        with self._lock:
            self._signal_generators[strategy_id] = generator
        
        logger.info(f"信号生成器注册成功: {strategy_id}")
    
    def unregister_signal_generator(self, strategy_id: str):
        """
        注销信号生成器
        
        Args:
            strategy_id: 策略ID
        """
        with self._lock:
            if strategy_id in self._signal_generators:
                del self._signal_generators[strategy_id]
        
        logger.info(f"信号生成器注销成功: {strategy_id}")
    
    def register_websocket_callback(self, callback: Callable):
        """
        注册WebSocket推送回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback not in self._websocket_callbacks:
                self._websocket_callbacks.append(callback)
        
        logger.info("WebSocket回调注册成功")
    
    def unregister_websocket_callback(self, callback: Callable):
        """
        注销WebSocket推送回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback in self._websocket_callbacks:
                self._websocket_callbacks.remove(callback)
        
        logger.info("WebSocket回调注销成功")
    
    def _on_market_data(self, data: RealtimeMarketData):
        """
        处理实时市场数据
        
        Args:
            data: 实时市场数据
        """
        if not self._is_running:
            return
        
        try:
            # 生成信号
            signals = self._generate_signals(data)
            
            # 推送信号
            for signal in signals:
                self._push_signal(signal)
                
        except Exception as e:
            logger.error(f"处理市场数据失败 {data.symbol}: {e}")
    
    def _generate_signals(
        self,
        data: RealtimeMarketData
    ) -> List[TradingSignal]:
        """
        生成交易信号
        
        Args:
            data: 实时市场数据
            
        Returns:
            交易信号列表
        """
        signals = []
        
        with self._lock:
            generators = list(self._signal_generators.items())
        
        for strategy_id, generator in generators:
            try:
                signal = generator(data)
                if signal:
                    # 检查信号是否重复
                    if not self._is_signal_duplicate(signal):
                        signals.append(signal)
                        self._stats['signals_generated'] += 1
                    else:
                        self._stats['signals_deduplicated'] += 1
                        
            except Exception as e:
                logger.error(f"生成信号失败 {strategy_id}: {e}")
        
        return signals
    
    def _is_signal_duplicate(self, signal: TradingSignal) -> bool:
        """
        检查信号是否重复
        
        Args:
            signal: 交易信号
            
        Returns:
            是否重复
        """
        # 生成信号键
        signal_key = f"{signal.symbol}_{signal.signal_type}_{signal.strategy_id}"
        
        with self._lock:
            if signal_key in self._signal_cache:
                last_time = self._signal_cache[signal_key]
                if datetime.now() - last_time < self._signal_cache_ttl:
                    return True
            
            # 更新缓存
            self._signal_cache[signal_key] = datetime.now()
            
            # 清理过期缓存
            self._cleanup_signal_cache()
        
        return False
    
    def _cleanup_signal_cache(self):
        """清理过期信号缓存"""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self._signal_cache.items()
            if now - timestamp > self._signal_cache_ttl
        ]
        for key in expired_keys:
            del self._signal_cache[key]
    
    def _push_signal(self, signal: TradingSignal):
        """
        推送信号
        
        Args:
            signal: 交易信号
        """
        with self._lock:
            callbacks = self._websocket_callbacks.copy()
        
        # 调用所有WebSocket回调
        for callback in callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"推送信号失败: {e}")
        
        self._stats['signals_pushed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                **self._stats,
                'signal_generators': len(self._signal_generators),
                'websocket_callbacks': len(self._websocket_callbacks),
                'signal_cache_size': len(self._signal_cache)
            }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._signal_cache.clear()
        
        logger.info("信号缓存已清空")


# 简单的信号生成器示例
def simple_momentum_signal_generator(
    data: RealtimeMarketData
) -> Optional[TradingSignal]:
    """
    简单动量信号生成器
    
    Args:
        data: 实时市场数据
        
    Returns:
        交易信号或None
    """
    # 这里可以实现更复杂的信号生成逻辑
    # 目前只是一个示例
    
    # 如果价格变化超过阈值，生成信号
    price_change = (data.close - data.open) / data.open if data.open > 0 else 0
    
    if abs(price_change) > 0.02:  # 2%阈值
        signal_type = 'buy' if price_change > 0 else 'sell'
        confidence = min(abs(price_change) * 10, 1.0)  # 最大置信度1.0
        
        return TradingSignal(
            symbol=data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=datetime.now(),
            strategy_id='simple_momentum',
            price=data.close,
            volume=data.volume,
            metadata={
                'price_change': price_change,
                'source': data.source
            }
        )
    
    return None


# 单例实例
_integration: Optional[RealtimeSignalIntegration] = None


def get_realtime_signal_integration() -> RealtimeSignalIntegration:
    """
    获取实时信号集成服务单例
    
    Returns:
        RealtimeSignalIntegration实例
    """
    global _integration
    if _integration is None:
        _integration = RealtimeSignalIntegration()
    return _integration
