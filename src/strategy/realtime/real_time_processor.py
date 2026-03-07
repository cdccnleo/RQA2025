#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时处理器
Real - time Processor

支持毫秒级响应的实时策略执行和高频交易。
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging

from ..interfaces.strategy_interfaces import StrategyConfig, StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class RealTimeConfig:

    """实时配置"""
    buffer_size: int = 1000  # 数据缓冲区大小
    processing_interval: float = 0.001  # 处理间隔(毫秒)
    max_latency: float = 0.01  # 最大延迟(秒)
    batch_size: int = 10  # 批处理大小
    enable_caching: bool = True
    cache_ttl: int = 60  # 缓存TTL(秒)


@dataclass
class MarketData:

    """市场数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTimeMetrics:

    """实时性能指标"""
    processing_latency: float
    throughput: float
    queue_length: int
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class RealTimeDataStream:

    """实时数据流"""

    def __init__(self, config: RealTimeConfig):

        self.config = config
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        self.metrics = RealTimeMetrics(0, 0, 0, 0, 0)

        # 数据缓存
        self.data_cache: Dict[str, MarketData] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # 统计信息
        self.total_processed = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0

    async def start(self):
        """启动实时数据流"""
        self.is_running = True
        logger.info("RealTimeDataStream started")

        # 启动处理循环
        asyncio.create_task(self._processing_loop())

        # 启动监控循环
        asyncio.create_task(self._monitoring_loop())

    async def stop(self):
        """停止实时数据流"""
        self.is_running = False
        logger.info("RealTimeDataStream stopped")

    async def ingest_data(self, data: MarketData):
        """摄入市场数据"""
        if not self.is_running:
            return

        # 检查缓存
        if self.config.enable_caching:
            cache_key = f"{data.symbol}_{data.timestamp.isoformat()}"
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                return  # 使用缓存数据
            else:
                self.cache_misses += 1

        # 添加到缓冲区
        self.data_buffer.append(data)

        # 添加到处理队列
        await self.processing_queue.put(data)

        # 更新缓存
        if self.config.enable_caching:
            cache_key = f"{data.symbol}_{data.timestamp.isoformat()}"
            self.data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache_timestamps:
            return False

        age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return age < self.config.cache_ttl

    async def _processing_loop(self):
        """数据处理循环"""
        batch = []

        while self.is_running:
            try:
                # 收集批数据
                while len(batch) < self.config.batch_size:
                    try:
                        data = await asyncio.wait_for(
                            self.processing_queue.get(),
                            timeout=self.config.processing_interval
                        )
                        batch.append(data)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    start_time = time.time()

                    # 批处理数据
                    await self._process_batch(batch)

                    # 更新指标
                    processing_time = time.time() - start_time
                    self.metrics.processing_latency = processing_time / len(batch)
                    self.metrics.throughput = len(batch) / processing_time
                    self.total_processed += len(batch)

                    batch.clear()

                await asyncio.sleep(self.config.processing_interval)

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.total_errors += 1
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[MarketData]):
        """处理数据批"""
        # 这里可以实现批处理逻辑
        # 例如：计算技术指标、触发策略信号等

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self.metrics.queue_length = self.processing_queue.qsize()
                self.metrics.cache_hit_rate = (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0 else 0
                )
                self.metrics.error_rate = (
                    self.total_errors / self.total_processed
                    if self.total_processed > 0 else 0
                )
                self.metrics.timestamp = datetime.now()

                await asyncio.sleep(1.0)  # 每秒更新一次

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)

    def get_metrics(self) -> RealTimeMetrics:
        """获取性能指标"""
        return self.metrics


class HighFrequencyStrategy:

    """高频交易策略"""

    def __init__(self, config: StrategyConfig):

        self.config = config
        self.position = 0
        self.cash = 100000  # 初始资金
        self.trades = []

        # 高频交易参数
        self.max_position = config.parameters.get('max_position', 100)
        self.min_spread = config.parameters.get('min_spread', 0.001)
        self.max_slippage = config.parameters.get('max_slippage', 0.0001)
        self.order_size = config.parameters.get('order_size', 10)

        # 市场数据缓存
        self.last_price = 0
        self.bid_ask_spread = 0

        logger.info(f"HighFrequencyStrategy initialized: {config.strategy_id}")

    def process_market_data(self, data: MarketData) -> List[StrategySignal]:
        """处理市场数据"""
        signals = []

        try:
            # 更新市场状态
            self.last_price = data.price
            self.bid_ask_spread = data.ask - data.bid

            # 高频交易逻辑
            if self._should_enter_long(data):
                signals.append(StrategySignal(
                    signal_id=str(len(signals) + 1),
                    strategy_id=self.config.strategy_id,
                    signal_type="BUY",
                    symbol=data.symbol,
                    quantity=self.order_size,
                    price=data.ask,
                    timestamp=data.timestamp,
                    metadata={
                        'reason': 'high_frequency_entry',
                        'spread': self.bid_ask_spread,
                        'confidence': self._calculate_signal_confidence(data)
                    }
                ))

            elif self._should_enter_short(data):
                signals.append(StrategySignal(
                    signal_id=str(len(signals) + 1),
                    strategy_id=self.config.strategy_id,
                    signal_type="SELL",
                    symbol=data.symbol,
                    quantity=self.order_size,
                    price=data.bid,
                    timestamp=data.timestamp,
                    metadata={
                        'reason': 'high_frequency_exit',
                        'spread': self.bid_ask_spread,
                        'confidence': self._calculate_signal_confidence(data)
                    }
                ))

            elif self._should_close_position(data):
                if self.position > 0:
                    signals.append(StrategySignal(
                        signal_id=str(len(signals) + 1),
                        strategy_id=self.config.strategy_id,
                        signal_type="SELL",
                        symbol=data.symbol,
                        quantity=abs(self.position),
                        price=data.bid,
                        timestamp=data.timestamp,
                        metadata={'reason': 'position_close'}
                    ))
                elif self.position < 0:
                    signals.append(StrategySignal(
                        signal_id=str(len(signals) + 1),
                        strategy_id=self.config.strategy_id,
                        signal_type="BUY",
                        symbol=data.symbol,
                        quantity=abs(self.position),
                        price=data.ask,
                        timestamp=data.timestamp,
                        metadata={'reason': 'position_close'}
                    ))

        except Exception as e:
            logger.error(f"High frequency processing error: {e}")

        return signals

    def _should_enter_long(self, data: MarketData) -> bool:
        """判断是否应该做多"""
        # 高频交易入场条件
        if self.position >= self.max_position:
            return False

        # 价差足够大
        if self.bid_ask_spread < self.min_spread:
            return False

        # 价格相对稳定（避免在价格剧烈波动时入场）
        price_change = abs(data.price - self.last_price) / self.last_price
        if price_change > self.max_slippage:
            return False

        # 基于订单簿的判断
        if data.bid > self.last_price and data.ask > data.bid:
            return True

        return False

    def _should_enter_short(self, data: MarketData) -> bool:
        """判断是否应该做空"""
        # 高频交易出场条件
        if self.position <= -self.max_position:
            return False

        # 价差足够大
        if self.bid_ask_spread < self.min_spread:
            return False

        # 价格相对稳定
        price_change = abs(data.price - self.last_price) / self.last_price
        if price_change > self.max_slippage:
            return False

        # 基于订单簿的判断
        if data.ask < self.last_price and data.bid < data.ask:
            return True

        return False

    def _should_close_position(self, data: MarketData) -> bool:
        """判断是否应该平仓"""
        if self.position == 0:
            return False

        # 价差过大时平仓
        if self.bid_ask_spread > self.min_spread * 2:
            return True

        # 价格大幅波动时平仓
        price_change = abs(data.price - self.last_price) / self.last_price
        if price_change > self.max_slippage * 2:
            return True

        return False

    def _calculate_signal_confidence(self, data: MarketData) -> float:
        """计算信号置信度"""
        # 基于多个因素计算置信度
        spread_factor = min(self.bid_ask_spread / self.min_spread, 2.0)
        stability_factor = 1.0 - min(
            abs(data.price - self.last_price) / self.last_price / self.max_slippage,
            1.0
        )
        volume_factor = min(data.volume / 1000, 1.0)  # 成交量因子

        confidence = (spread_factor + stability_factor + volume_factor) / 3
        return min(confidence, 1.0)

    def update_position(self, signal: StrategySignal):
        """更新持仓"""
        if signal.signal_type == "BUY":
            self.position += signal.quantity
            self.cash -= signal.price * signal.quantity
        elif signal.signal_type == "SELL":
            self.position -= signal.quantity
            self.cash += signal.price * signal.quantity

        self.trades.append({
            'timestamp': signal.timestamp,
            'type': signal.signal_type,
            'quantity': signal.quantity,
            'price': signal.price,
            'position': self.position,
            'cash': self.cash
        })


class RealTimeStrategyEngine:

    """实时策略引擎"""

    def __init__(self, config: RealTimeConfig = None):

        self.config = config or RealTimeConfig()
        self.data_stream = RealTimeDataStream(self.config)
        self.strategies: Dict[str, HighFrequencyStrategy] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 性能监控
        self.performance_monitor = RealTimePerformanceMonitor()

        logger.info("RealTimeStrategyEngine initialized")

    async def start(self):
        """启动实时引擎"""
        await self.data_stream.start()
        await self.performance_monitor.start()
        logger.info("RealTimeStrategyEngine started")

    async def stop(self):
        """停止实时引擎"""
        await self.data_stream.stop()
        await self.performance_monitor.stop()
        self.executor.shutdown(wait=True)
        logger.info("RealTimeStrategyEngine stopped")

    def register_strategy(self, config: StrategyConfig):
        """注册策略"""
        strategy = HighFrequencyStrategy(config)
        self.strategies[config.strategy_id] = strategy
        logger.info(f"Strategy registered: {config.strategy_id}")

    def unregister_strategy(self, strategy_id: str):
        """注销策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info(f"Strategy unregistered: {strategy_id}")

    async def process_market_data(self, data: MarketData):
        """处理市场数据"""
        # 摄入数据到流
        await self.data_stream.ingest_data(data)

        # 并行处理所有策略
        tasks = []
        for strategy in self.strategies.values():
            task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                strategy.process_market_data,
                data
            )
            tasks.append(task)

        # 收集所有信号
        all_signals = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_signals.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Strategy processing error: {result}")

        # 更新性能监控
        await self.performance_monitor.record_signals(len(all_signals))

        return all_signals

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        stream_metrics = self.data_stream.get_metrics()

        return {
            'stream_metrics': {
                'processing_latency': stream_metrics.processing_latency,
                'throughput': stream_metrics.throughput,
                'queue_length': stream_metrics.queue_length,
                'cache_hit_rate': stream_metrics.cache_hit_rate,
                'error_rate': stream_metrics.error_rate
            },
            'strategy_metrics': {
                'active_strategies': len(self.strategies),
                'total_positions': sum(s.position for s in self.strategies.values()),
                'total_trades': sum(len(s.trades) for s in self.strategies.values())
            },
            'system_metrics': self.performance_monitor.get_metrics()
        }


class RealTimePerformanceMonitor:

    """实时性能监控器"""

    def __init__(self):

        self.is_running = False
        self.metrics_history = deque(maxlen=1000)

        # 性能指标
        self.signals_processed = 0
        self.average_latency = 0
        self.peak_throughput = 0
        self.error_count = 0

    async def start(self):
        """启动监控"""
        self.is_running = True
        logger.info("RealTimePerformanceMonitor started")

    async def stop(self):
        """停止监控"""
        self.is_running = False
        logger.info("RealTimePerformanceMonitor stopped")

    async def record_signals(self, signal_count: int):
        """记录信号处理"""
        if not self.is_running:
            return

        self.signals_processed += signal_count

        # 计算实时指标
        current_time = time.time()
        metrics = {
            'timestamp': current_time,
            'signals_processed': self.signals_processed,
            'average_latency': self.average_latency,
            'peak_throughput': self.peak_throughput,
            'error_count': self.error_count
        }

        self.metrics_history.append(metrics)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        # 计算趋势
        if len(self.metrics_history) > 1:
            prev = self.metrics_history[-2]
            throughput_trend = (
                (latest['signals_processed'] - prev['signals_processed'])
                / (latest['timestamp'] - prev['timestamp'])
            )
        else:
            throughput_trend = 0

        return {
            'current_throughput': throughput_trend,
            'average_latency': latest.get('average_latency', 0),
            'peak_throughput': self.peak_throughput,
            'total_signals': latest['signals_processed'],
            'error_rate': self.error_count / max(latest['signals_processed'], 1),
            'monitoring_health': 'healthy' if self.is_running else 'stopped'
        }


class RealTimeDataAdapter:

    """实时数据适配器"""

    def __init__(self):

        self.data_sources = {}
        self.is_running = False

    async def start(self):
        """启动数据适配器"""
        self.is_running = True
        logger.info("RealTimeDataAdapter started")

    async def stop(self):
        """停止数据适配器"""
        self.is_running = False
        logger.info("RealTimeDataAdapter stopped")

    def register_data_source(self, source_id: str, source_config: Dict[str, Any]):
        """注册数据源"""
        self.data_sources[source_id] = source_config
        logger.info(f"Data source registered: {source_id}")

    async def subscribe_symbol(self, source_id: str, symbol: str):
        """订阅交易品种"""
        # 这里实现具体的数据源订阅逻辑
        logger.info(f"Subscribed to {symbol} from {source_id}")

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取市场数据"""
        # 这里实现具体的数据获取逻辑
        # 暂时返回模拟数据
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=np.secrets.uniform(100, 200),
            volume=np.secrets.randint(100, 1000),
            bid=np.secrets.uniform(99, 199),
            ask=np.secrets.uniform(101, 201)
        )


# 全局实例
_real_time_engine = None
_data_adapter = None


def get_real_time_strategy_engine() -> RealTimeStrategyEngine:
    """获取实时策略引擎实例"""
    global _real_time_engine
    if _real_time_engine is None:
        _real_time_engine = RealTimeStrategyEngine()
    return _real_time_engine


def get_real_time_data_adapter() -> RealTimeDataAdapter:
    """获取实时数据适配器实例"""
    global _data_adapter
    if _data_adapter is None:
        _data_adapter = RealTimeDataAdapter()
    return _data_adapter
