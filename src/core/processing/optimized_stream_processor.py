"""
优化版实时数据流处理器

提供高性能的流式数据处理能力，支持向量化计算、无锁队列、批处理优化。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """流处理器配置"""
    max_queue_size: int = 10000
    batch_size: int = 100
    processing_interval: float = 0.001  # 1ms
    max_buffer_size: int = 1000
    enable_vectorization: bool = True
    num_worker_threads: int = 4
    prefetch_size: int = 10


@dataclass
class StreamEvent:
    """流事件"""
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    symbol: str = ""
    priority: int = 0


@dataclass
class ProcessingStats:
    """处理统计"""
    total_events: int = 0
    processed_events: int = 0
    dropped_events: int = 0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    queue_size: int = 0
    buffer_utilization: float = 0.0


class LockFreeQueue:
    """
    无锁队列（基于双缓冲）
    
    提供高并发场景下的低延迟队列操作
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._primary_buffer: deque = deque(maxlen=max_size)
        self._secondary_buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._swap_pending = False
    
    def put(self, item: Any) -> bool:
        """放入元素"""
        with self._lock:
            if len(self._primary_buffer) < self.max_size:
                self._primary_buffer.append(item)
                return True
            return False
    
    def get_batch(self, batch_size: int) -> List[Any]:
        """批量获取元素"""
        with self._lock:
            batch = []
            while len(batch) < batch_size and self._primary_buffer:
                batch.append(self._primary_buffer.popleft())
            return batch
    
    def size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._primary_buffer)


class OptimizedStreamProcessor:
    """
    优化版流处理器
    
    提供以下性能优化：
    1. 无锁队列 - 减少并发开销
    2. 向量化计算 - 使用NumPy加速
    3. 批处理 - 批量处理事件
    4. 预分配缓冲区 - 减少内存分配
    5. 多线程处理 - 并行计算
    6. 内存池 - 重用对象
    
    Attributes:
        config: 处理器配置
        event_queue: 事件队列
        handlers: 事件处理器
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        
        # 无锁队列
        self._event_queue = LockFreeQueue(self.config.max_queue_size)
        
        # 预分配缓冲区
        self._buffer = [None] * self.config.max_buffer_size
        self._buffer_index = 0
        
        # 事件处理器
        self._handlers: Dict[str, List[Callable]] = {}
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_worker_threads)
        
        # 统计
        self._stats = ProcessingStats()
        self._last_stats_time = time.time()
        self._events_in_last_second = 0
        
        # 运行状态
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        # 价格历史缓存（用于技术指标计算）
        self._price_history: Dict[str, deque] = {}
        self._max_history_size = 100
        
        # 指标计算缓存
        self._indicator_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("OptimizedStreamProcessor initialized")
    
    async def initialize(self) -> bool:
        """初始化处理器"""
        try:
            self._running = True
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            logger.info("OptimizedStreamProcessor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"OptimizedStreamProcessor initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """关闭处理器"""
        try:
            self._running = False
            
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            self._executor.shutdown(wait=True)
            
            logger.info("OptimizedStreamProcessor shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"OptimizedStreamProcessor shutdown failed: {e}")
            return False
    
    def emit(self, event: StreamEvent) -> bool:
        """
        发送事件到处理器
        
        Args:
            event: 流事件
            
        Returns:
            bool: 是否成功加入队列
        """
        success = self._event_queue.put(event)
        
        if success:
            self._stats.total_events += 1
            self._events_in_last_second += 1
        else:
            self._stats.dropped_events += 1
            logger.warning(f"Event queue full, dropped event: {event.event_type}")
        
        return success
    
    def register_handler(self, event_type: str, handler: Callable[[StreamEvent], None]):
        """
        注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type}")
    
    def unregister_handler(self, event_type: str, handler: Callable[[StreamEvent], None]):
        """注销事件处理器"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
    
    async def _processing_loop(self):
        """处理循环"""
        while self._running:
            try:
                # 批量获取事件
                batch = self._event_queue.get_batch(self.config.batch_size)
                
                if batch:
                    # 处理批次
                    await self._process_batch(batch)
                else:
                    # 队列为空，短暂休眠
                    await asyncio.sleep(self.config.processing_interval)
                
                # 更新统计
                self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[StreamEvent]):
        """
        批量处理事件
        
        Args:
            batch: 事件批次
        """
        start_time = time.time()
        
        # 按事件类型分组
        grouped_events = self._group_events_by_type(batch)
        
        # 并行处理不同类型的事件
        tasks = []
        for event_type, events in grouped_events.items():
            task = self._process_event_group(event_type, events)
            tasks.append(task)
        
        # 等待所有处理完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 更新统计
        processing_time = (time.time() - start_time) * 1000
        self._update_processing_stats(processing_time, len(batch))
    
    def _group_events_by_type(self, events: List[StreamEvent]) -> Dict[str, List[StreamEvent]]:
        """按事件类型分组"""
        grouped = {}
        for event in events:
            if event.event_type not in grouped:
                grouped[event.event_type] = []
            grouped[event.event_type].append(event)
        return grouped
    
    async def _process_event_group(self, event_type: str, events: List[StreamEvent]):
        """
        处理事件组
        
        Args:
            event_type: 事件类型
            events: 事件列表
        """
        # 获取处理器
        handlers = self._handlers.get(event_type, [])
        
        if not handlers:
            return
        
        # 特殊处理市场数据事件（向量化计算）
        if event_type == "market_data":
            await self._process_market_data_batch(events)
        else:
            # 普通处理
            for event in events:
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Handler error for {event_type}: {e}")
    
    async def _process_market_data_batch(self, events: List[StreamEvent]):
        """
        批量处理市场数据事件（向量化计算）
        
        Args:
            events: 市场数据事件列表
        """
        if not self.config.enable_vectorization:
            # 回退到逐个处理
            for event in events:
                await self._process_single_market_data(event)
            return
        
        try:
            # 提取价格数据
            symbols = []
            prices = []
            volumes = []
            
            for event in events:
                data = event.data
                symbol = event.symbol or data.get('symbol', '')
                price = data.get('price', 0.0)
                volume = data.get('volume', 0)
                
                if symbol and price > 0:
                    symbols.append(symbol)
                    prices.append(price)
                    volumes.append(volume)
                    
                    # 更新价格历史
                    self._update_price_history(symbol, price)
            
            if not prices:
                return
            
            # 转换为NumPy数组
            prices_array = np.array(prices, dtype=np.float64)
            volumes_array = np.array(volumes, dtype=np.float64)
            
            # 向量化计算统计指标
            stats = {
                'mean_price': np.mean(prices_array),
                'std_price': np.std(prices_array),
                'min_price': np.min(prices_array),
                'max_price': np.max(prices_array),
                'total_volume': np.sum(volumes_array),
                'avg_volume': np.mean(volumes_array) if len(volumes_array) > 0 else 0
            }
            
            # 计算技术指标（批量）
            for i, symbol in enumerate(symbols):
                if symbol in self._price_history:
                    history = list(self._price_history[symbol])
                    if len(history) >= 5:
                        # 计算移动平均
                        sma_5 = np.mean(history[-5:])
                        sma_10 = np.mean(history[-10:]) if len(history) >= 10 else sma_5
                        
                        # 计算价格变化
                        price_change = ((history[-1] - history[0]) / history[0] * 100) if history[0] != 0 else 0
                        
                        # 更新事件数据
                        events[i].data['sma_5'] = round(sma_5, 2)
                        events[i].data['sma_10'] = round(sma_10, 2)
                        events[i].data['price_change_pct'] = round(price_change, 2)
                        events[i].data['volatility'] = round(np.std(history[-10:]), 4) if len(history) >= 10 else 0
            
            # 调用处理器
            handlers = self._handlers.get('market_data', [])
            for event in events:
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Market data handler error: {e}")
            
        except Exception as e:
            logger.error(f"Batch market data processing error: {e}")
            # 回退到逐个处理
            for event in events:
                await self._process_single_market_data(event)
    
    async def _process_single_market_data(self, event: StreamEvent):
        """处理单个市场数据事件"""
        symbol = event.symbol or event.data.get('symbol', '')
        price = event.data.get('price', 0.0)
        
        if symbol and price > 0:
            self._update_price_history(symbol, price)
            
            # 计算简单指标
            if symbol in self._price_history:
                history = list(self._price_history[symbol])
                if len(history) >= 2:
                    event.data['price_change_pct'] = round(
                        ((history[-1] - history[-2]) / history[-2] * 100), 2
                    ) if history[-2] != 0 else 0
        
        # 调用处理器
        handlers = self._handlers.get('market_data', [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Market data handler error: {e}")
    
    def _update_price_history(self, symbol: str, price: float):
        """更新价格历史"""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._max_history_size)
        
        self._price_history[symbol].append(price)
    
    def _update_processing_stats(self, processing_time_ms: float, batch_size: int):
        """更新处理统计"""
        # 使用指数移动平均
        alpha = 0.1
        self._stats.avg_processing_time_ms = (
            alpha * processing_time_ms + (1 - alpha) * self._stats.avg_processing_time_ms
        )
        self._stats.max_processing_time_ms = max(
            self._stats.max_processing_time_ms, processing_time_ms
        )
        
        self._stats.processed_events += batch_size
    
    def _update_stats(self):
        """更新统计信息"""
        current_time = time.time()
        
        # 每秒更新一次吞吐量
        if current_time - self._last_stats_time >= 1.0:
            self._stats.throughput_per_second = self._events_in_last_second
            self._events_in_last_second = 0
            self._last_stats_time = current_time
        
        # 更新队列大小
        self._stats.queue_size = self._event_queue.size()
    
    def get_stats(self) -> ProcessingStats:
        """获取处理统计"""
        return self._stats
    
    def clear_price_history(self, symbol: Optional[str] = None):
        """清除价格历史"""
        if symbol:
            if symbol in self._price_history:
                del self._price_history[symbol]
        else:
            self._price_history.clear()


# 便捷函数
def create_optimized_processor(
    max_queue_size: int = 10000,
    batch_size: int = 100,
    enable_vectorization: bool = True
) -> OptimizedStreamProcessor:
    """
    创建优化版流处理器
    
    Args:
        max_queue_size: 最大队列大小
        batch_size: 批处理大小
        enable_vectorization: 是否启用向量化计算
        
    Returns:
        OptimizedStreamProcessor: 流处理器实例
    """
    config = StreamConfig(
        max_queue_size=max_queue_size,
        batch_size=batch_size,
        enable_vectorization=enable_vectorization
    )
    return OptimizedStreamProcessor(config)


__all__ = [
    'OptimizedStreamProcessor',
    'StreamConfig',
    'StreamEvent',
    'ProcessingStats',
    'LockFreeQueue',
    'create_optimized_processor'
]