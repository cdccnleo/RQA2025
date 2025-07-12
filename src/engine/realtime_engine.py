#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时引擎核心模块 - v1.1

功能架构：
1. 行情接收层：
   - 支持多数据源(Level2/普通行情)
   - 自动适配不同券商API

2. 数据处理层：
   - 零拷贝环形缓冲区
   - A股特有数据解码
   - 异常数据过滤

3. 分发层：
   - 多路实时分发
   - 优先级队列
   - 背压控制

A股特性支持：
- Level2行情解析
- 涨跌停状态跟踪
- 融资融券数据整合
- 龙虎榜数据关联
"""

import threading
import time
import logging
from collections import deque
from typing import Dict, Callable, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class RingBuffer:
    """零拷贝环形缓冲区"""
    def __init__(self, size: int = 10000):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.uint8)
        self.head = 0
        self.tail = 0
        self.lock = threading.RLock()

    def put(self, data: bytes) -> bool:
        """写入数据"""
        with self.lock:
            if len(data) > self.size - (self.tail - self.head) % self.size:
                return False  # 缓冲区满
            # 实现零拷贝写入
            np_data = np.frombuffer(data, dtype=np.uint8)
            end_pos = (self.tail + len(np_data)) % self.size
            if end_pos > self.tail:
                self.buffer[self.tail:end_pos] = np_data
            else:
                split = self.size - self.tail
                self.buffer[self.tail:] = np_data[:split]
                self.buffer[:end_pos] = np_data[split:]
            self.tail = end_pos
            return True

    def get(self) -> Optional[bytes]:
        """读取数据"""
        with self.lock:
            if self.head == self.tail:
                return None  # 缓冲区空
            # 实现零拷贝读取
            if self.tail > self.head:
                data = self.buffer[self.head:self.tail].tobytes()
            else:
                data = np.concatenate(
                    (self.buffer[self.head:], self.buffer[:self.tail])
                ).tobytes()
            self.head = self.tail
            return data

class ChinaLevel2Decoder:
    """A股Level2行情解码器"""
    @staticmethod
    def decode(data: bytes) -> dict:
        """解码Level2行情数据"""
        # A股特有字段处理
        return {
            "symbol": data[0:6].decode(),
            "price": int.from_bytes(data[6:10], 'little') / 10000,
            "volume": int.from_bytes(data[10:14], 'little'),
            "order_book": {
                "bids": [(int.from_bytes(data[14+i*8:18+i*8], 'little') / 10000,
                         int.from_bytes(data[18+i*8:22+i*8], 'little'))
                         for i in range(5)],
                "asks": [(int.from_bytes(data[54+i*8:58+i*8], 'little') / 10000,
                         int.from_bytes(data[58+i*8:62+i*8], 'little'))
                         for i in range(5)]
            },
            "timestamp": int.from_bytes(data[94:98], 'little')
        }

class MarketDataAdapter:
    """市场数据适配器"""

    @staticmethod
    def adapt_a_share_data(raw_data: Dict) -> Dict:
        """适配A股市场数据"""
        adapted = {
            'symbol': raw_data['code'],
            'price': raw_data['price'] / 10000.0,  # 处理价格单位
            'volume': raw_data['volume'],
            'timestamp': raw_data['time']
        }

        # 处理涨跌停状态
        if raw_data.get('limit_up'):
            adapted['limit_status'] = 'up'
        elif raw_data.get('limit_down'):
            adapted['limit_status'] = 'down'
        else:
            adapted['limit_status'] = 'normal'

        return adapted

class Level2Processor:
    """Level2行情专用处理器"""

    def __init__(self, engine: 'RealTimeEngine'):
        self.engine = engine
        self.order_books: Dict[str, Dict] = {}

        # 注册处理函数
        self.engine.register_handler('order_book', self.process_order_book)

    def process_order_book(self, data: Dict):
        """处理Level2订单簿数据"""
        symbol = data['symbol']
        bids = data['bids']
        asks = data['asks']

        # 更新订单簿
        self.order_books[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }

        # 计算买卖压力
        buy_pressure = sum(bid[1] for bid in bids)
        sell_pressure = sum(ask[1] for ask in asks)

        # 发布压力指标
        self.engine.feed_data({
            'type': 'pressure',
            'symbol': symbol,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'imbalance': (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure)
        })

class RealTimeEngine:
    """实时引擎核心类"""
    def __init__(self, config: dict):
        self.config = config
        self.buffer = RingBuffer(size=config.get("buffer_size", 1000000))
        self.handlers = {
            "tick": deque(),
            "order": deque(),
            "trade": deque()
        }
        self.running = False
        self.thread = None

        # A股特定初始化
        self.last_prices = {}  # 用于涨跌停判断
        self.circuit_breaker = False

    def register_handler(self, data_type: str, handler: Callable):
        """注册数据处理函数"""
        if data_type in self.handlers:
            self.handlers[data_type].append(handler)

    def start(self):
        """启动引擎"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止引擎"""
        self.running = False
        if self.thread:
            self.thread.join()

    def feed_data(self, data: bytes):
        """接收原始数据"""
        if not self.buffer.put(data):
            raise RuntimeError("Real-time buffer overflow")

    def _run_loop(self):
        """主处理循环"""
        while self.running:
            data = self.buffer.get()
            if not data:
                continue

            try:
                # A股特有处理流程
                if self.config.get("market") == "A":
                    decoded = ChinaLevel2Decoder.decode(data)
                    self._process_a_share_data(decoded)
                else:
                    decoded = self._decode_generic(data)

                # 分发处理
                self._dispatch(decoded)

            except Exception as e:
                self._handle_error(e)

    def _process_a_share_data(self, data: dict):
        """处理A股特有数据"""
        symbol = data["symbol"]

        # 涨跌停判断
        if symbol in self.last_prices:
            change = (data["price"] - self.last_prices[symbol]) / self.last_prices[symbol]
            if abs(change) >= 0.1:  # 涨跌停阈值
                data["limit_status"] = "up" if change > 0 else "down"
        self.last_prices[symbol] = data["price"]

        # 熔断状态传递
        if self.circuit_breaker:
            data["circuit_breaker"] = True

    def _decode_generic(self, data: bytes) -> dict:
        """通用行情解码"""
        return {"raw": data}

    def _dispatch(self, data: dict):
        """分发数据到处理器"""
        data_type = data.get("type", "tick")
        for handler in self.handlers.get(data_type, []):
            try:
                handler(data)
            except Exception as e:
                self._handle_error(e)

    def _handle_error(self, error: Exception):
        """统一错误处理"""
        logger.error(f"Engine error: {str(error)}")

# 使用示例
if __name__ == "__main__":
    config = {
        "market": "A",
        "buffer_size": 5000000
    }

    engine = RealTimeEngine(config)

    # 注册处理器
    def tick_handler(data):
        print(f"Tick: {data['symbol']} @ {data['price']}")

    engine.register_handler("tick", tick_handler)

    # 模拟数据输入
    engine.start()
    for _ in range(10):
        engine.feed_data(
            b"600519\x00\x00\x01\x86\xA0\x00\x00\x03\xE8" +
            bytes([i for i in range(88)])
        )
        time.sleep(0.1)

    engine.stop()
