#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from unittest.mock import MagicMock
from src.engine.realtime_engine import (
    RealTimeEngine,
    RingBuffer,
    ChinaLevel2Decoder
)

class TestRingBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = RingBuffer(size=100)

    def test_basic_io(self):
        """测试基本读写功能"""
        test_data = b"test_data"
        self.assertTrue(self.buffer.put(test_data))
        self.assertEqual(self.buffer.get(), test_data)

    def test_buffer_full(self):
        """测试缓冲区满情况"""
        large_data = b"x" * 120
        self.assertFalse(self.buffer.put(large_data))

    def test_zero_copy(self):
        """测试零拷贝特性"""
        data = np.arange(10, dtype=np.uint8).tobytes()
        self.buffer.put(data)
        out = self.buffer.get()
        self.assertEqual(out, data)

class TestChinaLevel2Decoder(unittest.TestCase):
    def test_decode(self):
        """测试A股Level2解码"""
        # 构造测试数据 (104字节)
        data = bytearray(104)
        # 股票代码
        data[0:6] = b"600519"
        # 最新价 (182.56元)
        data[6:10] = (18256).to_bytes(4, 'little')
        # 成交量 (1000股)
        data[10:14] = (1000).to_bytes(4, 'little')
        # 买一价 (182.50元)
        data[14:18] = (18250).to_bytes(4, 'little')
        # 买一量 (500股)
        data[18:22] = (500).to_bytes(4, 'little')

        decoded = ChinaLevel2Decoder.decode(bytes(data))
        self.assertEqual(decoded["symbol"], "600519")
        self.assertAlmostEqual(decoded["price"], 182.56)
        self.assertEqual(decoded["volume"], 1000)
        self.assertAlmostEqual(decoded["order_book"]["bids"][0][0], 182.50)
        self.assertEqual(decoded["order_book"]["bids"][0][1], 500)

class TestRealTimeEngine(unittest.TestCase):
    def setUp(self):
        self.config = {"market": "A"}
        self.engine = RealTimeEngine(self.config)
        self.mock_handler = MagicMock()
        self.engine.register_handler("tick", self.mock_handler)

    def test_lifecycle(self):
        """测试引擎启动停止"""
        self.engine.start()
        self.assertTrue(self.engine.running)
        self.engine.stop()
        self.assertFalse(self.engine.running)

    def test_a_share_processing(self):
        """测试A股特有处理逻辑"""
        # 构造Level2数据
        data = bytearray(104)
        data[0:6] = b"600519"
        data[6:10] = (18256).to_bytes(4, 'little')  # 182.56元

        self.engine.start()
        self.engine.feed_data(bytes(data))
        self.engine.stop()

        # 验证处理器被调用且包含涨跌停状态
        args, _ = self.mock_handler.call_args
        self.assertEqual(args[0]["symbol"], "600519")
        self.assertIn("limit_status", args[0])

    def test_circuit_breaker(self):
        """测试熔断状态传递"""
        self.engine.circuit_breaker = True
        data = bytearray(104)
        data[0:6] = b"600519"

        self.engine.start()
        self.engine.feed_data(bytes(data))
        self.engine.stop()

        args, _ = self.mock_handler.call_args
        self.assertTrue(args[0]["circuit_breaker"])

if __name__ == "__main__":
    unittest.main()
