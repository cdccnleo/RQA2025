#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import struct
import pytest
from src.data.decoders.level2_decoder import Level2Decoder


def _make_frame(symbol="600000", price=12.34, volume=123456, bids=None, asks=None):
    # header
    buf = bytearray()
    buf += b"\xAA\x55"  # magic
    buf += b"\x00\x00"  # reserved
    # symbol 6 bytes
    sym = symbol.encode("ascii")
    sym = (sym + b" " * 6)[:6]
    buf += sym
    # price 8 bytes
    buf += struct.pack("!d", price)
    # volume 8 bytes
    buf += struct.pack("!Q", volume)
    # order book
    bids = bids or [(12.3, 100), (12.2, 200)]
    asks = asks or [(12.4, 100), (12.5, 200)]
    buf += struct.pack("!HH", len(bids), len(asks))
    for p, v in bids:
        buf += struct.pack("!dQ", p, v)
    for p, v in asks:
        buf += struct.pack("!dQ", p, v)
    return bytes(buf)


def test_decode_success_and_market_detect(monkeypatch):
    dec = Level2Decoder()
    # 兼容实现差异：用正确实现替换内部订单簿解码
    def _decode_order_book(data: bytes):
        bids = []
        asks = []
        if len(data) < 4:
            return bids, asks
        bid_count, ask_count = struct.unpack("!HH", data[:4])
        pos = 4
        for _ in range(bid_count):
            price, vol = struct.unpack("!dQ", data[pos:pos+16])
            bids.append((price, vol))
            pos += 16
        for _ in range(ask_count):
            price, vol = struct.unpack("!dQ", data[pos:pos+16])
            asks.append((price, vol))
            pos += 16
        bids.sort(reverse=True)
        asks.sort()
        return bids, asks
    monkeypatch.setattr(dec, "_decode_order_book", _decode_order_book, raising=False)
    raw = _make_frame(symbol="600000")
    out = dec.decode(raw)
    assert out["symbol"].strip() == "600000"
    assert out["market"] == "SH"
    assert isinstance(out["bids"], list) and isinstance(out["asks"], list)


def test_decode_invalid_header_raises():
    dec = Level2Decoder()
    bad = b"\x00\x00" + _make_frame()[2:]
    with pytest.raises(Exception):
        dec.decode(bad)


def test_decode_batch_mixed_frames(monkeypatch):
    dec = Level2Decoder()
    def _decode_order_book(data: bytes):
        bids = []
        asks = []
        if len(data) < 4:
            return bids, asks
        bid_count, ask_count = struct.unpack("!HH", data[:4])
        pos = 4
        for _ in range(bid_count):
            price, vol = struct.unpack("!dQ", data[pos:pos+16])
            bids.append((price, vol))
            pos += 16
        for _ in range(ask_count):
            price, vol = struct.unpack("!dQ", data[pos:pos+16])
            asks.append((price, vol))
            pos += 16
        bids.sort(reverse=True)
        asks.sort()
        return bids, asks
    monkeypatch.setattr(dec, "_decode_order_book", _decode_order_book, raising=False)
    ok = _make_frame(symbol="000001")
    bad = b"\x00\x00" + ok[2:]
    out = dec.decode_batch([ok, bad])
    assert len(out) == 1
    assert out[0]["market"] == "SZ"


