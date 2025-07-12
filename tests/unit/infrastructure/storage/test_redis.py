#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import fakeredis
import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.storage.adapters.redis import (
    RedisAdapter,
    AShareRedisAdapter
)

# 统一mock redis.Redis和fakeredis.FakeRedis
@pytest.fixture(autouse=True)
def mock_redis():
    with patch('redis.Redis', MagicMock()), \
         patch('fakeredis.FakeRedis', MagicMock()):
        yield

class TestRedisAdapter(unittest.TestCase):
    """Redis适配器测试"""

    def setUp(self):
        self.fake_redis = fakeredis.FakeStrictRedis()
        self.adapter = RedisAdapter()
        self.adapter.client = self.fake_redis

    def test_write_read(self):
        """测试基础读写功能"""
        test_data = {"key": "value"}
        path = "test/path"

        # 写入测试
        write_result = self.adapter.write(path, test_data)
        self.assertTrue(write_result)

        # 读取验证
        read_data = self.adapter.read(path)
        self.assertEqual(read_data, test_data)

    def test_invalid_data(self):
        """测试无效数据处理"""
        # 测试无效JSON
        self.fake_redis.set("bad/path", "{invalid}")
        self.assertIsNone(self.adapter.read("bad/path"))

class TestAShareRedisAdapter(unittest.TestCase):
    """A股Redis适配器测试"""

    def setUp(self):
        self.fake_redis = fakeredis.FakeStrictRedis()
        self.adapter = AShareRedisAdapter()
        self.adapter.client = self.fake_redis

    def test_save_quote(self):
        """测试A股行情存储"""
        test_data = {
            "price": 1720.5,
            "volume": 1500,
            "time": "09:30:00",
            "limit_status": "up",
            "timestamp": 1234567890
        }

        # 测试SH市场
        result = self.adapter.save_quote("600519", test_data)
        self.assertTrue(result)

        # 验证数据字段
        key = "ashare:quotes:600519"
        self.assertEqual(
            self.fake_redis.hget(key, "price"),
            "1720.5"
        )
        self.assertEqual(
            self.fake_redis.hget(key, "status"),
            "up"
        )

        # 验证时间索引
        self.assertEqual(
            self.fake_redis.zscore("ashare:timestamps", "600519"),
            1234567890
        )

    def test_bulk_save(self):
        """测试批量存储"""
        quotes = {
            "600519": {
                "price": 1720.5,
                "volume": 1500,
                "time": "09:30:00",
                "timestamp": 1234567890
            },
            "000001": {
                "price": 15.2,
                "volume": 2000,
                "time": "09:30:01",
                "timestamp": 1234567891
            }
        }

        result = self.adapter.bulk_save(quotes)
        self.assertTrue(result)

        # 验证数据数量
        self.assertEqual(
            len(self.fake_redis.keys("ashare:quotes:*")),
            2
        )

        # 验证时间索引
        self.assertEqual(
            self.fake_redis.zcard("ashare:timestamps"),
            2
        )

    def test_invalid_symbol(self):
        """测试无效股票代码"""
        with self.assertRaises(ValueError):
            self.adapter.save_quote("invalid", {})
