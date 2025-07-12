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

class TestEnhancedRedisAdapter(unittest.TestCase):
    """增强Redis适配器测试"""

    def setUp(self):
        # 初始化模拟Redis
        self.fake_redis = fakeredis.FakeStrictRedis()
        self.fake_cluster = MagicMock() # Use MagicMock for FakeRedisCluster

        # 单机模式适配器
        self.single_adapter = RedisAdapter()
        self.single_adapter.client = self.fake_redis

        # 集群模式适配器
        self.cluster_adapter = RedisAdapter(cluster_mode=True)
        self.cluster_adapter.client = self.fake_cluster

        # A股适配器
        self.ashare_adapter = AShareRedisAdapter()
        self.ashare_adapter.client = self.fake_redis

        # 准备测试数据
        self.test_data = {
            "key1": {"field": "value1"},
            "key2": {"field": "value2"},
            "key3": {"field": "value3"}
        }
        for k, v in self.test_data.items():
            self.fake_redis.set(k, json.dumps(v))
            self.fake_cluster.set(k, json.dumps(v))

    def test_bulk_delete_single_mode(self):
        """测试单机模式批量删除"""
        keys = ["key1", "key2"]
        deleted = self.single_adapter.bulk_delete(keys)
        self.assertEqual(deleted, 2)
        self.assertIsNone(self.fake_redis.get("key1"))
        self.assertIsNone(self.fake_redis.get("key2"))
        self.assertIsNotNone(self.fake_redis.get("key3"))

    def test_bulk_delete_cluster_mode(self):
        """测试集群模式批量删除"""
        keys = ["key1", "key2"]
        deleted = self.cluster_adapter.bulk_delete(keys)
        self.assertEqual(deleted, 2)
        self.assertIsNone(self.fake_cluster.get("key1"))
        self.assertIsNone(self.fake_cluster.get("key2"))
        self.assertIsNotNone(self.fake_cluster.get("key3"))

    def test_ashare_bulk_delete(self):
        """测试A股批量删除"""
        # 准备A股数据
        quotes = {
            "600519": {"price": 1720.5, "volume": 1000},
            "000001": {"price": 15.2, "volume": 2000}
        }
        self.ashare_adapter.bulk_save(quotes)

        # 验证删除
        deleted = self.ashare_adapter.bulk_delete_quotes(list(quotes.keys()))
        self.assertEqual(deleted, 2)

        # 验证指标
        metrics = self.ashare_adapter.get_metrics()
        self.assertEqual(metrics['delete_count'], 2)

    def test_pipeline_operations(self):
        """测试流水线操作"""
        pipe = self.single_adapter.pipeline()

        # 添加多个操作
        pipe.set("pipe1", json.dumps({"test": "data1"}))
        pipe.set("pipe2", json.dumps({"test": "data2"}))
        pipe.execute()

        # 验证执行结果
        self.assertIsNotNone(self.fake_redis.get("pipe1"))
        self.assertIsNotNone(self.fake_redis.get("pipe2"))

    def test_ashare_metrics(self):
        """测试A股监控指标"""
        # 执行各种操作
        self.ashare_adapter.save_quote("600519", {"price": 1720.5, "volume": 1000})
        self.ashare_adapter.bulk_save({"000001": {"price": 15.2, "volume": 2000}})
        self.ashare_adapter.bulk_delete_quotes(["600519", "000001"])

        # 验证指标收集
        metrics = self.ashare_adapter.get_metrics()
        self.assertEqual(metrics['write_count'], 2)
        self.assertEqual(metrics['pipeline_ops'], 2)
        self.assertEqual(metrics['delete_count'], 2)

if __name__ == '__main__':
    unittest.main()
