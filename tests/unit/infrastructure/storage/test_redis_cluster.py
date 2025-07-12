#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import fakeredis
import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.storage.adapters.redis import (
    RedisAdapter,
    RedisClusterAdapter
)

# 统一mock redis.Redis和fakeredis.FakeRedis
@pytest.fixture(autouse=True)
def mock_redis():
    with patch('redis.Redis', MagicMock()), \
         patch('fakeredis.FakeRedis', MagicMock()):
        yield

class TestRedisClusterAdapter(unittest.TestCase):
    """Redis集群适配器测试"""

    def setUp(self):
        self.fake_cluster = FakeRedisCluster()
        self.adapter = RedisAdapter(cluster_mode=True)
        self.adapter.client = self.fake_cluster

    def test_cluster_write_read(self):
        """测试集群模式基础读写"""
        test_data = {"key": "value"}
        path = "cluster/test"

        # 写入测试
        write_result = self.adapter.write(path, test_data)
        self.assertTrue(write_result)

        # 读取验证
        read_data = self.adapter.read(path)
        self.assertEqual(read_data, test_data)

    def test_cluster_nodes(self):
        """测试集群节点访问"""
        # 模拟3节点集群
        nodes = [
            {'host': '127.0.0.1', 'port': 7000},
            {'host': '127.0.0.1', 'port': 7001},
            {'host': '127.0.0.1', 'port': 7002}
        ]
        self.fake_cluster.connection_pool.nodes.add_node(nodes[0])
        self.fake_cluster.connection_pool.nodes.add_node(nodes[1])
        self.fake_cluster.connection_pool.nodes.add_node(nodes[2])

        # 验证节点数量
        self.assertEqual(
            len(self.adapter.client.connection_pool.nodes.nodes),
            3
        )

class TestAShareRedisClusterAdapter(unittest.TestCase):
    """A股Redis集群适配器测试"""

    def setUp(self):
        self.fake_cluster = FakeRedisCluster()
        self.adapter = AShareRedisAdapter(cluster_mode=True)
        self.adapter.client = self.fake_cluster

    def test_cluster_bulk_save(self):
        """测试集群批量存储"""
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

        # 验证数据分布
        keys = self.fake_cluster.keys("ashare:*")
        self.assertEqual(len(keys), 4)  # 2 quotes + timestamp + metadata

    def test_metrics_collection(self):
        """测试监控指标收集"""
        # 模拟操作
        self.adapter.save_quote("600519", {
            "price": 1720.5,
            "volume": 1500,
            "time": "09:30:00",
            "timestamp": 1234567890
        })

        metrics = self.adapter.get_metrics()
        self.assertEqual(metrics['write_count'], 1)
        self.assertTrue(metrics['cluster_mode'])
