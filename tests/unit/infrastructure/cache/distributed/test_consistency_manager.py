#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分布式缓存一致性管理器测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.infrastructure.cache.distributed.consistency_manager import (
    ConsistencyManager,
    ConsistencyConfig,
    ConflictResolutionStrategy,
    DataEntry,
    VectorClock,
    VersionInfo
)
from src.infrastructure.cache.interfaces import ConsistencyLevel


class TestVectorClock:
    """测试向量时钟"""

    def test_init(self):
        """测试初始化"""
        clock = VectorClock("node1")
        assert clock.node_id == "node1"
        assert isinstance(clock.clock, dict)
        assert clock.clock["node1"] == 0

    def test_increment(self):
        """测试递增"""
        clock = VectorClock("node1")
        clock.increment()
        assert clock.clock["node1"] == 1

        clock.increment()
        assert clock.clock["node1"] == 2

    def test_update(self):
        """测试更新"""
        clock1 = VectorClock("node1")
        clock2 = VectorClock("node2")

        # 更新clock1
        clock1.increment()
        clock1.increment()

        # clock2接收clock1的更新（update会递增本地时钟）
        clock2.update(clock1.clock)

        assert clock2.clock["node1"] == 2
        assert clock2.clock["node2"] == 1  # 本地时钟被递增

    def test_compare(self):
        """测试比较"""
        clock1 = VectorClock("node1")
        clock2 = VectorClock("node2")

        # 初始状态相等
        assert clock1.compare(clock2.clock) == "equal"

        # clock1递增后，clock1在node1上是"after"（因为clock1有更新）
        clock1.increment()
        assert clock1.compare(clock2.clock) == "after"

        # clock2也递增后，它们的更新是并发的
        clock2.increment()
        assert clock1.compare(clock2.clock) == "concurrent"

    def test_get_clock(self):
        """测试获取时钟"""
        clock = VectorClock("node1")
        clock.increment()

        clock_dict = clock.get_clock()
        assert clock_dict == {"node1": 1}
        assert clock_dict is not clock.clock  # 应该是副本


class TestConsistencyManager:
    """测试一致性管理器"""

    def setup_method(self):
        """测试前准备"""
        self.config = ConsistencyConfig()
        self.manager = ConsistencyManager("test_node", self.config)

    def test_init(self):
        """测试初始化"""
        assert self.manager.node_id == "test_node"
        assert self.manager.config == self.config
        assert isinstance(self.manager.vector_clock, VectorClock)
        assert isinstance(self.manager.cache_nodes, dict)
        assert self.manager.is_running is False

    def test_register_cache_node(self):
        """测试注册缓存节点"""
        mock_cache = Mock()
        self.manager.register_cache_node("node1", mock_cache)

        assert "node1" in self.manager.cache_nodes
        assert self.manager.cache_nodes["node1"] == mock_cache
        assert self.manager.node_status["node1"] is True

    def test_register_duplicate_node(self):
        """测试注册重复节点"""
        mock_cache1 = Mock()
        mock_cache2 = Mock()

        # 第一次注册
        self.manager.register_cache_node("node1", mock_cache1)
        assert self.manager.cache_nodes["node1"] == mock_cache1

        # 第二次注册会覆盖（没有重复检查）
        self.manager.register_cache_node("node1", mock_cache2)
        assert self.manager.cache_nodes["node1"] == mock_cache2

    def test_unregister_cache_node(self):
        """测试注销缓存节点"""
        mock_cache = Mock()
        self.manager.register_cache_node("node1", mock_cache)

        # 注销存在的节点
        self.manager.unregister_cache_node("node1")
        assert "node1" not in self.manager.cache_nodes
        assert "node1" not in self.manager.node_status

        # 注销不存在的节点（不会抛出异常）
        self.manager.unregister_cache_node("nonexistent")

    def test_eventual_consistent_read_no_nodes(self):
        """测试最终一致性读取（无节点）"""
        result = self.manager.consistent_read("test_key")
        assert result is None

    def test_eventual_consistent_read_with_node(self):
        """测试最终一致性读取（有节点）"""
        # 注册一个mock节点
        mock_cache = Mock()
        mock_cache.get = Mock(return_value="test_value")
        self.manager.register_cache_node("node1", mock_cache)

        result = self.manager.consistent_read("test_key")
        assert result == "test_value"
        mock_cache.get.assert_called_once_with("test_key")

    def test_eventual_consistent_write_no_nodes(self):
        """测试最终一致性写入（无节点）"""
        result = self.manager.consistent_write("test_key", "test_value")
        assert result is False

    def test_eventual_consistent_write_with_node(self):
        """测试最终一致性写入（有节点）"""
        # 注册一个mock节点
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        self.manager.register_cache_node("node1", mock_cache)

        result = self.manager.consistent_write("test_key", "test_value")
        assert result is True
        # 验证set方法被调用，但参数是DataEntry对象
        assert mock_cache.set.called
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "test_key"  # key
        # 第二个参数是DataEntry对象
        assert hasattr(call_args[0][1], 'key')

    def test_get_consistency_metrics(self):
        """测试获取一致性指标"""
        metrics = self.manager.get_consistency_metrics()

        assert isinstance(metrics, dict)
        assert "total_operations" in metrics
        assert "consistent_reads" in metrics
        assert "inconsistent_reads" in metrics
        assert "consistency_ratio" in metrics
        assert "conflicts_detected" in metrics
        assert "active_nodes" in metrics
        assert "total_nodes" in metrics
        assert metrics["total_operations"] == 0
        assert metrics["active_nodes"] == 0
        assert metrics["total_nodes"] == 0

    def test_set_custom_conflict_resolver(self):
        """测试设置自定义冲突解决器"""
        def custom_resolver(conflicts):
            return conflicts[0]  # 返回第一个

        self.manager.set_custom_conflict_resolver(custom_resolver)
        assert self.manager.custom_conflict_resolver == custom_resolver

    def test_cleanup(self):
        """测试清理"""
        # 注册一些节点
        self.manager.register_cache_node("node1", Mock())
        self.manager.register_cache_node("node2", Mock())

        # 设置一些状态
        self.manager.is_running = True

        self.manager.cleanup()

        assert len(self.manager.cache_nodes) == 0
        assert self.manager.is_running is False

    def test_strong_consistent_read_no_quorum(self):
        """测试强一致性读取（无仲裁）"""
        # 配置需要2个节点的读仲裁，但只有一个节点
        self.config.read_quorum = 2
        self.config.level = ConsistencyLevel.STRONG

        mock_cache = Mock()
        mock_cache.get = Mock(return_value="value")
        self.manager.register_cache_node("node1", mock_cache)

        result = self.manager.consistent_read("key")
        # 强一致性读取需要仲裁，这里只有一个节点，应该返回None
        assert result is None

    def test_session_consistent_read(self):
        """测试会话一致性读取"""
        mock_cache = Mock()
        mock_cache.get = Mock(return_value="value")
        self.manager.register_cache_node("node1", mock_cache)

        # 配置为会话一致性
        self.config.level = ConsistencyLevel.SESSION

        # 测试会话一致性
        result = self.manager.consistent_read("key", session_id="session1")
        assert result == "value"

    def test_start_stop_consistency_manager(self):
        """测试启动和停止一致性管理器"""
        # 初始状态
        assert self.manager.is_running is False

        # 启动
        self.manager.start_consistency_manager()
        assert self.manager.is_running is True

        # 停止
        self.manager.stop_consistency_manager()
        assert self.manager.is_running is False

    def test_read_repair_functionality(self):
        """测试读修复功能"""
        # 这个测试比较复杂，需要多个节点和版本冲突
        # 这里只是基本结构测试
        self.config.enable_read_repair = True
        self.config.level = ConsistencyLevel.STRONG

        # 创建真正的VersionInfo和DataEntry对象
        version_info1 = VersionInfo(version=1, timestamp=1000.0, node_id="node1", checksum="checksum1")
        version_info2 = VersionInfo(version=2, timestamp=2000.0, node_id="node2", checksum="checksum2")

        data_entry1 = DataEntry("key", "value1", version_info1)
        data_entry2 = DataEntry("key", "value2", version_info2)

        mock_cache1 = Mock()
        mock_cache1.get = Mock(return_value=data_entry1)

        mock_cache2 = Mock()
        mock_cache2.get = Mock(return_value=data_entry2)
        mock_cache2.set = Mock(return_value=True)

        self.manager.register_cache_node("node1", mock_cache1)
        self.manager.register_cache_node("node2", mock_cache2)

        # 执行强一致性读取，应该触发读修复
        result = self.manager.consistent_read("key")

        # 验证读修复被调用（这里是简化测试）
        # 只要返回其中一个值就可以，证明方法能正常执行
        assert result in ["value1", "value2"]


class TestDataEntry:
    """测试数据条目"""

    def test_init(self):
        """测试初始化"""
        version_info = {"node1": 1}
        entry = DataEntry("test_key", "test_value", version_info)
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.version_info == version_info

    def test_init_without_version_vector(self):
        """测试不带版本向量初始化"""
        version_info = {}
        entry = DataEntry("test_key", "test_value", version_info)
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.version_info == version_info


class TestConsistencyConfig:
    """测试一致性配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = ConsistencyConfig()
        assert config.level.value == "eventual"
        assert config.conflict_resolution == ConflictResolutionStrategy.LAST_WRITE_WINS
        assert config.sync_timeout == 5.0
        assert config.max_retries == 3
        assert config.read_quorum == 1
        assert config.write_quorum == 1
        assert config.enable_version_vector is True
        assert config.enable_read_repair is True
        assert config.anti_entropy_interval == 60.0
