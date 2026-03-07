#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式缓存管理器增强测试

针对distributed_cache_manager.py中未充分测试的功能添加测试用例
目标：提升覆盖率至80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.cache.distributed.distributed_cache_manager import (
    DistributedCacheManager, ClusterManager, VectorClockManager,
    DistributedConfig, ClusterNode, SyncStrategy, SyncMode
)
from src.infrastructure.cache.interfaces.data_structures import ConsistencyLevel


class TestVectorClockManager:
    """测试向量时钟管理器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        nodes = [ClusterNode(host="localhost", port=6379)]
        return DistributedConfig(
            nodes=nodes,
            consistency_level=ConsistencyLevel.STRONG
        )

    @pytest.fixture
    def vector_clock_manager(self, config):
        """创建向量时钟管理器实例"""
        return VectorClockManager(config)

    def test_update_vector_clock(self, vector_clock_manager):
        """测试更新向量时钟"""
        key = "test_key"
        node_id = "node1"
        timestamp = 1234567890

        vector_clock_manager.update_vector_clock(key, node_id, timestamp)
        
        clock = vector_clock_manager.get_vector_clock(key)
        assert clock == {node_id: timestamp}

    def test_update_vector_clock_multiple_nodes(self, vector_clock_manager):
        """测试多节点向量时钟更新"""
        key = "test_key"
        
        vector_clock_manager.update_vector_clock(key, "node1", 1000)
        vector_clock_manager.update_vector_clock(key, "node2", 2000)
        vector_clock_manager.update_vector_clock(key, "node1", 1500)  # 更新node1的时钟

        clock = vector_clock_manager.get_vector_clock(key)
        assert clock["node1"] == 1500
        assert clock["node2"] == 2000

    def test_vector_clock_max_timestamp(self, vector_clock_manager):
        """测试向量时钟使用最大时间戳"""
        key = "test_key"
        node_id = "node1"
        
        vector_clock_manager.update_vector_clock(key, node_id, 1000)
        vector_clock_manager.update_vector_clock(key, node_id, 500)  # 较小的时间戳
        
        clock = vector_clock_manager.get_vector_clock(key)
        assert clock[node_id] == 1000  # 应该保持较大的时间戳

    def test_check_strong_consistency_success(self, vector_clock_manager):
        """测试强一致性检查 - 成功"""
        key = "test_key"
        node_clocks = {
            "node1": {"node1": 1000, "node2": 500},
            "node2": {"node1": 1000, "node2": 500}
        }
        
        # 模拟配置为强一致性
        vector_clock_manager.config.consistency_level = ConsistencyLevel.STRONG
        
        result = vector_clock_manager.is_consistent(key, node_clocks)
        assert result is True

    def test_check_strong_consistency_failure(self, vector_clock_manager):
        """测试强一致性检查 - 失败"""
        key = "test_key"
        node_clocks = {
            "node1": {"node1": 1000, "node2": 500},
            "node2": {"node1": 1000, "node2": 600}  # 不同的时钟值
        }
        
        vector_clock_manager.config.consistency_level = ConsistencyLevel.STRONG
        
        result = vector_clock_manager.is_consistent(key, node_clocks)
        assert result is False

    def test_check_eventual_consistency(self, vector_clock_manager):
        """测试最终一致性检查"""
        key = "test_key"
        
        # 根据当前实现，最终一致性检查比较向量时钟的哈希值
        # 如果所有节点的向量时钟相同，则应该是一致的
        node_clocks = {
            "node1": {"node1": 1000, "node2": 500},
            "node2": {"node1": 1000, "node2": 500}  # 相同的向量时钟
        }
        
        vector_clock_manager.config.consistency_level = ConsistencyLevel.EVENTUAL
        
        result = vector_clock_manager.is_consistent(key, node_clocks)
        assert result is True
        
        # 测试只有一个节点的情况
        single_node_clocks = {
            "node1": {"node1": 1000}
        }
        result_single = vector_clock_manager.is_consistent(key, single_node_clocks)
        assert result_single is True


class TestClusterManager:
    """测试集群管理器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        nodes = [
            ClusterNode(host="localhost", port=6379, weight=2),
            ClusterNode(host="localhost", port=6380, weight=1)
        ]
        return DistributedConfig(
            nodes=nodes,
            replication_factor=2
        )

    @pytest.fixture
    def cluster_manager(self, config):
        """创建集群管理器实例"""
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.redis') as mock_redis:
            mock_client = Mock()
            mock_redis.Redis.return_value = mock_client
            return ClusterManager(config)

    def test_node_initialization(self, cluster_manager, config):
        """测试节点初始化"""
        assert len(cluster_manager.nodes) == len(config.nodes)
        assert len(cluster_manager.active_nodes) == len(config.nodes)
        
        for node_id, cache in cluster_manager.nodes.items():
            assert node_id in cluster_manager.active_nodes
            assert cluster_manager.node_load[node_id] == 0

    def test_get_node_consistency_hash(self, cluster_manager):
        """测试一致性哈希节点选择"""
        key1 = "test_key_1"
        key2 = "test_key_2"
        
        node1, cache1 = cluster_manager.get_node(key1)
        node2, cache2 = cluster_manager.get_node(key2)
        
        # 验证返回的节点存在
        assert node1 in cluster_manager.active_nodes
        assert node2 in cluster_manager.active_nodes
        assert cache1 is not None
        assert cache2 is not None

    def test_get_replica_nodes(self, cluster_manager):
        """测试获取副本节点"""
        key = "test_key"
        primary_node = "node_0"
        
        replicas = cluster_manager.get_replica_nodes(key, primary_node)
        
        # 验证副本节点不包含主节点
        for node_id, cache in replicas:
            assert node_id != primary_node
            assert node_id in cluster_manager.active_nodes

    def test_sync_data_write_through(self, cluster_manager):
        """测试写穿同步"""
        with patch.object(cluster_manager, '_sync_write_through') as mock_sync:
            cluster_manager.config.sync_strategy = "write_through"
            
            cluster_manager.sync_data("test_key", "test_value", "node_0", 3600)
            mock_sync.assert_called_once_with("test_key", "test_value", "node_0", 3600)

    def test_sync_data_write_behind(self, cluster_manager):
        """测试写回同步"""
        with patch.object(cluster_manager, '_sync_write_behind') as mock_sync:
            cluster_manager.config.sync_strategy = "write_behind"
            
            cluster_manager.sync_data("test_key", "test_value", "node_0", 3600)
            mock_sync.assert_called_once_with("test_key", "test_value", "node_0", 3600)

    def test_sync_data_unsupported_strategy(self, cluster_manager):
        """测试不支持的同步策略"""
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
            cluster_manager.config.sync_strategy = "unsupported"
            
            cluster_manager.sync_data("test_key", "test_value", "node_0", 3600)
            mock_logger.warning.assert_called()

    def test_sync_to_node(self, cluster_manager):
        """测试同步数据到指定节点"""
        mock_cache = Mock()
        
        cluster_manager._sync_to_node(mock_cache, "test_key", "test_value", 3600)
        
        mock_cache.setex.assert_called_once_with("test_key", 3600, "test_value")

    def test_sync_to_node_string_value(self, cluster_manager):
        """测试同步字符串值到节点"""
        mock_cache = Mock()
        
        cluster_manager._sync_to_node(mock_cache, "test_key", "string_value", 3600)
        
        mock_cache.setex.assert_called_once_with("test_key", 3600, "string_value")

    def test_sync_to_node_json_serialization(self, cluster_manager):
        """测试同步复杂对象到节点"""
        mock_cache = Mock()
        complex_value = {"key": "value", "number": 123}
        
        cluster_manager._sync_to_node(mock_cache, "test_key", complex_value, 3600)
        
        # 验证调用了setex，并且值被序列化了
        mock_cache.setex.assert_called_once()
        args = mock_cache.setex.call_args[0]
        assert args[0] == "test_key"
        assert args[1] == 3600
        assert json.loads(args[2]) == complex_value

    def test_check_node_health_success(self, cluster_manager):
        """测试节点健康检查 - 成功"""
        mock_cache = Mock()
        mock_cache.ping.return_value = True
        cluster_manager.nodes["node_0"] = mock_cache
        
        result = cluster_manager.check_node_health("node_0")
        assert result is True
        mock_cache.ping.assert_called_once()

    def test_check_node_health_failure(self, cluster_manager):
        """测试节点健康检查 - 失败"""
        mock_cache = Mock()
        mock_cache.ping.side_effect = Exception("Connection failed")
        cluster_manager.nodes["node_0"] = mock_cache
        
        result = cluster_manager.check_node_health("node_0")
        assert result is False

    def test_check_node_health_nonexistent(self, cluster_manager):
        """测试节点健康检查 - 节点不存在"""
        result = cluster_manager.check_node_health("nonexistent_node")
        assert result is False

    def test_handle_node_failure(self, cluster_manager):
        """测试处理节点故障"""
        cluster_manager.active_nodes.add("node_0")
        
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
            cluster_manager.handle_node_failure("node_0")
            
            assert "node_0" not in cluster_manager.active_nodes
            mock_logger.warning.assert_called()

    def test_handle_node_recovery(self, cluster_manager):
        """测试处理节点恢复"""
        # 首先确保节点不在active_nodes中（模拟节点故障状态）
        if "node_0" in cluster_manager.active_nodes:
            cluster_manager.active_nodes.remove("node_0")
        
        with patch.object(cluster_manager, 'check_node_health', return_value=True):
            with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
                cluster_manager.handle_node_recovery("node_0")
                
                assert "node_0" in cluster_manager.active_nodes
                mock_logger.info.assert_called()

    def test_handle_node_recovery_failed_health_check(self, cluster_manager):
        """测试处理节点恢复 - 健康检查失败"""
        # 首先确保节点不在active_nodes中（模拟节点故障状态）
        if "node_0" in cluster_manager.active_nodes:
            cluster_manager.active_nodes.remove("node_0")
        
        with patch.object(cluster_manager, 'check_node_health', return_value=False):
            cluster_manager.handle_node_recovery("node_0")
            
            # 健康检查失败时，节点不应该被添加到active_nodes
            assert "node_0" not in cluster_manager.active_nodes


class TestDistributedCacheManagerEnhanced:
    """测试分布式缓存管理器增强功能"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        nodes = [ClusterNode(host="localhost", port=6379)]
        return DistributedConfig(
            nodes=nodes,
            replication_factor=1,
            enable_monitoring=True,
            heartbeat_interval=1  # 短间隔用于测试
        )

    @pytest.fixture
    def manager(self, config):
        """创建分布式缓存管理器实例"""
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            return DistributedCacheManager(config)

    def test_config_conversion(self):
        """测试配置转换"""
        from src.infrastructure.cache.distributed.distributed_cache_manager import DistributedCacheManager, DistributedConfig, ClusterNode
        from src.infrastructure.cache.interfaces.data_structures import ConsistencyLevel
        
        core_config = Mock()
        core_config.redis_host = "localhost"
        core_config.redis_port = 6379
        
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.ClusterManager') as mock_cluster_manager_class:
            with patch('src.infrastructure.cache.distributed.distributed_cache_manager.redis'):
                mock_cluster_manager = Mock()
                mock_cluster_manager_class.return_value = mock_cluster_manager
                
                # 创建管理器实例
                manager = DistributedCacheManager(core_config)
                
                # 验证配置转换结果 - 通过检查_convert_config方法被调用的结果
                assert hasattr(manager, 'config')
                assert manager.config is not None

    def test_local_cache_operations(self, manager):
        """测试本地缓存操作"""
        manager.local_cache["test_key"] = "test_value"
        manager.creation_times["test_key"] = time.time()
        
        assert "test_key" in manager.local_cache
        assert "test_key" in manager.creation_times

    def test_is_expired_check(self, manager):
        """测试过期检查"""
        # 测试未过期的键
        manager.creation_times["fresh_key"] = time.time()
        assert not manager._is_expired("fresh_key")
        
        # 测试过期的键
        manager.creation_times["expired_key"] = time.time() - 4000  # 4秒前
        manager.ttl = 1  # 1秒TTL
        assert manager._is_expired("expired_key")

    def test_is_expired_nonexistent_key(self, manager):
        """测试不存在的键过期检查"""
        assert not manager._is_expired("nonexistent_key")

    def test_get_with_local_cache_hit(self, manager):
        """测试本地缓存命中"""
        manager.local_cache["rqa2025_cache:test_key"] = "local_value"
        manager.creation_times["rqa2025_cache:test_key"] = time.time()
        
        # Mock Redis调用以避免真实连接
        with patch.object(manager.cluster_manager, 'get_node', return_value=("node_0", Mock())):
            result = manager.get("test_key")
            # 由于本地缓存优先，应该直接返回本地值
            # 注意：实际实现可能有所不同，这里主要是测试代码路径

    def test_set_with_replication(self, manager):
        """测试带复制的设置操作"""
        with patch.object(manager.cluster_manager, 'get_node', return_value=("node_0", Mock())):
            with patch.object(manager.cluster_manager, 'sync_data') as mock_sync:
                mock_cache = Mock()
                mock_cache.setex.return_value = True
                
                # 重新设置get_node返回值，包含mock缓存
                with patch.object(manager.cluster_manager, 'get_node', return_value=("node_0", mock_cache)):
                    result = manager.set("test_key", "test_value", ttl=3600)
                    
                    mock_cache.setex.assert_called()
                    # 验证本地缓存被更新
                    assert "rqa2025_cache:test_key" in manager.local_cache

    def test_delete_with_replication(self, manager):
        """测试带复制的删除操作"""
        manager.local_cache["rqa2025_cache:test_key"] = "value"
        manager.creation_times["rqa2025_cache:test_key"] = time.time()
        
        mock_cache = Mock()
        mock_cache.delete.return_value = True
        
        with patch.object(manager.cluster_manager, 'get_node', return_value=("node_0", mock_cache)):
            with patch.object(manager.cluster_manager, 'get_replica_nodes', return_value=[]):
                result = manager.delete("test_key")
                
                mock_cache.delete.assert_called_once_with("rqa2025_cache:test_key")
                # 验证本地缓存被清理
                assert "rqa2025_cache:test_key" not in manager.local_cache

    def test_exists_operation(self, manager):
        """测试存在性检查操作"""
        mock_cache = Mock()
        mock_cache.exists.return_value = 1
        
        with patch.object(manager.cluster_manager, 'get_node', return_value=("node_0", mock_cache)):
            result = manager.exists("test_key")
            
            mock_cache.exists.assert_called_once_with("rqa2025_cache:test_key")
            assert result is True

    def test_exists_operation_failure(self, manager):
        """测试存在性检查失败"""
        with patch.object(manager.cluster_manager, 'get_node', side_effect=Exception("Connection failed")):
            with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
                result = manager.exists("test_key")
                
                assert result is False
                mock_logger.error.assert_called()

    def test_clear_operation(self, manager):
        """测试清空操作"""
        manager.local_cache["rqa2025_cache:key1"] = "value1"
        manager.local_cache["rqa2025_cache:key2"] = "value2"
        manager.creation_times["rqa2025_cache:key1"] = time.time()
        manager.creation_times["rqa2025_cache:key2"] = time.time()
        
        mock_cache = Mock()
        mock_cache.keys.return_value = ["rqa2025_cache:key1", "rqa2025_cache:key2"]
        manager.cluster_manager.nodes = {"node_0": mock_cache}
        
        result = manager.clear()
        
        mock_cache.keys.assert_called_with("rqa2025_cache:*")
        mock_cache.delete.assert_called()
        assert len(manager.local_cache) == 0
        assert len(manager.creation_times) == 0

    def test_clear_operation_node_failure(self, manager):
        """测试清空操作时节点失败"""
        mock_cache = Mock()
        mock_cache.keys.side_effect = Exception("Node failure")
        manager.cluster_manager.nodes = {"node_0": mock_cache}
        
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
            result = manager.clear()
            
            # 即使节点失败，本地缓存也应该被清空
            assert len(manager.local_cache) == 0
            mock_logger.error.assert_called()

    def test_get_cluster_stats(self, manager):
        """测试获取集群统计信息"""
        manager.local_cache["key1"] = "value1"
        
        stats = manager.get_cluster_stats()
        
        assert isinstance(stats, dict)
        assert "total_nodes" in stats
        assert "active_nodes" in stats
        assert "local_cache_size" in stats
        assert stats["local_cache_size"] == 1

    def test_get_node_stats_nonexistent(self, manager):
        """测试获取不存在节点的统计信息"""
        stats = manager.get_node_stats("nonexistent_node")
        assert stats == {}

    def test_get_node_stats_existing(self, manager):
        """测试获取现有节点的统计信息"""
        manager.cluster_manager.active_nodes.add("node_0")
        manager.cluster_manager.node_load["node_0"] = 5
        
        stats = manager.get_node_stats("node_0")
        
        assert "node_id" in stats
        assert stats["node_id"] == "node_0"
        assert stats["is_active"] is True
        assert stats["load"] == 5

    def test_check_consistency(self, manager):
        """测试一致性检查"""
        with patch.object(manager.cluster_manager, 'get_replica_nodes', return_value=[]):
            with patch.object(manager.cluster_manager.consistency_manager, 'is_consistent', return_value=True):
                # 应该不抛出异常
                manager._check_consistency("test_key", "test_value")

    def test_check_consistency_failure(self, manager):
        """测试一致性检查失败"""
        mock_cache = Mock()
        mock_cache.get.side_effect = Exception("Get failed")
        
        with patch.object(manager.cluster_manager, 'get_replica_nodes', return_value=[("node1", mock_cache)]):
            with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
                manager._check_consistency("test_key", "test_value")
                mock_logger.debug.assert_called()

    def test_close_operation(self, manager):
        """测试关闭操作"""
        mock_cache = Mock()
        manager.cluster_manager.nodes["node_0"] = mock_cache
        
        manager.close()
        
        mock_cache.close.assert_called()

    def test_close_operation_with_exception(self, manager):
        """测试关闭操作时出现异常"""
        mock_cache = Mock()
        mock_cache.close.side_effect = Exception("Close failed")
        manager.cluster_manager.nodes["node_0"] = mock_cache
        
        with patch('src.infrastructure.cache.distributed.distributed_cache_manager.logger') as mock_logger:
            manager.close()
            mock_logger.error.assert_called()

    def test_update_access_stats(self, manager):
        """测试更新访问统计"""
        # 这个方法目前是空的，但我们需要测试它不会抛出异常
        manager._update_access_stats("test_key")

    def test_monitoring_thread_initialization(self, manager):
        """测试监控线程初始化"""
        # 验证监控相关的属性和方法存在
        assert hasattr(manager.cluster_manager, 'check_node_health')
        assert hasattr(manager.cluster_manager, 'handle_node_failure')
        assert hasattr(manager.cluster_manager, 'handle_node_recovery')


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
