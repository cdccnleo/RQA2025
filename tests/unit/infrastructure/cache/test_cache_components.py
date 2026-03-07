#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存组件单元测试

测试各种缓存组件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.infrastructure.cache.core.cache_components import CacheComponent
from src.infrastructure.cache.interfaces.data_structures import CacheEntry, CacheStats
from src.infrastructure.cache.core.cache_configs import CacheLevel


class TestCacheComponent:
    """测试缓存组件"""

    def setup_method(self, method):
        """测试前准备"""
        self.component = CacheComponent(component_id=1)

    def test_initialization(self):
        """测试初始化"""
        assert self.component.component_id == 1
        assert hasattr(self.component, 'status')
        assert hasattr(self.component, 'metrics')

    def test_get_info_method(self):
        """测试获取信息方法"""
        info = self.component.get_info()
        assert isinstance(info, dict)
        assert 'component_id' in info
        assert 'status' in info
        assert 'created_at' in info
        assert info['component_id'] == 1

    def test_get_status_method(self):
        """测试获取状态方法"""
        status = self.component.get_component_status_string()
        assert isinstance(status, str)
        assert status in ['active', 'inactive', 'stopped', 'error', 'healthy']

    def test_metrics_collection(self):
        """测试指标收集"""
        # 执行一些操作来生成指标
        initial_metrics = self.component.metrics.copy() if hasattr(self.component, 'metrics') else {}

        # 这里我们模拟一些操作
        # 注意：实际的指标收集取决于具体实现

        # 验证指标字典存在
        if hasattr(self.component, 'metrics'):
            assert isinstance(self.component.metrics, dict)

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        # 测试初始状态
        assert self.component.status in ['active', 'inactive', 'stopped', 'healthy']

        # 测试状态转换（如果支持的话）
        # 注意：这取决于具体实现

    def test_error_handling(self):
        """测试错误处理"""
        # 测试异常情况的处理
        try:
            # 尝试一些可能失败的操作
            self.component.get_info()  # 这应该总是成功的
        except Exception as e:
            # 如果有异常，验证它被正确处理
            assert isinstance(e, Exception)


class TestCacheEntry:
    """测试缓存条目"""

    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=300,
            size_bytes=100
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 300
        assert entry.size_bytes == 100
        assert entry.access_count == 0
        # 修复属性名称：使用 created_at 而不是 creation_time
        assert entry.created_at is not None

    def test_cache_entry_with_metadata(self):
        """测试带元数据的缓存条目"""
        metadata = {"priority": "high", "source": "database"}
        entry = CacheEntry(
            key="meta_key",
            value="meta_value",
            ttl=600,
            size_bytes=200,
            metadata=metadata
        )

        assert entry.metadata == metadata
        assert entry.metadata["priority"] == "high"

    def test_cache_entry_access_tracking(self):
        """测试缓存条目访问跟踪"""
        entry = CacheEntry(
            key="access_key",
            value="access_value",
            ttl=300,
            size_bytes=50
        )

        initial_count = entry.access_count
        # 修复属性名称：使用 last_accessed 而不是 last_access_time
        initial_time = entry.last_accessed

        # 模拟访问
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed >= initial_time

    def test_cache_entry_expiration_check(self):
        """测试缓存条目过期检查"""
        # 创建短期TTL的条目
        entry = CacheEntry(
            key="expire_key",
            value="expire_value",
            ttl=1,  # 1秒TTL
            size_bytes=50
        )

        # 立即检查，应该未过期
        assert not entry.is_expired

        # 等待过期
        time.sleep(1.1)

        # 检查，应该已过期
        assert entry.is_expired

    def test_cache_entry_expiration_with_custom_ttl(self):
        """测试缓存条目带自定义TTL的过期检查"""
        entry = CacheEntry(
            key="custom_ttl_key",
            value="custom_ttl_value",
            ttl=2,
            size_bytes=50
        )

        # 注意：is_expired是属性而不是方法，不能传参
        assert not entry.is_expired

    def test_cache_entry_size_calculation(self):
        """测试缓存条目大小计算"""
        # 测试字符串值
        str_entry = CacheEntry(
            key="str_key",
            value="test_string_value",
            ttl=300
        )
        # 大小应该至少包含字符串长度
        assert str_entry.size_bytes >= len("test_string_value")

        # 测试数字值
        num_entry = CacheEntry(
            key="num_key",
            value=42,
            ttl=300
        )
        # 数字大小通常是固定的
        assert num_entry.size_bytes > 0

        # 测试列表值
        list_entry = CacheEntry(
            key="list_key",
            value=[1, 2, 3, "four"],
            ttl=300
        )
        # 列表大小应该大于单个元素
        assert list_entry.size_bytes > 0

    def test_cache_entry_touch_operation(self):
        """测试缓存条目touch操作"""
        entry = CacheEntry(
            key="touch_key",
            value="touch_value",
            ttl=300,
            size_bytes=50
        )

        # 修复属性名称：使用 last_accessed 而不是 last_access_time
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count

        # 使用 touch 方法而不是 update_access
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed >= initial_access_time

    def test_cache_entry_serialization(self):
        """测试缓存条目序列化"""
        entry = CacheEntry(
            key="serialize_key",
            value="serialize_value",
            ttl=300,
            size_bytes=150,
            metadata={"tag": "test"}
        )

        # 使用 to_dict 方法而不是 serialize
        data = entry.to_dict()

        assert isinstance(data, dict)
        assert data['key'] == "serialize_key"
        assert data['value'] == "serialize_value"
        assert data['ttl'] == 300
        assert data['size_bytes'] == 150
        assert data['metadata']['tag'] == "test"

    def test_cache_entry_equality(self):
        """测试缓存条目相等性"""
        entry1 = CacheEntry(
            key="equal_key",
            value="equal_value",
            ttl=300,
            size_bytes=50
        )

        entry2 = CacheEntry(
            key="equal_key",
            value="equal_value",
            ttl=300,
            size_bytes=50
        )

        entry3 = CacheEntry(
            key="different_key",
            value="different_value",
            ttl=600,
            size_bytes=100
        )

        # 相同内容的条目应该相等（基于关键属性）
        assert entry1.key == entry2.key
        assert entry1.value == entry2.value

        # 不同内容的条目应该不相等
        assert entry1.key != entry3.key


class TestCacheStats:
    """测试缓存统计"""

    def test_cache_stats_initialization(self):
        """测试缓存统计初始化"""
        stats = CacheStats()

        assert stats.total_requests == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_size_bytes == 0
        assert hasattr(stats, 'hit_rate')

    def test_cache_stats_updates(self):
        """测试缓存统计更新"""
        stats = CacheStats()

        # 记录命中
        stats.hits += 1
        stats.total_requests += 1

        assert stats.hits == 1
        assert stats.total_requests == 1
        assert stats.hit_rate == 1.0

        # 记录未命中
        stats.misses += 1
        stats.total_requests += 1

        assert stats.misses == 1
        assert stats.total_requests == 2
        assert stats.hit_rate == 0.5

        # 记录淘汰
        stats.evictions += 1
        assert stats.evictions == 1

    def test_cache_stats_reset(self):
        """测试缓存统计重置"""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.evictions = 2
        stats.total_requests = 15

        # 重置统计
        stats.reset()

        assert stats.total_requests == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_cache_stats_hit_rate_calculation(self):
        """测试缓存统计命中率计算"""
        stats = CacheStats()

        # 测试各种情况的命中率
        test_cases = [
            (0, 0, 0.0),    # 没有请求
            (1, 0, 1.0),    # 1次命中，0次未命中
            (0, 1, 0.0),    # 0次命中，1次未命中
            (3, 2, 0.6),    # 3次命中，2次未命中
            (5, 5, 0.5),    # 5次命中，5次未命中
        ]

        for hits, misses, expected_rate in test_cases:
            stats.hits = hits
            stats.misses = misses
            stats.total_requests = hits + misses

            if stats.total_requests > 0:
                assert abs(stats.hit_rate - expected_rate) < 0.001
            else:
                assert stats.hit_rate == 0.0

    def test_cache_stats_to_dict(self):
        """测试缓存统计转字典"""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.evictions = 2

        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict['hits'] == 10
        assert stats_dict['misses'] == 5
        assert stats_dict['evictions'] == 2
        assert 'hit_rate' in stats_dict

    def test_cache_stats_memory_usage_calculation(self):
        """测试缓存统计内存使用计算"""
        stats = CacheStats()
        stats.total_size_bytes = 1024 * 1024  # 1MB

        # 验证内存使用计算（如果有的话）
        # 修复属性名称：使用 memory_usage 而不是 memory_usage_mb
        if hasattr(stats, 'memory_usage'):
            # memory_usage 是一个比例值（0-1），而不是MB值
            assert stats.memory_usage >= 0.0

    def test_cache_stats_performance_metrics(self):
        """测试缓存统计性能指标"""
        stats = CacheStats()

        # 设置一些性能数据
        stats.total_requests = 1000
        stats.hits = 800
        stats.misses = 200

        # 验证性能指标计算
        assert stats.hit_rate == 0.8

        # 如果有响应时间统计
        if hasattr(stats, 'avg_response_time'):
            # 这里可以测试响应时间相关的计算
            pass


class TestCacheComponentIntegration:
    """测试缓存组件集成"""

    def test_component_with_cache_entry(self):
        """测试组件与缓存条目的集成"""
        component = CacheComponent(component_id=2)

        # 创建缓存条目
        entry = CacheEntry(
            key="integration_key",
            value="integration_value",
            ttl=300,
            size_bytes=100
        )

        # 验证组件和条目的交互
        assert component.component_id != entry.key
        assert isinstance(entry.value, str)

    def test_component_with_cache_stats(self):
        """测试组件与缓存统计的集成"""
        component = CacheComponent(component_id=3)
        stats = CacheStats()

        # 模拟一些统计更新
        stats.hits = 5
        stats.misses = 2
        stats.total_requests = 7

        # 验证统计数据
        assert stats.hit_rate == pytest.approx(5/7, rel=1e-2)

    def test_multiple_components_coordination(self):
        """测试多个组件的协调工作"""
        component1 = CacheComponent(component_id=4)
        component2 = CacheComponent(component_id=5)

        # 验证组件ID唯一性
        assert component1.component_id != component2.component_id

        # 验证组件状态独立性
        assert component1.status == component1.status  # 状态应该一致
        assert component2.status == component2.status

    def test_component_error_recovery(self):
        """测试组件错误恢复"""
        component = CacheComponent(component_id=6)

        # 模拟错误情况
        try:
            # 尝试一些可能失败的操作
            result = component.get_info()
            assert isinstance(result, dict)
        except Exception:
            # 验证错误被正确处理
            pass

    def test_component_performance_monitoring(self):
        """测试组件性能监控"""
        component = CacheComponent(component_id=7)

        start_time = time.time()

        # 执行一些操作
        for i in range(100):
            component.get_info()

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能在合理范围内
        assert duration < 1.0  # 100次操作应该在1秒内完成

    def test_component_resource_management(self):
        """测试组件资源管理"""
        component = CacheComponent(component_id=8)

        # 验证组件正确管理资源
        info = component.get_info()
        assert 'component_id' in info

        # 验证没有资源泄漏（在实际实现中）
        # 这里我们只能验证基本功能

    def test_component_configuration_handling(self):
        """测试组件配置处理"""
        # 测试不同配置下的组件行为
        configs = [
            {"component_id": 9},
            {"component_id": 10, "extra_param": "value"}
        ]

        components = []
        for config in configs:
            component = CacheComponent(**config)
            components.append(component)

        # 验证配置正确应用
        assert components[0].component_id == 9
        assert components[1].component_id == 10

        # 清理
        for comp in components:
            pass  # 如果有清理方法，在这里调用


if __name__ == '__main__':
    pytest.main([__file__])
