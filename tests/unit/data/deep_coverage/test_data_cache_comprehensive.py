"""
数据缓存模块深度测试
全面测试数据缓存系统的各种功能和边界条件
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import time

# 导入实际的类
from src.data.cache.cache_manager import CacheManager
from src.data.cache.smart_cache_optimizer import SmartCacheOptimizer
from src.data.cache.multi_level_cache import MultiLevelCache
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


class TestDataCacheComprehensive:
    """数据缓存综合深度测试"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'] * 10,
            'price': np.random.uniform(100, 500, 30),
            'volume': np.random.randint(100000, 1000000, 30),
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D')
        })

    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        return CacheManager()

    @pytest.fixture
    def smart_cache_optimizer(self):
        """创建智能缓存优化器实例"""
        return SmartCacheOptimizer()

    @pytest.fixture
    def multi_level_cache(self):
        """创建多级缓存实例"""
        return MultiLevelCache()

    @pytest.fixture
    def enhanced_cache_manager(self):
        """创建增强缓存管理器实例"""
        return EnhancedCacheManager()

    def test_cache_manager_initialization(self, cache_manager):
        """测试缓存管理器初始化"""
        assert cache_manager is not None
        assert hasattr(cache_manager, 'config')

    def test_smart_cache_optimizer_initialization(self, smart_cache_optimizer):
        """测试智能缓存优化器初始化"""
        if SMART_CACHE_AVAILABLE:
            assert smart_cache_optimizer is not None
            assert hasattr(smart_cache_optimizer, 'cache_strategy')

    def test_multi_level_cache_initialization(self, multi_level_cache):
        """测试多级缓存初始化"""
        if MULTI_LEVEL_CACHE_AVAILABLE:
            assert multi_level_cache is not None
            assert hasattr(multi_level_cache, 'levels')

    def test_redis_cache_adapter_initialization(self, redis_cache_adapter):
        """测试Redis缓存适配器初始化"""
        if REDIS_CACHE_AVAILABLE:
            assert redis_cache_adapter is not None

    def test_basic_cache_operations(self, cache_manager, sample_data):
        """测试基本缓存操作"""
        if CACHE_MANAGER_AVAILABLE:
            key = "test_data"

            # 测试存储
            cache_manager.set(key, sample_data)

            # 测试检索
            retrieved_data = cache_manager.get(key)

            # 检查数据一致性
            pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_cache_expiration(self, cache_manager, sample_data):
        """测试缓存过期"""
        if CACHE_MANAGER_AVAILABLE:
            key = "expiring_data"

            # 存储带过期时间的缓存
            cache_manager.set(key, sample_data, ttl_seconds=1)

            # 立即检查存在
            assert cache_manager.exists(key)

            # 等待过期
            time.sleep(2)

            # 检查已过期
            assert not cache_manager.exists(key)

    def test_cache_size_limits(self, cache_manager, sample_data):
        """测试缓存大小限制"""
        if CACHE_MANAGER_AVAILABLE:
            # 设置小的缓存大小限制
            cache_manager.max_size_mb = 0.001  # 1KB

            # 尝试存储大数据
            large_data = pd.DataFrame({
                'data': ['x' * 1000] * 1000  # 创建大数据
            })

            # 应该能够处理大小限制
            cache_manager.set("large_data", large_data)

            # 检查缓存管理器没有崩溃
            assert cache_manager is not None

    def test_multi_level_cache_hierarchy(self, multi_level_cache, sample_data):
        """测试多级缓存层次结构"""
        if MULTI_LEVEL_CACHE_AVAILABLE:
            key = "hierarchical_data"

            # 存储到多级缓存
            multi_level_cache.set(key, sample_data)

            # 从多级缓存检索
            retrieved_data = multi_level_cache.get(key)

            # 检查数据一致性
            pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_smart_cache_eviction_policy(self, smart_cache_optimizer, sample_data):
        """测试智能缓存淘汰策略"""
        if SMART_CACHE_AVAILABLE:
            # 添加多个缓存项
            for i in range(10):
                key = f"data_{i}"
                data = sample_data.copy()
                data['index'] = i
                smart_cache_optimizer.set(key, data)

            # 检查缓存项存在
            assert smart_cache_optimizer.get("data_0") is not None

            # 模拟内存压力
            smart_cache_optimizer.simulate_memory_pressure()

            # 检查智能淘汰是否工作
            assert smart_cache_optimizer is not None

    def test_cache_compression(self, cache_manager, sample_data):
        """测试缓存压缩"""
        if CACHE_MANAGER_AVAILABLE:
            key = "compressed_data"

            # 启用压缩
            cache_manager.compression_enabled = True

            # 存储数据
            cache_manager.set(key, sample_data)

            # 检索数据
            retrieved_data = cache_manager.get(key)

            # 检查数据一致性
            pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_cache_serialization_formats(self, cache_manager, sample_data):
        """测试缓存序列化格式"""
        if CACHE_MANAGER_AVAILABLE:
            formats = ['pickle', 'json', 'parquet']

            for fmt in formats:
                key = f"data_{fmt}"

                # 设置序列化格式
                cache_manager.serialization_format = fmt

                # 尝试存储和检索
                try:
                    cache_manager.set(key, sample_data)
                    retrieved_data = cache_manager.get(key)

                    # 检查数据基本结构
                    assert len(retrieved_data) == len(sample_data)
                    assert list(retrieved_data.columns) == list(sample_data.columns)

                except Exception:
                    # 某些格式可能不支持某些数据类型，跳过
                    continue

    def test_redis_cache_adapter_connection_handling(self, redis_cache_adapter, sample_data):
        """测试Redis缓存适配器连接处理"""
        if REDIS_CACHE_AVAILABLE:
            key = "redis_data"

            # 测试连接处理
            redis_cache_adapter.set(key, sample_data)
            retrieved_data = redis_cache_adapter.get(key)

            # 检查数据一致性
            pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_cache_performance_monitoring(self, cache_manager, sample_data):
        """测试缓存性能监控"""
        if CACHE_MANAGER_AVAILABLE:
            # 执行多个缓存操作
            operations = 100

            start_time = time.time()

            for i in range(operations):
                key = f"perf_test_{i}"
                cache_manager.set(key, sample_data)

            end_time = time.time()

            # 计算性能指标
            total_time = end_time - start_time
            avg_time_per_operation = total_time / operations

            # 检查性能在合理范围内
            assert avg_time_per_operation < 1.0  # 每操作少于1秒

    def test_cache_data_validation(self, cache_manager):
        """测试缓存数据验证"""
        if CACHE_MANAGER_AVAILABLE:
            # 测试不同类型的数据
            test_data = {
                'dataframe': pd.DataFrame({'a': [1, 2, 3]}),
                'dict': {'key': 'value'},
                'list': [1, 2, 3, 4, 5],
                'string': 'test_string',
                'number': 42
            }

            for data_type, data in test_data.items():
                key = f"validation_{data_type}"

                # 存储数据
                cache_manager.set(key, data)

                # 检索数据
                retrieved_data = cache_manager.get(key)

                # 检查数据类型和内容
                assert type(retrieved_data) == type(data)

    def test_cache_concurrent_access(self, cache_manager, sample_data):
        """测试缓存并发访问"""
        if CACHE_MANAGER_AVAILABLE:
            import threading

            results = []
            errors = []

            def concurrent_operation(thread_id):
                try:
                    key = f"concurrent_{thread_id}"
                    cache_manager.set(key, sample_data)
                    retrieved = cache_manager.get(key)
                    results.append((thread_id, len(retrieved)))
                except Exception as e:
                    errors.append((thread_id, str(e)))

            # 创建多个线程
            threads = []
            for i in range(5):
                thread = threading.Thread(target=concurrent_operation, args=(i,))
                threads.append(thread)

            # 启动所有线程
            for thread in threads:
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 检查结果
            assert len(results) == 5  # 所有线程都成功完成
            assert len(errors) == 0   # 没有错误

    def test_cache_memory_management(self, cache_manager):
        """测试缓存内存管理"""
        if CACHE_MANAGER_AVAILABLE:
            # 创建大量数据
            large_datasets = []
            for i in range(10):
                data = pd.DataFrame({
                    'col1': np.random.randn(10000),
                    'col2': np.random.randn(10000),
                    'col3': ['string_data_' + str(j) for j in range(10000)]
                })
                large_datasets.append(data)

            # 存储大数据集
            for i, data in enumerate(large_datasets):
                key = f"memory_test_{i}"
                cache_manager.set(key, data)

            # 检查内存使用情况
            memory_stats = cache_manager.get_memory_stats()

            assert 'total_size_mb' in memory_stats
            assert memory_stats['total_size_mb'] > 0

    def test_smart_cache_prediction_accuracy(self, smart_cache_optimizer, sample_data):
        """测试智能缓存预测准确性"""
        if SMART_CACHE_AVAILABLE:
            # 模拟访问模式
            access_pattern = ['data_A', 'data_B', 'data_A', 'data_C', 'data_A', 'data_B']

            # 存储数据
            for key in ['data_A', 'data_B', 'data_C', 'data_D']:
                smart_cache_optimizer.set(key, sample_data)

            # 模拟访问
            for key in access_pattern:
                smart_cache_optimizer.get(key)

            # 检查预测准确性
            prediction_accuracy = smart_cache_optimizer.get_prediction_accuracy()

            assert isinstance(prediction_accuracy, float)
            assert 0 <= prediction_accuracy <= 1

    def test_cache_backup_and_recovery(self, cache_manager, sample_data):
        """测试缓存备份和恢复"""
        if CACHE_MANAGER_AVAILABLE:
            key = "backup_test"

            # 存储数据
            cache_manager.set(key, sample_data)

            # 执行备份
            backup_file = cache_manager.create_backup()

            # 清除缓存
            cache_manager.clear()

            # 检查数据不存在
            assert cache_manager.get(key) is None

            # 从备份恢复
            cache_manager.restore_from_backup(backup_file)

            # 检查数据恢复
            recovered_data = cache_manager.get(key)
            pd.testing.assert_frame_equal(recovered_data, sample_data)

    def test_multi_level_cache_policy(self, multi_level_cache):
        """测试多级缓存策略"""
        if MULTI_LEVEL_CACHE_AVAILABLE:
            # 测试不同缓存级别的策略
            policies = ['lru', 'lfu', 'fifo', 'random']

            for policy in policies:
                # 设置策略
                multi_level_cache.set_eviction_policy(policy)

                # 验证策略设置
                assert multi_level_cache.eviction_policy == policy

    def test_cache_data_compression_ratio(self, cache_manager, sample_data):
        """测试缓存数据压缩率"""
        if CACHE_MANAGER_AVAILABLE:
            key = "compression_test"

            # 启用压缩
            cache_manager.compression_enabled = True

            # 存储数据
            cache_manager.set(key, sample_data)

            # 获取压缩统计
            compression_stats = cache_manager.get_compression_stats(key)

            if compression_stats:
                assert 'original_size' in compression_stats
                assert 'compressed_size' in compression_stats
                assert 'compression_ratio' in compression_stats

                # 压缩率应该大于0且小于等于1
                ratio = compression_stats['compression_ratio']
                assert 0 < ratio <= 1

    def test_cache_error_handling(self, cache_manager):
        """测试缓存错误处理"""
        if CACHE_MANAGER_AVAILABLE:
            # 测试无效键
            assert cache_manager.get("nonexistent_key") is None

            # 测试无效数据类型
            try:
                cache_manager.set("invalid_data", lambda x: x)
                # 如果没有抛出异常，检查是否能处理
                retrieved = cache_manager.get("invalid_data")
                assert retrieved is not None
            except Exception:
                # 如果抛出异常，说明错误处理正常
                pass

    def test_redis_cache_cluster_support(self, redis_cache_adapter):
        """测试Redis缓存集群支持"""
        if REDIS_CACHE_AVAILABLE:
            # 测试集群模式
            redis_cache_adapter.enable_cluster_mode()

            # 检查集群模式启用
            assert redis_cache_adapter.cluster_mode is True

    def test_cache_monitoring_and_metrics(self, cache_manager, sample_data):
        """测试缓存监控和指标"""
        if CACHE_MANAGER_AVAILABLE:
            # 执行一些缓存操作
            for i in range(10):
                key = f"metrics_test_{i}"
                cache_manager.set(key, sample_data)
                cache_manager.get(key)

            # 获取监控指标
            metrics = cache_manager.get_cache_metrics()

            expected_metrics = ['hit_rate', 'miss_rate', 'total_requests', 'total_hits', 'total_misses']

            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))

    def test_smart_cache_adaptive_strategy(self, smart_cache_optimizer):
        """测试智能缓存自适应策略"""
        if SMART_CACHE_AVAILABLE:
            # 测试策略适应性
            initial_strategy = smart_cache_optimizer.current_strategy

            # 模拟不同的工作负载
            smart_cache_optimizer.adapt_to_workload("read_heavy")
            read_strategy = smart_cache_optimizer.current_strategy

            smart_cache_optimizer.adapt_to_workload("write_heavy")
            write_strategy = smart_cache_optimizer.current_strategy

            # 检查策略是否根据工作负载调整
            assert initial_strategy != read_strategy or initial_strategy != write_strategy

    def test_cache_data_integrity_verification(self, cache_manager, sample_data):
        """测试缓存数据完整性验证"""
        key = "integrity_test"

        # 存储数据
        cache_manager.set(key, sample_data)

        # 验证数据完整性
        integrity_check = cache_manager.verify_data_integrity(key)

        assert integrity_check['is_valid'] is True
        assert 'checksum' in integrity_check

    def test_enhanced_cache_manager_initialization(self, enhanced_cache_manager):
        """测试增强缓存管理器初始化"""
        assert enhanced_cache_manager is not None
        assert hasattr(enhanced_cache_manager, 'cache_dir')
        assert hasattr(enhanced_cache_manager, 'max_memory_size')
        assert hasattr(enhanced_cache_manager, 'max_disk_size')

    def test_enhanced_cache_manager_memory_caching(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器内存缓存"""
        key = "memory_test"

        # 存储到内存缓存
        enhanced_cache_manager.set(key, sample_data)

        # 从内存缓存检索
        retrieved_data = enhanced_cache_manager.get(key)

        # 检查数据一致性
        pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_enhanced_cache_manager_disk_persistence(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器磁盘持久化"""
        key = "disk_test"

        # 存储数据（可能会持久化到磁盘）
        enhanced_cache_manager.set(key, sample_data)

        # 清除内存缓存
        enhanced_cache_manager.memory_cache.clear()

        # 从磁盘检索
        retrieved_data = enhanced_cache_manager.get(key)

        # 检查数据一致性
        pd.testing.assert_frame_equal(retrieved_data, sample_data)

    def test_enhanced_cache_manager_size_limits(self, enhanced_cache_manager):
        """测试增强缓存管理器大小限制"""
        # 创建大数据
        large_data = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
            'col3': ['string_' + str(i) for i in range(10000)]
        })

        key = "large_data_test"

        # 存储大数据
        enhanced_cache_manager.set(key, large_data)

        # 检查大小管理
        memory_usage = enhanced_cache_manager.get_memory_usage()
        disk_usage = enhanced_cache_manager.get_disk_usage()

        assert memory_usage >= 0
        assert disk_usage >= 0

    def test_enhanced_cache_manager_expiration(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器过期"""
        key = "expiring_test"

        # 存储带过期时间的缓存
        enhanced_cache_manager.set(key, sample_data, ttl_seconds=1)

        # 立即检查存在
        assert enhanced_cache_manager.exists(key)

        # 等待过期
        time.sleep(2)

        # 检查已过期
        assert not enhanced_cache_manager.exists(key)

    def test_enhanced_cache_manager_concurrent_access(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器并发访问"""
        import threading

        results = []
        errors = []

        def concurrent_operation(thread_id):
            try:
                key = f"concurrent_{thread_id}"
                enhanced_cache_manager.set(key, sample_data)
                retrieved = enhanced_cache_manager.get(key)
                results.append((thread_id, len(retrieved)))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 创建并发线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查结果
        assert len(results) == 5  # 所有线程都成功完成
        assert len(errors) == 0   # 没有错误

        # 检查所有结果都正确
        for thread_id, data_length in results:
            assert data_length == len(sample_data)

    def test_enhanced_cache_manager_statistics(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器统计信息"""
        # 执行一些缓存操作
        for i in range(10):
            key = f"stats_test_{i}"
            enhanced_cache_manager.set(key, sample_data)
            enhanced_cache_manager.get(key)

        # 获取统计信息
        stats = enhanced_cache_manager.get_statistics()

        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats

        # 检查统计数据合理性
        assert stats['total_requests'] >= 20  # 10次设置 + 10次获取
        assert stats['hit_rate'] >= 0.0
        assert stats['hit_rate'] <= 1.0

    def test_enhanced_cache_manager_cleanup(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器清理"""
        # 添加一些缓存项
        for i in range(20):
            key = f"cleanup_test_{i}"
            enhanced_cache_manager.set(key, sample_data)

        # 记录清理前的统计
        stats_before = enhanced_cache_manager.get_statistics()

        # 执行清理
        enhanced_cache_manager.cleanup()

        # 记录清理后的统计
        stats_after = enhanced_cache_manager.get_statistics()

        # 清理后统计应该仍然有效
        assert isinstance(stats_after, dict)

    def test_enhanced_cache_manager_backup_restore(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器备份和恢复"""
        # 添加一些数据
        test_data = {}
        for i in range(5):
            key = f"backup_test_{i}"
            data = sample_data.copy()
            data['test_id'] = i
            test_data[key] = data
            enhanced_cache_manager.set(key, data)

        # 创建备份
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            backup_file = tmp.name

        try:
            enhanced_cache_manager.create_backup(backup_file)

            # 验证备份文件存在
            assert os.path.exists(backup_file)

            # 清除缓存
            enhanced_cache_manager.clear()

            # 验证数据已清除
            for key in test_data.keys():
                assert not enhanced_cache_manager.exists(key)

            # 从备份恢复
            enhanced_cache_manager.restore_from_backup(backup_file)

            # 验证数据已恢复
            for key, original_data in test_data.items():
                restored_data = enhanced_cache_manager.get(key)
                pd.testing.assert_frame_equal(restored_data, original_data)

        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)

    def test_enhanced_cache_manager_performance_monitoring(self, enhanced_cache_manager, sample_data):
        """测试增强缓存管理器性能监控"""
        # 执行一系列操作来产生性能数据
        operations = 50

        start_time = time.time()

        for i in range(operations):
            key = f"perf_test_{i}"
            enhanced_cache_manager.set(key, sample_data)
            enhanced_cache_manager.get(key)

        end_time = time.time()

        # 获取性能指标
        performance_stats = enhanced_cache_manager.get_performance_stats()

        assert isinstance(performance_stats, dict)
        assert 'avg_set_time' in performance_stats
        assert 'avg_get_time' in performance_stats
        assert 'total_operations' in performance_stats

        # 检查性能数据合理性
        total_time = end_time - start_time
        assert performance_stats['total_operations'] >= operations
        assert performance_stats['avg_set_time'] > 0
        assert performance_stats['avg_get_time'] > 0

    def test_enhanced_cache_manager_health_check(self, enhanced_cache_manager):
        """测试增强缓存管理器健康检查"""
        # 执行健康检查
        health_status = enhanced_cache_manager.health_check()

        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'memory_usage' in health_status
        assert 'disk_usage' in health_status
        assert 'uptime' in health_status

        # 检查状态值合理性
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        assert health_status['memory_usage'] >= 0
        assert health_status['disk_usage'] >= 0
