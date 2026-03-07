#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Core日志器池管理

测试logging/core/logger_pool.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch


class TestCoreLoggerPool:
    """测试Core日志器池管理"""

    def setup_method(self):
        """测试前准备"""
        from src.infrastructure.logging.core.logger_pool import (
            LoggerPool, LoggerPoolState, LoggerStats
        )
        self.LoggerPool = LoggerPool
        self.LoggerPoolState = LoggerPoolState
        self.LoggerStats = LoggerStats

    def test_logger_pool_state_enum(self):
        """测试日志器池状态枚举"""
        assert self.LoggerPoolState.HEALTHY.value == "healthy"
        assert self.LoggerPoolState.WARNING.value == "warning"
        assert self.LoggerPoolState.CRITICAL.value == "critical"
        assert self.LoggerPoolState.FAILED.value == "failed"

    def test_logger_stats_creation(self):
        """测试日志器统计信息创建"""
        stats = self.LoggerStats(
            logger_id="test_logger",
            created_time=1234567890.0,
            last_used_time=1234567900.0,
            use_count=5,
            error_count=1,
            memory_usage=1024
        )

        assert stats.logger_id == "test_logger"
        assert stats.created_time == 1234567890.0
        assert stats.last_used_time == 1234567900.0
        assert stats.use_count == 5
        assert stats.error_count == 1
        assert stats.memory_usage == 1024

    def test_logger_stats_to_dict(self):
        """测试日志器统计信息转换为字典"""
        stats = self.LoggerStats(
            logger_id="test_logger",
            created_time=1234567890.0,
            last_used_time=1234567900.0,
            use_count=5,
            error_count=1,
            memory_usage=1024
        )

        dict_result = stats.to_dict()
        assert isinstance(dict_result, dict)
        assert dict_result["logger_id"] == "test_logger"
        assert dict_result["created_time"] == 1234567890.0
        assert dict_result["last_used_time"] == 1234567900.0
        assert dict_result["use_count"] == 5
        assert dict_result["error_count"] == 1
        assert dict_result["memory_usage"] == 1024

    def test_logger_pool_initialization(self):
        """测试日志器池初始化"""
        pool = self.LoggerPool()

        assert hasattr(pool, '_loggers')
        assert hasattr(pool, '_stats')
        assert hasattr(pool, '_max_size')
        assert hasattr(pool, '_lock')
        assert isinstance(pool._loggers, dict)
        assert isinstance(pool._stats, dict)
        assert pool._max_size == 100

    def test_logger_pool_singleton(self):
        """测试日志器池单例模式"""
        pool1 = self.LoggerPool.get_instance()
        pool2 = self.LoggerPool.get_instance()

        # 单例模式下应该返回同一个实例
        assert pool1 is pool2

    def test_get_logger_new(self):
        """测试获取新的日志器"""
        pool = self.LoggerPool()

        # 清空池以确保测试独立性
        pool._loggers.clear()
        pool._stats.clear()

        logger = pool.get_logger("test_logger")

        assert logger is not None
        assert "test_logger" in pool._loggers
        assert "test_logger" in pool._stats

        # 检查统计信息
        stats = pool._stats["test_logger"]
        assert stats.logger_id == "test_logger"
        assert stats.use_count == 1
        assert stats.error_count == 0
        assert isinstance(stats.created_time, float)
        assert isinstance(stats.last_used_time, float)

    def test_get_logger_existing(self):
        """测试获取已存在的日志器"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 第一次获取
        logger1 = pool.get_logger("test_logger")
        stats1 = pool._stats["test_logger"]
        initial_use_count = stats1.use_count

        # 第二次获取
        logger2 = pool.get_logger("test_logger")
        stats2 = pool._stats["test_logger"]

        # 应该是同一个实例
        assert logger1 is logger2
        # 使用次数应该增加
        assert stats2.use_count == initial_use_count + 1
        # 最后使用时间应该更新
        assert stats2.last_used_time >= stats1.last_used_time

    def test_get_pool_stats(self):
        """测试获取池统计信息"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 添加一些日志器
        pool.get_logger("logger1")
        pool.get_logger("logger2")
        pool.get_logger("logger1")  # 再次使用logger1

        stats = pool.get_stats()

        assert isinstance(stats, dict)
        assert "pool_size" in stats
        assert "total_loggers_created" in stats
        assert "state" in stats

        assert stats["pool_size"] == 2  # logger1 和 logger2
        assert stats["total_loggers_created"] == 2
        assert stats["state"] == "healthy"

    def test_remove_logger(self):
        """测试移除日志器"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 添加日志器
        pool.get_logger("test_logger")
        assert "test_logger" in pool._loggers
        assert "test_logger" in pool._stats

        # 移除日志器
        pool.remove_logger("test_logger")
        assert "test_logger" not in pool._loggers
        assert "test_logger" not in pool._stats

        # 移除不存在的日志器（不应该抛出异常）
        pool.remove_logger("nonexistent")

    def test_shutdown_pool(self):
        """测试关闭池"""
        pool = self.LoggerPool()

        # 添加一些日志器
        pool.get_logger("logger1")
        pool.get_logger("logger2")
        pool.get_logger("logger3")

        assert len(pool._loggers) > 0
        assert len(pool._stats) > 0

        # 关闭池
        pool.shutdown()

        assert len(pool._loggers) == 0
        assert len(pool._stats) == 0
        assert pool._state == self.LoggerPoolState.FAILED

    def test_get_pool_status(self):
        """测试获取池状态"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 添加日志器
        pool.get_logger("alpha")
        pool.get_logger("beta")
        pool.get_logger("gamma")

        status = pool.get_pool_status()
        assert isinstance(status, dict)
        assert "pool_size" in status
        assert "utilization_rate" in status
        assert "health_status" in status
        assert status["pool_size"] == 3

    def test_pool_size_limits(self):
        """测试池大小限制"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 设置小的最大大小
        pool._max_size = 2

        # 添加日志器
        pool.get_logger("logger1")
        pool.get_logger("logger2")
        pool.get_logger("logger3")  # 超过限制

        # 应该只有2个日志器
        assert len(pool._loggers) <= pool._max_size

    def test_thread_safety(self):
        """测试线程安全性"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        results = []

        def worker(worker_id):
            """工作线程"""
            for i in range(10):
                logger_name = f"logger_{worker_id}_{i}"
                logger = pool.get_logger(logger_name)
                results.append((worker_id, logger_name, logger is not None))

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 30  # 3线程 * 10次调用
        assert all(result[2] for result in results)  # 所有调用都成功

        # 验证最终状态
        assert len(pool._loggers) == 30
        assert len(pool._stats) == 30

    def test_error_handling(self):
        """测试错误处理"""
        pool = self.LoggerPool()

        # 验证_create_logger方法正常工作
        result = pool._create_logger("test_logger")
        assert result is not None
        assert hasattr(result, 'logger')  # UnifiedLogger有logger属性

    def test_performance_monitoring(self):
        """测试性能监控"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        start_time = time.time()

        # 执行一系列操作
        for i in range(50):
            pool.get_logger(f"perf_logger_{i}")

        end_time = time.time()

        # 验证操作在合理时间内完成
        duration = end_time - start_time
        assert duration < 1.0  # 应该很快完成

        # 验证统计信息准确性
        stats = pool.get_stats()
        assert stats["pool_size"] == 50
        assert stats["total_loggers_created"] == 50

    def test_memory_usage_tracking(self):
        """测试内存使用跟踪"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 添加日志器
        pool.get_logger("memory_test")

        # 检查内存使用是否被跟踪
        stats = pool._stats["memory_test"]
        assert hasattr(stats, 'memory_usage')
        assert isinstance(stats.memory_usage, int)

    def test_pool_status(self):
        """测试池状态"""
        pool = self.LoggerPool()

        # 清空池
        pool._loggers.clear()
        pool._stats.clear()

        # 空的池状态
        status = pool.get_pool_status()
        assert isinstance(status, dict)
        assert "pool_size" in status
        assert "utilization_rate" in status
        assert "health_status" in status

        # 验证健康状态
        assert status["health_status"] in ["healthy", "warning", "critical", "failed"]

    def test_pool_eviction_mechanism(self):
        """测试池驱逐机制"""
        pool = self.LoggerPool(max_size=2)

        # 创建超过最大大小的日志器
        logger1 = pool.get_logger("evict_test_1")
        logger2 = pool.get_logger("evict_test_2")
        logger3 = pool.get_logger("evict_test_3")  # 这应该触发驱逐

        # 池大小应该仍然是2
        stats = pool.get_stats()
        assert stats["pool_size"] == 2

        # 最早的日志器应该被驱逐
        assert "evict_test_1" not in pool._loggers
        assert "evict_test_2" in pool._loggers
        assert "evict_test_3" in pool._loggers

    def test_idle_logger_cleanup(self):
        """测试空闲日志器清理"""
        pool = self.LoggerPool(max_size=10, idle_timeout=1)  # 1秒超时

        # 创建日志器
        logger1 = pool.get_logger("cleanup_test_1")
        logger2 = pool.get_logger("cleanup_test_2")

        # 立即获取状态
        initial_stats = pool.get_stats()
        assert initial_stats["pool_size"] == 2

        # 等待超时
        import time
        time.sleep(1.1)

        # 清理空闲日志器
        removed_count = pool.cleanup_idle_loggers()
        assert removed_count == 2

        # 验证清理结果
        final_stats = pool.get_stats()
        assert final_stats["pool_size"] == 0

    def test_pool_status_health_states(self):
        """测试池状态健康状态"""
        pool = self.LoggerPool(max_size=10)

        # 空池应该是健康状态
        status = pool.get_pool_status()
        assert status["health_status"] == "healthy"
        assert status["utilization_rate"] == 0.0

        # 添加一些日志器但不超过70%
        for i in range(6):
            pool.get_logger(f"health_test_{i}")

        status = pool.get_pool_status()
        assert status["health_status"] == "healthy"
        assert status["utilization_rate"] == 0.6

        # 添加更多日志器超过70%
        for i in range(3):
            pool.get_logger(f"warning_test_{i}")

        status = pool.get_pool_status()
        assert status["health_status"] == "warning"
        assert status["utilization_rate"] == 0.9

        # 添加更多日志器超过90%
        pool.get_logger("critical_test")
        status = pool.get_pool_status()
        assert status["health_status"] == "critical"
        assert status["utilization_rate"] == 1.0

    def test_logger_release_functionality(self):
        """测试日志器释放功能"""
        pool = self.LoggerPool()

        # 获取日志器
        logger = pool.get_logger("release_test")
        initial_stats = pool._stats["release_test"]

        # 释放日志器（更新最后使用时间）
        pool.release_logger("release_test")
        updated_stats = pool._stats["release_test"]

        # 验证最后使用时间被更新
        assert updated_stats.last_used_time >= initial_stats.last_used_time

    def test_pool_shutdown_functionality(self):
        """测试池关闭功能"""
        pool = self.LoggerPool()

        # 创建一些日志器
        for i in range(3):
            pool.get_logger(f"shutdown_test_{i}")

        # 验证日志器存在
        initial_stats = pool.get_stats()
        assert initial_stats["pool_size"] == 3

        # 关闭池
        pool.shutdown()

        # 验证池被清空
        final_stats = pool.get_stats()
        assert final_stats["pool_size"] == 0
        assert final_stats["state"] == "failed"
        assert len(final_stats["logger_stats"]) == 0

    def test_concurrent_access_safety(self):
        """测试并发访问安全性"""
        import threading
        import time

        pool = self.LoggerPool(max_size=50)
        results = []
        errors = []

        def worker(worker_id):
            """工作线程"""
            try:
                for i in range(10):
                    logger = pool.get_logger(f"concurrent_{worker_id}_{i}")
                    time.sleep(0.001)  # 小延迟以增加并发性
                results.append(f"worker_{worker_id}_done")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 开始并发执行
        for thread in threads:
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 5
        assert len(errors) == 0

        # 验证池状态
        final_stats = pool.get_stats()
        assert final_stats["pool_size"] <= 50  # 不超过最大大小

    def test_singleton_pattern(self):
        """测试单例模式"""
        # 获取实例
        instance1 = self.LoggerPool.get_instance()
        instance2 = self.LoggerPool.get_instance()

        # 验证是同一个实例
        assert instance1 is instance2
        assert isinstance(instance1, self.LoggerPool)

    def test_logger_creation_failure_handling(self):
        """测试日志器创建失败处理"""
        pool = self.LoggerPool()

        # Mock _create_logger 方法使其返回None
        original_create = pool._create_logger
        pool._create_logger = lambda *args, **kwargs: None

        try:
            # 尝试创建日志器
            logger = pool.get_logger("failure_test")

            # 应该返回None
            assert logger is None

            # 验证没有添加到池中
            assert "failure_test" not in pool._loggers
            assert "failure_test" not in pool._stats

        finally:
            # 恢复原始方法
            pool._create_logger = original_create

    def test_stats_update_on_reuse(self):
        """测试重用时统计信息更新"""
        pool = self.LoggerPool()

        # 第一次获取
        logger1 = pool.get_logger("reuse_test")
        initial_stats = pool._stats["reuse_test"].use_count

        # 再次获取（重用）
        logger2 = pool.get_logger("reuse_test")
        updated_stats = pool._stats["reuse_test"].use_count

        # 验证使用计数增加
        assert updated_stats == initial_stats + 1
        assert logger1 is logger2  # 应该是同一个实例