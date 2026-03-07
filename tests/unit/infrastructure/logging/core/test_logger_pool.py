"""
Logger Pool 单元测试

测试日志器池管理功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
import threading
from datetime import datetime

from src.infrastructure.logging.core.logger_pool import (
    LoggerPoolState,
    LoggerStats,
    LoggerPool,
)


class TestLoggerPoolState:
    """测试日志器池状态枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LoggerPoolState.HEALTHY.value == "healthy"
        assert LoggerPoolState.WARNING.value == "warning"
        assert LoggerPoolState.CRITICAL.value == "critical"
        assert LoggerPoolState.FAILED.value == "failed"

    def test_enum_str(self):
        """测试枚举字符串表示"""
        assert str(LoggerPoolState.HEALTHY) == "LoggerPoolState.HEALTHY"


class TestLoggerStats:
    """测试日志器统计信息"""

    def test_init(self):
        """测试初始化"""
        stats = LoggerStats(
            logger_id="test_logger",
            created_time=1000.0,
            last_used_time=1100.0,
            use_count=5,
            error_count=1,
            memory_usage=1024
        )

        assert stats.logger_id == "test_logger"
        assert stats.created_time == 1000.0
        assert stats.last_used_time == 1100.0
        assert stats.use_count == 5
        assert stats.error_count == 1
        assert stats.memory_usage == 1024

    def test_to_dict(self):
        """测试转换为字典"""
        stats = LoggerStats(
            logger_id="test_logger",
            created_time=1000.0,
            last_used_time=1100.0,
            use_count=5,
            error_count=1,
            memory_usage=1024
        )

        result = stats.to_dict()
        expected = {
            "logger_id": "test_logger",
            "created_time": 1000.0,
            "last_used_time": 1100.0,
            "use_count": 5,
            "error_count": 1,
            "memory_usage": 1024
        }

        assert result == expected


class TestLoggerPool:
    """测试日志器池"""

    @pytest.fixture
    def logger_pool(self):
        """创建日志器池实例"""
        # 重置单例
        LoggerPool._instance = None
        pool = LoggerPool(max_size=10, idle_timeout=60)
        return pool

    def test_singleton_pattern(self):
        """测试单例模式"""
        # 重置单例
        LoggerPool._instance = None

        pool1 = LoggerPool.get_instance()
        pool2 = LoggerPool.get_instance()

        assert pool1 is pool2
        assert isinstance(pool1, LoggerPool)

    def test_init(self, logger_pool):
        """测试初始化"""
        assert logger_pool._loggers == {}
        assert logger_pool._stats == {}
        assert logger_pool._max_size == 10
        assert logger_pool._idle_timeout == 60

    def test_get_logger_new(self, logger_pool):
        """测试获取新日志器"""
        logger = logger_pool.get_logger("test_logger")

        assert logger is not None
        assert "test_logger" in logger_pool._loggers
        assert "test_logger" in logger_pool._stats

        # 验证统计信息
        stats = logger_pool._stats["test_logger"]
        assert stats.logger_id == "test_logger"
        assert stats.use_count == 1
        assert stats.error_count == 0

    def test_get_logger_existing(self, logger_pool):
        """测试获取现有日志器"""
        # 第一次获取
        logger1 = logger_pool.get_logger("existing_logger")
        stats1 = logger_pool._stats["existing_logger"]

        # 第二次获取
        logger2 = logger_pool.get_logger("existing_logger")
        stats2 = logger_pool._stats["existing_logger"]

        # 应该是同一个实例
        assert logger1 is logger2
        # 使用次数应该增加
        assert stats2.use_count == 2

    def test_get_logger_max_size(self, logger_pool):
        """测试达到最大容量限制"""
        # 设置很小的最大容量
        logger_pool._max_size = 2

        # 添加两个日志器
        logger_pool.get_logger("logger1")
        logger_pool.get_logger("logger2")

        # 第三个应该触发清理
        logger3 = logger_pool.get_logger("logger3")

        assert logger3 is not None
        # 应该清理最旧的日志器
        assert "logger1" not in logger_pool._loggers

    def test_release_logger(self, logger_pool):
        """测试释放日志器"""
        logger_pool.get_logger("test_release")

        # 获取初始的最后使用时间
        initial_time = logger_pool._stats["test_release"].last_used_time

        # 释放日志器（只是更新最后使用时间）
        logger_pool.release_logger("test_release")

        # 日志器仍然存在，但最后使用时间已更新
        assert "test_release" in logger_pool._loggers
        assert "test_release" in logger_pool._stats
        assert logger_pool._stats["test_release"].last_used_time >= initial_time

    def test_release_nonexistent_logger(self, logger_pool):
        """测试释放不存在的日志器"""
        # 不应该抛出异常
        logger_pool.release_logger("nonexistent")

    def test_remove_logger(self, logger_pool):
        """测试移除日志器"""
        logger_pool.get_logger("test_remove")

        assert "test_remove" in logger_pool._loggers

        logger_pool.remove_logger("test_remove")

        assert "test_remove" not in logger_pool._loggers
        assert "test_remove" not in logger_pool._stats

    def test_cleanup_idle_loggers(self, logger_pool):
        """测试清理空闲日志器"""
        # 设置短的空闲超时
        logger_pool._idle_timeout = 1

        logger_pool.get_logger("idle_logger")

        # 等待超过空闲超时
        time.sleep(1.1)

        logger_pool.cleanup_idle_loggers()

        # 空闲日志器应该被清理
        assert "idle_logger" not in logger_pool._loggers

    def test_get_stats(self, logger_pool):
        """测试获取统计信息"""
        logger_pool.get_logger("stats_logger1")
        logger_pool.get_logger("stats_logger2")

        stats = logger_pool.get_stats()

        assert isinstance(stats, dict)
        assert "pool_size" in stats
        assert "max_size" in stats
        assert "total_loggers_created" in stats
        assert "idle_timeout" in stats
        assert stats["pool_size"] == 2

    def test_get_pool_status_healthy(self, logger_pool):
        """测试获取健康状态"""
        # 添加一些日志器
        logger_pool.get_logger("healthy1")
        logger_pool.get_logger("healthy2")

        status = logger_pool.get_pool_status()

        assert status["health_status"] == LoggerPoolState.HEALTHY.value
        assert status["utilization_rate"] < 0.7

    def test_get_pool_status_warning(self, logger_pool):
        """测试获取警告状态"""
        # 设置小的最大容量
        logger_pool._max_size = 5

        # 添加接近容量的日志器
        for i in range(4):
            logger_pool.get_logger(f"warning{i}")

        status = logger_pool.get_pool_status()

        assert status["utilization_rate"] >= 0.7

    def test_get_pool_status_critical(self, logger_pool):
        """测试获取临界状态"""
        # 设置小的最大容量
        logger_pool._max_size = 5

        # 填满容量
        for i in range(5):
            logger_pool.get_logger(f"critical{i}")

        status = logger_pool.get_pool_status()

        assert status["utilization_rate"] >= 0.9

    def test_thread_safety(self, logger_pool):
        """测试线程安全性"""
        results = []

        def worker(worker_id):
            for i in range(10):
                logger_name = f"thread_{worker_id}_{i}"
                logger = logger_pool.get_logger(logger_name)
                results.append((worker_id, logger_name, logger is not None))

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待线程完成
        for thread in threads:
            thread.join()

        # 验证所有操作都成功
        assert len(results) == 30  # 3个线程 * 10个操作
        assert all(result[2] for result in results)  # 所有操作都返回了有效的日志器

    def test_evict_oldest(self, logger_pool):
        """测试驱逐最旧的日志器"""
        # 设置小的容量
        logger_pool._max_size = 3

        # 添加日志器
        logger_pool.get_logger("oldest")
        time.sleep(0.01)  # 确保时间差异
        logger_pool.get_logger("middle")
        time.sleep(0.01)
        logger_pool.get_logger("newest")

        # 手动调用驱逐（模拟容量满的情况）
        logger_pool._evict_oldest()

        # 最旧的应该被移除
        assert "oldest" not in logger_pool._loggers
        assert "middle" in logger_pool._loggers
        assert "newest" in logger_pool._loggers

    def test_logger_stats_update(self, logger_pool):
        """测试日志器统计信息更新"""
        logger = logger_pool.get_logger("stats_test")

        # 获取初始统计
        initial_stats = logger_pool._stats["stats_test"]
        initial_use_count = initial_stats.use_count
        initial_last_used = initial_stats.last_used_time

        # 等待一小段时间
        time.sleep(0.001)

        # 再次获取同一个日志器
        logger2 = logger_pool.get_logger("stats_test")

        # 统计应该更新
        updated_stats = logger_pool._stats["stats_test"]
        assert updated_stats.use_count == initial_use_count + 1
        assert updated_stats.last_used_time >= initial_last_used

    @patch('time.time')
    def test_logger_creation_time(self, mock_time, logger_pool):
        """测试日志器创建时间"""
        mock_time.return_value = 1234567890.0

        logger_pool.get_logger("time_test")

        stats = logger_pool._stats["time_test"]
        assert stats.created_time == 1234567890.0
        assert stats.last_used_time == 1234567890.0

    @patch('src.infrastructure.logging.core.logger_pool.UnifiedLogger')
    def test_create_logger_exception_handling(self, mock_unified_logger, logger_pool):
        """测试创建日志器时的异常处理"""
        # 模拟UnifiedLogger抛出异常
        mock_unified_logger.side_effect = Exception("Creation failed")

        logger = logger_pool.get_logger("exception_test")

        # 应该返回None
        assert logger is None
        # 统计信息不应该被创建
        assert "exception_test" not in logger_pool._stats

    def test_evict_oldest_empty_pool(self, logger_pool):
        """测试在空池上驱逐最旧的日志器"""
        # 池是空的，应该不会抛出异常
        logger_pool._evict_oldest()

        # 池仍然是空的
        assert len(logger_pool._loggers) == 0

    def test_shutdown(self, logger_pool):
        """测试关闭日志器池"""
        # 添加一些日志器
        logger_pool.get_logger("shutdown1")
        logger_pool.get_logger("shutdown2")

        assert len(logger_pool._loggers) == 2
        assert len(logger_pool._stats) == 2
        assert logger_pool._state == LoggerPoolState.HEALTHY

        # 关闭池
        logger_pool.shutdown()

        # 所有日志器和统计信息都应该被清除
        assert len(logger_pool._loggers) == 0
        assert len(logger_pool._stats) == 0
        assert logger_pool._state == LoggerPoolState.FAILED