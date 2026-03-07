"""
测试配置相关常量定义

覆盖 ConfigConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.config_constants import ConfigConstants


class TestConfigConstants:
    """ConfigConstants 单元测试"""

    def test_cache_size_constants(self):
        """测试缓存大小相关常量"""
        assert ConfigConstants.DEFAULT_CACHE_SIZE == 1024
        assert ConfigConstants.MAX_CACHE_SIZE == 1048576
        assert ConfigConstants.MIN_CACHE_SIZE == 64

    def test_ttl_constants(self):
        """测试TTL相关常量"""
        assert ConfigConstants.DEFAULT_TTL == 3600
        assert ConfigConstants.MIN_TTL == 60
        assert ConfigConstants.MAX_TTL == 86400

    def test_cleanup_constants(self):
        """测试清理相关常量"""
        assert ConfigConstants.CLEANUP_INTERVAL == 300
        assert ConfigConstants.CLEANUP_BATCH_SIZE == 1000

    def test_timeout_constants(self):
        """测试超时相关常量"""
        assert ConfigConstants.REQUEST_TIMEOUT == 30
        assert ConfigConstants.CONNECT_TIMEOUT == 10
        assert ConfigConstants.READ_TIMEOUT == 30

    def test_config_file_constants(self):
        """测试配置文件相关常量"""
        assert ConfigConstants.MAX_CONFIG_FILE_SIZE == 10485760
        assert ConfigConstants.CONFIG_WATCH_TIMEOUT == 30
        assert ConfigConstants.CONFIG_RELOAD_DELAY == 1

    def test_retry_constants(self):
        """测试重试相关常量"""
        assert ConfigConstants.MAX_RETRIES == 3
        assert ConfigConstants.RETRY_DELAY == 1
        assert ConfigConstants.RETRY_BACKOFF_FACTOR == 2

    def test_queue_size_constants(self):
        """测试队列大小相关常量"""
        assert ConfigConstants.MAX_QUEUE_SIZE == 100000
        assert ConfigConstants.MIN_QUEUE_SIZE == 100

    def test_thread_pool_constants(self):
        """测试线程池相关常量"""
        assert ConfigConstants.MIN_THREAD_POOL_SIZE == 2
        assert ConfigConstants.MAX_THREAD_POOL_SIZE == 32
        assert ConfigConstants.DEFAULT_THREAD_POOL_SIZE == 10

    def test_cache_ttl_strategy_constants(self):
        """测试缓存TTL策略相关常量"""
        assert ConfigConstants.CACHE_TTL_SHORT == 300
        assert ConfigConstants.CACHE_TTL_MEDIUM == 1800
        assert ConfigConstants.CACHE_TTL_LONG == 3600
        assert ConfigConstants.CACHE_TTL_EXTENDED == 86400

    def test_version_constants(self):
        """测试版本相关常量"""
        assert ConfigConstants.VERSION_RETENTION_DAYS == 30
        assert ConfigConstants.VERSION_MAX_KEEP == 100

    def test_constant_relationships(self):
        """测试常量之间的关系"""
        # 缓存大小关系
        assert ConfigConstants.MIN_CACHE_SIZE < ConfigConstants.DEFAULT_CACHE_SIZE < ConfigConstants.MAX_CACHE_SIZE

        # TTL关系
        assert ConfigConstants.MIN_TTL < ConfigConstants.DEFAULT_TTL < ConfigConstants.MAX_TTL

        # 队列大小关系
        assert ConfigConstants.MIN_QUEUE_SIZE < ConfigConstants.MAX_QUEUE_SIZE

        # 线程池大小关系
        assert ConfigConstants.MIN_THREAD_POOL_SIZE < ConfigConstants.DEFAULT_THREAD_POOL_SIZE < ConfigConstants.MAX_THREAD_POOL_SIZE

        # 缓存TTL策略关系
        assert (ConfigConstants.CACHE_TTL_SHORT <
                ConfigConstants.CACHE_TTL_MEDIUM <
                ConfigConstants.CACHE_TTL_LONG <
                ConfigConstants.CACHE_TTL_EXTENDED)

    def test_timeout_relationships(self):
        """测试超时常量关系"""
        # 连接超时应该小于等于读取超时
        assert ConfigConstants.CONNECT_TIMEOUT <= ConfigConstants.READ_TIMEOUT
        # 请求超时应该大于等于读取超时
        assert ConfigConstants.REQUEST_TIMEOUT >= ConfigConstants.READ_TIMEOUT

    def test_positive_values(self):
        """测试所有常量都是正值"""
        constants = [
            ConfigConstants.DEFAULT_CACHE_SIZE,
            ConfigConstants.MAX_CACHE_SIZE,
            ConfigConstants.MIN_CACHE_SIZE,
            ConfigConstants.DEFAULT_TTL,
            ConfigConstants.MIN_TTL,
            ConfigConstants.MAX_TTL,
            ConfigConstants.CLEANUP_INTERVAL,
            ConfigConstants.CLEANUP_BATCH_SIZE,
            ConfigConstants.REQUEST_TIMEOUT,
            ConfigConstants.CONNECT_TIMEOUT,
            ConfigConstants.READ_TIMEOUT,
            ConfigConstants.MAX_CONFIG_FILE_SIZE,
            ConfigConstants.CONFIG_WATCH_TIMEOUT,
            ConfigConstants.CONFIG_RELOAD_DELAY,
            ConfigConstants.MAX_RETRIES,
            ConfigConstants.RETRY_DELAY,
            ConfigConstants.RETRY_BACKOFF_FACTOR,
            ConfigConstants.MAX_QUEUE_SIZE,
            ConfigConstants.MIN_QUEUE_SIZE,
            ConfigConstants.MIN_THREAD_POOL_SIZE,
            ConfigConstants.MAX_THREAD_POOL_SIZE,
            ConfigConstants.DEFAULT_THREAD_POOL_SIZE,
            ConfigConstants.CACHE_TTL_SHORT,
            ConfigConstants.CACHE_TTL_MEDIUM,
            ConfigConstants.CACHE_TTL_LONG,
            ConfigConstants.CACHE_TTL_EXTENDED,
            ConfigConstants.VERSION_RETENTION_DAYS,
            ConfigConstants.VERSION_MAX_KEEP
        ]

        for constant in constants:
            assert constant > 0, f"Constant {constant} should be positive"

    def test_sensible_defaults(self):
        """测试默认值是否合理"""
        # 默认缓存大小应该在合理范围内
        assert 100 <= ConfigConstants.DEFAULT_CACHE_SIZE <= 10000

        # 默认TTL应该在合理范围内（1小时）
        assert 1800 <= ConfigConstants.DEFAULT_TTL <= 7200

        # 默认线程池大小应该在合理范围内
        assert 4 <= ConfigConstants.DEFAULT_THREAD_POOL_SIZE <= 20

        # 重试次数应该在合理范围内
        assert 1 <= ConfigConstants.MAX_RETRIES <= 10