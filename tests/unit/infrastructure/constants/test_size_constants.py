"""
测试大小相关常量定义

覆盖 SizeConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.size_constants import SizeConstants


class TestSizeConstants:
    """SizeConstants 单元测试"""

    def test_basic_units(self):
        """测试基础单位常量"""
        assert SizeConstants.BYTE == 1
        assert SizeConstants.KB == 1024
        assert SizeConstants.MB == 1024 * 1024
        assert SizeConstants.GB == 1024 * 1024 * 1024
        assert SizeConstants.TB == 1024 * 1024 * 1024 * 1024

    def test_cache_sizes(self):
        """测试缓存大小常量"""
        assert SizeConstants.CACHE_SIZE_TINY == 64
        assert SizeConstants.CACHE_SIZE_SMALL == 1024
        assert SizeConstants.CACHE_SIZE_MEDIUM == 10240
        assert SizeConstants.CACHE_SIZE_LARGE == 102400
        assert SizeConstants.CACHE_SIZE_XLARGE == 1048576

    def test_file_size_limits(self):
        """测试文件大小限制常量"""
        assert SizeConstants.MAX_UPLOAD_SIZE == 10 * SizeConstants.MB
        assert SizeConstants.MAX_CONFIG_FILE_SIZE == 10 * SizeConstants.MB
        assert SizeConstants.MAX_LOG_FILE_SIZE == 100 * SizeConstants.MB
        assert SizeConstants.MAX_BACKUP_SIZE == 1 * SizeConstants.GB

    def test_queue_sizes(self):
        """测试队列大小常量"""
        assert SizeConstants.QUEUE_SIZE_SMALL == 100
        assert SizeConstants.QUEUE_SIZE_MEDIUM == 1000
        assert SizeConstants.QUEUE_SIZE_LARGE == 10000
        assert SizeConstants.QUEUE_SIZE_XLARGE == 100000

    def test_batch_sizes(self):
        """测试批处理大小常量"""
        assert SizeConstants.BATCH_SIZE_SMALL == 10
        assert SizeConstants.BATCH_SIZE_MEDIUM == 50
        assert SizeConstants.BATCH_SIZE_LARGE == 100
        assert SizeConstants.BATCH_SIZE_XLARGE == 500
        assert SizeConstants.BATCH_SIZE_XXLARGE == 1000

    def test_memory_thresholds(self):
        """测试内存阈值常量"""
        assert SizeConstants.MEMORY_SMALL_OBJECT == 1024
        assert SizeConstants.MEMORY_MEDIUM_OBJECT == 10240
        assert SizeConstants.MEMORY_LARGE_OBJECT == 10000
        assert SizeConstants.MEMORY_XLARGE_OBJECT == 1048576

    def test_large_object_compatibility(self):
        """测试向后兼容的大对象常量"""
        assert SizeConstants.LARGE_OBJECT == 10000

    def test_database_constants(self):
        """测试数据库相关常量"""
        assert SizeConstants.DB_MAX_CONNECTIONS == 100
        assert SizeConstants.DB_MIN_CONNECTIONS == 5
        assert SizeConstants.DB_DEFAULT_CONNECTIONS == 20

    def test_thread_pool_constants(self):
        """测试线程池大小常量"""
        assert SizeConstants.THREAD_POOL_MIN == 2
        assert SizeConstants.THREAD_POOL_MAX == 32
        assert SizeConstants.THREAD_POOL_DEFAULT == 10

    def test_page_sizes(self):
        """测试分页大小常量"""
        assert SizeConstants.PAGE_SIZE_SMALL == 10
        assert SizeConstants.PAGE_SIZE_MEDIUM == 20
        assert SizeConstants.PAGE_SIZE_LARGE == 50
        assert SizeConstants.PAGE_SIZE_XLARGE == 100
        assert SizeConstants.PAGE_SIZE_MAX == 1000

    def test_string_length_limits(self):
        """测试字符串长度限制常量"""
        assert SizeConstants.MAX_STRING_LENGTH_SHORT == 50
        assert SizeConstants.MAX_STRING_LENGTH_MEDIUM == 255
        assert SizeConstants.MAX_STRING_LENGTH_LONG == 1000
        assert SizeConstants.MAX_STRING_LENGTH_XLARGE == 10000

    def test_unit_progression(self):
        """测试单位递增规律"""
        # 验证KB = BYTE * 1024
        assert SizeConstants.KB == SizeConstants.BYTE * 1024
        # 验证MB = KB * 1024
        assert SizeConstants.MB == SizeConstants.KB * 1024
        # 验证GB = MB * 1024
        assert SizeConstants.GB == SizeConstants.MB * 1024
        # 验证TB = GB * 1024
        assert SizeConstants.TB == SizeConstants.GB * 1024

    def test_cache_size_progression(self):
        """测试缓存大小递增规律"""
        cache_sizes = [
            SizeConstants.CACHE_SIZE_TINY,
            SizeConstants.CACHE_SIZE_SMALL,
            SizeConstants.CACHE_SIZE_MEDIUM,
            SizeConstants.CACHE_SIZE_LARGE,
            SizeConstants.CACHE_SIZE_XLARGE
        ]

        # 验证递增顺序
        for i in range(len(cache_sizes) - 1):
            assert cache_sizes[i] < cache_sizes[i + 1]

    def test_queue_size_progression(self):
        """测试队列大小递增规律"""
        queue_sizes = [
            SizeConstants.QUEUE_SIZE_SMALL,
            SizeConstants.QUEUE_SIZE_MEDIUM,
            SizeConstants.QUEUE_SIZE_LARGE,
            SizeConstants.QUEUE_SIZE_XLARGE
        ]

        # 验证递增顺序
        for i in range(len(queue_sizes) - 1):
            assert queue_sizes[i] < queue_sizes[i + 1]

    def test_batch_size_progression(self):
        """测试批处理大小递增规律"""
        batch_sizes = [
            SizeConstants.BATCH_SIZE_SMALL,
            SizeConstants.BATCH_SIZE_MEDIUM,
            SizeConstants.BATCH_SIZE_LARGE,
            SizeConstants.BATCH_SIZE_XLARGE,
            SizeConstants.BATCH_SIZE_XXLARGE
        ]

        # 验证递增顺序
        for i in range(len(batch_sizes) - 1):
            assert batch_sizes[i] < batch_sizes[i + 1]

    def test_memory_object_sizes(self):
        """测试内存对象大小定义"""
        # 验证各个大小级别
        assert SizeConstants.MEMORY_SMALL_OBJECT == 1024  # 1KB
        assert SizeConstants.MEMORY_MEDIUM_OBJECT == 10240  # 10KB
        assert SizeConstants.MEMORY_LARGE_OBJECT == 10000  # ~10KB
        assert SizeConstants.MEMORY_XLARGE_OBJECT == 1048576  # 1MB

        # 验证小对象小于中对象和大对象
        assert SizeConstants.MEMORY_SMALL_OBJECT < SizeConstants.MEMORY_MEDIUM_OBJECT
        assert SizeConstants.MEMORY_SMALL_OBJECT < SizeConstants.MEMORY_LARGE_OBJECT

        # 验证超大对象远大于其他对象
        assert SizeConstants.MEMORY_XLARGE_OBJECT > SizeConstants.MEMORY_MEDIUM_OBJECT
        assert SizeConstants.MEMORY_XLARGE_OBJECT > SizeConstants.MEMORY_LARGE_OBJECT

    def test_page_size_progression(self):
        """测试分页大小递增规律"""
        page_sizes = [
            SizeConstants.PAGE_SIZE_SMALL,
            SizeConstants.PAGE_SIZE_MEDIUM,
            SizeConstants.PAGE_SIZE_LARGE,
            SizeConstants.PAGE_SIZE_XLARGE,
            SizeConstants.PAGE_SIZE_MAX
        ]

        # 验证递增顺序
        for i in range(len(page_sizes) - 1):
            assert page_sizes[i] < page_sizes[i + 1]

    def test_string_length_progression(self):
        """测试字符串长度递增规律"""
        string_lengths = [
            SizeConstants.MAX_STRING_LENGTH_SHORT,
            SizeConstants.MAX_STRING_LENGTH_MEDIUM,
            SizeConstants.MAX_STRING_LENGTH_LONG,
            SizeConstants.MAX_STRING_LENGTH_XLARGE
        ]

        # 验证递增顺序
        for i in range(len(string_lengths) - 1):
            assert string_lengths[i] < string_lengths[i + 1]

    def test_database_connection_bounds(self):
        """测试数据库连接边界"""
        assert (SizeConstants.DB_MIN_CONNECTIONS <
                SizeConstants.DB_DEFAULT_CONNECTIONS <
                SizeConstants.DB_MAX_CONNECTIONS)

    def test_thread_pool_bounds(self):
        """测试线程池边界"""
        assert (SizeConstants.THREAD_POOL_MIN <
                SizeConstants.THREAD_POOL_DEFAULT <
                SizeConstants.THREAD_POOL_MAX)

    def test_positive_values(self):
        """测试所有常量都是正值"""
        numeric_constants = [
            SizeConstants.BYTE,
            SizeConstants.KB,
            SizeConstants.MB,
            SizeConstants.GB,
            SizeConstants.TB,
            SizeConstants.CACHE_SIZE_TINY,
            SizeConstants.CACHE_SIZE_SMALL,
            SizeConstants.CACHE_SIZE_MEDIUM,
            SizeConstants.CACHE_SIZE_LARGE,
            SizeConstants.CACHE_SIZE_XLARGE,
            SizeConstants.MAX_UPLOAD_SIZE,
            SizeConstants.MAX_CONFIG_FILE_SIZE,
            SizeConstants.MAX_LOG_FILE_SIZE,
            SizeConstants.MAX_BACKUP_SIZE,
            SizeConstants.QUEUE_SIZE_SMALL,
            SizeConstants.QUEUE_SIZE_MEDIUM,
            SizeConstants.QUEUE_SIZE_LARGE,
            SizeConstants.QUEUE_SIZE_XLARGE,
            SizeConstants.BATCH_SIZE_SMALL,
            SizeConstants.BATCH_SIZE_MEDIUM,
            SizeConstants.BATCH_SIZE_LARGE,
            SizeConstants.BATCH_SIZE_XLARGE,
            SizeConstants.BATCH_SIZE_XXLARGE,
            SizeConstants.MEMORY_SMALL_OBJECT,
            SizeConstants.MEMORY_MEDIUM_OBJECT,
            SizeConstants.MEMORY_LARGE_OBJECT,
            SizeConstants.MEMORY_XLARGE_OBJECT,
            SizeConstants.LARGE_OBJECT,
            SizeConstants.DB_MAX_CONNECTIONS,
            SizeConstants.DB_MIN_CONNECTIONS,
            SizeConstants.DB_DEFAULT_CONNECTIONS,
            SizeConstants.THREAD_POOL_MIN,
            SizeConstants.THREAD_POOL_MAX,
            SizeConstants.THREAD_POOL_DEFAULT,
            SizeConstants.PAGE_SIZE_SMALL,
            SizeConstants.PAGE_SIZE_MEDIUM,
            SizeConstants.PAGE_SIZE_LARGE,
            SizeConstants.PAGE_SIZE_XLARGE,
            SizeConstants.PAGE_SIZE_MAX,
            SizeConstants.MAX_STRING_LENGTH_SHORT,
            SizeConstants.MAX_STRING_LENGTH_MEDIUM,
            SizeConstants.MAX_STRING_LENGTH_LONG,
            SizeConstants.MAX_STRING_LENGTH_XLARGE
        ]

        for constant in numeric_constants:
            assert constant > 0, f"Constant {constant} should be positive"

    def test_file_size_calculations(self):
        """测试文件大小计算"""
        # 验证文件大小是以MB和GB为单位计算的
        assert SizeConstants.MAX_UPLOAD_SIZE == 10 * 1024 * 1024
        assert SizeConstants.MAX_CONFIG_FILE_SIZE == 10 * 1024 * 1024
        assert SizeConstants.MAX_LOG_FILE_SIZE == 100 * 1024 * 1024
        assert SizeConstants.MAX_BACKUP_SIZE == 1 * 1024 * 1024 * 1024

    def test_memory_object_consistency(self):
        """测试内存对象大小一致性"""
        # 验证LARGE_OBJECT与MEMORY_LARGE_OBJECT一致
        assert SizeConstants.LARGE_OBJECT == SizeConstants.MEMORY_LARGE_OBJECT

    def test_reasonable_defaults(self):
        """测试合理的默认值"""
        # 数据库默认连接数应该在合理范围内
        assert 5 <= SizeConstants.DB_DEFAULT_CONNECTIONS <= 50

        # 线程池默认大小应该在合理范围内
        assert 4 <= SizeConstants.THREAD_POOL_DEFAULT <= 20

        # 分页默认大小应该在合理范围内
        assert 10 <= SizeConstants.PAGE_SIZE_MEDIUM <= 50

        # 缓存默认大小应该在合理范围内
        assert 1000 <= SizeConstants.CACHE_SIZE_MEDIUM <= 50000