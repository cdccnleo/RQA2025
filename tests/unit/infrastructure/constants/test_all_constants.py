"""
基础设施层测试文件

自动修复导入问题
"""

import pytest
import sys
import importlib
from pathlib import Path

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 直接导入，避免pytest钩子问题
import sys
import os

# 直接设置正确的Python路径
import sys
import os

# 计算项目根目录路径并添加到sys.path
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)  # tests/unit/infrastructure/constants
unit_dir = os.path.dirname(tests_dir)     # tests/unit/infrastructure
infra_dir = os.path.dirname(unit_dir)     # tests/unit
unit_tests_dir = os.path.dirname(infra_dir)  # tests
project_tests_dir = os.path.dirname(unit_tests_dir)  # 项目根目录

src_dir = os.path.join(project_tests_dir, 'src')

# 确保src目录在路径的最前面
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 现在可以正常导入
import src.infrastructure.constants as constants_module
ConfigConstants = constants_module.ConfigConstants
FormatConstants = constants_module.FormatConstants
HTTPConstants = constants_module.HTTPConstants
SizeConstants = constants_module.SizeConstants
TimeConstants = constants_module.TimeConstants
ThresholdConstants = constants_module.ThresholdConstants
PerformanceConstants = constants_module.PerformanceConstants


class TestConfigConstants:
    """配置常量测试"""
    
    def test_cache_size_constants(self):
        """测试缓存大小常量"""
        assert ConfigConstants.DEFAULT_CACHE_SIZE == 1024
        assert ConfigConstants.MAX_CACHE_SIZE == 1048576
        assert ConfigConstants.MIN_CACHE_SIZE == 64
        assert ConfigConstants.MIN_CACHE_SIZE < ConfigConstants.DEFAULT_CACHE_SIZE < ConfigConstants.MAX_CACHE_SIZE
    
    def test_ttl_constants(self):
        """测试TTL常量"""
        assert ConfigConstants.DEFAULT_TTL == 3600
        assert ConfigConstants.MIN_TTL == 60
        assert ConfigConstants.MAX_TTL == 86400
        assert ConfigConstants.MIN_TTL < ConfigConstants.DEFAULT_TTL < ConfigConstants.MAX_TTL
    
    def test_cleanup_constants(self):
        """测试清理常量"""
        assert ConfigConstants.CLEANUP_INTERVAL == 300
        assert ConfigConstants.CLEANUP_BATCH_SIZE == 1000
        assert ConfigConstants.CLEANUP_INTERVAL > 0
        assert ConfigConstants.CLEANUP_BATCH_SIZE > 0
    
    def test_timeout_constants(self):
        """测试超时常量"""
        assert ConfigConstants.REQUEST_TIMEOUT == 30
        assert ConfigConstants.CONNECT_TIMEOUT == 10
        assert ConfigConstants.READ_TIMEOUT == 30
        assert ConfigConstants.CONNECT_TIMEOUT <= ConfigConstants.REQUEST_TIMEOUT
    
    def test_retry_constants(self):
        """测试重试常量"""
        assert ConfigConstants.MAX_RETRIES == 3
        assert ConfigConstants.RETRY_DELAY == 1
        assert ConfigConstants.RETRY_BACKOFF_FACTOR == 2
        assert ConfigConstants.MAX_RETRIES > 0
        assert ConfigConstants.RETRY_BACKOFF_FACTOR >= 1
    
    def test_thread_pool_constants(self):
        """测试线程池常量"""
        assert ConfigConstants.MIN_THREAD_POOL_SIZE == 2
        assert ConfigConstants.MAX_THREAD_POOL_SIZE == 32
        assert ConfigConstants.DEFAULT_THREAD_POOL_SIZE == 10
        assert ConfigConstants.MIN_THREAD_POOL_SIZE <= ConfigConstants.DEFAULT_THREAD_POOL_SIZE <= ConfigConstants.MAX_THREAD_POOL_SIZE
    
    def test_cache_ttl_strategies(self):
        """测试缓存TTL策略常量"""
        assert ConfigConstants.CACHE_TTL_SHORT == 300
        assert ConfigConstants.CACHE_TTL_MEDIUM == 1800
        assert ConfigConstants.CACHE_TTL_LONG == 3600
        assert ConfigConstants.CACHE_TTL_EXTENDED == 86400
        # 验证递增顺序
        assert ConfigConstants.CACHE_TTL_SHORT < ConfigConstants.CACHE_TTL_MEDIUM
        assert ConfigConstants.CACHE_TTL_MEDIUM < ConfigConstants.CACHE_TTL_LONG
        assert ConfigConstants.CACHE_TTL_LONG < ConfigConstants.CACHE_TTL_EXTENDED
    
    def test_version_retention_constants(self):
        """测试版本保留常量"""
        assert ConfigConstants.VERSION_RETENTION_DAYS == 30
        assert ConfigConstants.VERSION_MAX_KEEP == 100
        assert ConfigConstants.VERSION_RETENTION_DAYS > 0
        assert ConfigConstants.VERSION_MAX_KEEP > 0


class TestFormatConstants:
    """格式化常量测试"""
    
    def test_separator_lengths(self):
        """测试分隔符长度常量"""
        assert FormatConstants.SEPARATOR_LENGTH_SHORT == 40
        assert FormatConstants.SEPARATOR_LENGTH_MEDIUM == 50
        assert FormatConstants.SEPARATOR_LENGTH_LONG == 60
        assert FormatConstants.SEPARATOR_LENGTH_FULL == 80
        assert FormatConstants.SEPARATOR_LENGTH_WIDE == 100
        # 验证递增顺序
        lengths = [
            FormatConstants.SEPARATOR_LENGTH_SHORT,
            FormatConstants.SEPARATOR_LENGTH_MEDIUM,
            FormatConstants.SEPARATOR_LENGTH_LONG,
            FormatConstants.SEPARATOR_LENGTH_FULL,
            FormatConstants.SEPARATOR_LENGTH_WIDE,
        ]
        assert lengths == sorted(lengths)
    
    def test_separator_characters(self):
        """测试分隔符字符常量"""
        assert FormatConstants.SEPARATOR_CHAR_DASH == '-'
        assert FormatConstants.SEPARATOR_CHAR_EQUAL == '='
        assert FormatConstants.SEPARATOR_CHAR_STAR == '*'
        assert FormatConstants.SEPARATOR_CHAR_HASH == '#'
        # 验证都是单字符
        assert len(FormatConstants.SEPARATOR_CHAR_DASH) == 1
        assert len(FormatConstants.SEPARATOR_CHAR_EQUAL) == 1
    
    def test_indent_levels(self):
        """测试缩进级别常量"""
        assert FormatConstants.INDENT_LEVEL_0 == 0
        assert FormatConstants.INDENT_LEVEL_1 == 2
        assert FormatConstants.INDENT_LEVEL_2 == 4
        assert FormatConstants.INDENT_LEVEL_3 == 6
        assert FormatConstants.INDENT_LEVEL_4 == 8
        # 验证递增且步长为2
        for i in range(4):
            current = getattr(FormatConstants, f'INDENT_LEVEL_{i}')
            next_level = getattr(FormatConstants, f'INDENT_LEVEL_{i+1}')
            assert next_level - current == 2
    
    def test_json_format_constants(self):
        """测试JSON格式化常量"""
        assert FormatConstants.JSON_INDENT == 2
        assert FormatConstants.JSON_SEPARATORS == (',', ': ')
        assert FormatConstants.JSON_ENSURE_ASCII is False
    
    def test_log_format_constants(self):
        """测试日志格式常量"""
        assert FormatConstants.LOG_MAX_MESSAGE_LENGTH == 1000
        assert FormatConstants.LOG_MAX_STACKTRACE_DEPTH == 10
        assert FormatConstants.LOG_MAX_MESSAGE_LENGTH > 0
        assert FormatConstants.LOG_MAX_STACKTRACE_DEPTH > 0
    
    def test_table_column_widths(self):
        """测试表格列宽常量"""
        assert FormatConstants.TABLE_COLUMN_WIDTH_SMALL == 10
        assert FormatConstants.TABLE_COLUMN_WIDTH_MEDIUM == 20
        assert FormatConstants.TABLE_COLUMN_WIDTH_LARGE == 30
        assert FormatConstants.TABLE_COLUMN_WIDTH_XLARGE == 50
    
    def test_truncate_constants(self):
        """测试截断常量"""
        assert FormatConstants.TRUNCATE_LENGTH_SHORT == 50
        assert FormatConstants.TRUNCATE_LENGTH_MEDIUM == 100
        assert FormatConstants.TRUNCATE_LENGTH_LONG == 200
        assert FormatConstants.TRUNCATE_SUFFIX == '...'
    
    def test_encoding_constants(self):
        """测试编码常量"""
        assert FormatConstants.DEFAULT_ENCODING == 'utf-8'
        assert FormatConstants.FALLBACK_ENCODING == 'latin-1'


class TestHTTPConstants:
    """HTTP常量测试"""
    
    def test_success_status_codes(self):
        """测试成功状态码"""
        assert HTTPConstants.OK == 200
        assert HTTPConstants.CREATED == 201
        assert HTTPConstants.ACCEPTED == 202
        assert HTTPConstants.NO_CONTENT == 204
        # 验证所有成功码在2xx范围
        assert 200 <= HTTPConstants.OK < 300
        assert 200 <= HTTPConstants.CREATED < 300
    
    def test_redirect_status_codes(self):
        """测试重定向状态码"""
        assert HTTPConstants.MOVED_PERMANENTLY == 301
        assert HTTPConstants.FOUND == 302
        assert HTTPConstants.NOT_MODIFIED == 304
        # 验证所有重定向码在3xx范围
        assert 300 <= HTTPConstants.MOVED_PERMANENTLY < 400
        assert 300 <= HTTPConstants.FOUND < 400
    
    def test_client_error_status_codes(self):
        """测试客户端错误状态码"""
        assert HTTPConstants.BAD_REQUEST == 400
        assert HTTPConstants.UNAUTHORIZED == 401
        assert HTTPConstants.FORBIDDEN == 403
        assert HTTPConstants.NOT_FOUND == 404
        assert HTTPConstants.METHOD_NOT_ALLOWED == 405
        assert HTTPConstants.CONFLICT == 409
        assert HTTPConstants.TOO_MANY_REQUESTS == 429
        # 验证所有客户端错误码在4xx范围
        assert 400 <= HTTPConstants.BAD_REQUEST < 500
        assert 400 <= HTTPConstants.UNAUTHORIZED < 500
    
    def test_server_error_status_codes(self):
        """测试服务器错误状态码"""
        assert HTTPConstants.INTERNAL_SERVER_ERROR == 500
        assert HTTPConstants.NOT_IMPLEMENTED == 501
        assert HTTPConstants.BAD_GATEWAY == 502
        assert HTTPConstants.SERVICE_UNAVAILABLE == 503
        assert HTTPConstants.GATEWAY_TIMEOUT == 504
        # 验证所有服务器错误码在5xx范围
        assert 500 <= HTTPConstants.INTERNAL_SERVER_ERROR < 600
        assert 500 <= HTTPConstants.BAD_GATEWAY < 600
    
    def test_http_methods(self):
        """测试HTTP方法常量"""
        assert HTTPConstants.METHOD_GET == 'GET'
        assert HTTPConstants.METHOD_POST == 'POST'
        assert HTTPConstants.METHOD_PUT == 'PUT'
        assert HTTPConstants.METHOD_DELETE == 'DELETE'
        assert HTTPConstants.METHOD_PATCH == 'PATCH'
        assert HTTPConstants.METHOD_HEAD == 'HEAD'
        assert HTTPConstants.METHOD_OPTIONS == 'OPTIONS'
        # 验证都是大写
        for method in ['METHOD_GET', 'METHOD_POST', 'METHOD_PUT', 'METHOD_DELETE']:
            value = getattr(HTTPConstants, method)
            assert value == value.upper()
    
    def test_content_types(self):
        """测试Content-Type常量"""
        assert HTTPConstants.CONTENT_TYPE_JSON == 'application/json'
        assert HTTPConstants.CONTENT_TYPE_XML == 'application/xml'
        assert HTTPConstants.CONTENT_TYPE_FORM == 'application/x-www-form-urlencoded'
        assert HTTPConstants.CONTENT_TYPE_MULTIPART == 'multipart/form-data'
        assert HTTPConstants.CONTENT_TYPE_TEXT == 'text/plain'
        assert HTTPConstants.CONTENT_TYPE_HTML == 'text/html'
        # 验证JSON是最常用的
        assert 'json' in HTTPConstants.CONTENT_TYPE_JSON
    
    def test_default_ports(self):
        """测试默认端口常量"""
        assert HTTPConstants.DEFAULT_HTTP_PORT == 80
        assert HTTPConstants.DEFAULT_HTTPS_PORT == 443
        assert HTTPConstants.DEFAULT_API_PORT == 5000
        assert HTTPConstants.DEFAULT_ADMIN_PORT == 8080
        # 验证端口号在有效范围
        assert 1 <= HTTPConstants.DEFAULT_HTTP_PORT <= 65535
        assert 1 <= HTTPConstants.DEFAULT_HTTPS_PORT <= 65535


class TestPerformanceConstants:
    """性能常量测试"""
    
    def test_gc_thresholds(self):
        """测试GC阈值常量"""
        assert PerformanceConstants.GC_THRESHOLD_0 == 700
        assert PerformanceConstants.GC_THRESHOLD_1 == 10
        assert PerformanceConstants.GC_THRESHOLD_2 == 10
        assert PerformanceConstants.GC_THRESHOLD_0 > 0
    
    def test_benchmark_constants(self):
        """测试性能基准常量"""
        assert PerformanceConstants.BENCHMARK_EXCELLENT == 10
        assert PerformanceConstants.BENCHMARK_GOOD == 50
        assert PerformanceConstants.BENCHMARK_ACCEPTABLE == 100
        assert PerformanceConstants.BENCHMARK_SLOW == 500
        assert PerformanceConstants.BENCHMARK_VERY_SLOW == 1000
        # 验证递增顺序
        benchmarks = [
            PerformanceConstants.BENCHMARK_EXCELLENT,
            PerformanceConstants.BENCHMARK_GOOD,
            PerformanceConstants.BENCHMARK_ACCEPTABLE,
            PerformanceConstants.BENCHMARK_SLOW,
            PerformanceConstants.BENCHMARK_VERY_SLOW,
        ]
        assert benchmarks == sorted(benchmarks)
    
    def test_concurrency_limits(self):
        """测试并发限制常量"""
        assert PerformanceConstants.MAX_CONCURRENT_REQUESTS == 1000
        assert PerformanceConstants.MAX_CONCURRENT_CONNECTIONS == 500
        assert PerformanceConstants.MAX_CONCURRENT_THREADS == 100
        # 验证合理性
        assert PerformanceConstants.MAX_CONCURRENT_THREADS < PerformanceConstants.MAX_CONCURRENT_CONNECTIONS
    
    def test_batch_size_constants(self):
        """测试批处理大小常量"""
        assert PerformanceConstants.OPTIMAL_BATCH_SIZE == 100
        assert PerformanceConstants.MAX_BATCH_SIZE == 1000
        assert PerformanceConstants.MIN_BATCH_SIZE == 10
        assert PerformanceConstants.MIN_BATCH_SIZE <= PerformanceConstants.OPTIMAL_BATCH_SIZE <= PerformanceConstants.MAX_BATCH_SIZE
    
    def test_cache_performance_constants(self):
        """测试缓存性能常量"""
        assert PerformanceConstants.TARGET_CACHE_HIT_RATE == 80.0
        assert PerformanceConstants.MIN_ACCEPTABLE_HIT_RATE == 60.0
        assert 0 <= PerformanceConstants.MIN_ACCEPTABLE_HIT_RATE <= 100
        assert 0 <= PerformanceConstants.TARGET_CACHE_HIT_RATE <= 100
    
    def test_database_performance_constants(self):
        """测试数据库性能常量"""
        assert PerformanceConstants.DB_QUERY_TIMEOUT == 30
        assert PerformanceConstants.DB_SLOW_QUERY_THRESHOLD == 1000
        assert PerformanceConstants.DB_CONNECTION_POOL_SIZE == 20
        assert PerformanceConstants.DB_QUERY_TIMEOUT > 0
        assert PerformanceConstants.DB_CONNECTION_POOL_SIZE > 0
    
    def test_latency_targets(self):
        """测试延迟目标常量"""
        assert PerformanceConstants.LATENCY_P50 == 50
        assert PerformanceConstants.LATENCY_P90 == 100
        assert PerformanceConstants.LATENCY_P95 == 200
        assert PerformanceConstants.LATENCY_P99 == 500
        # 验证百分位递增
        assert PerformanceConstants.LATENCY_P50 < PerformanceConstants.LATENCY_P90
        assert PerformanceConstants.LATENCY_P90 < PerformanceConstants.LATENCY_P95
        assert PerformanceConstants.LATENCY_P95 < PerformanceConstants.LATENCY_P99
    
    def test_throughput_targets(self):
        """测试吞吐量目标常量"""
        assert PerformanceConstants.THROUGHPUT_LOW == 100
        assert PerformanceConstants.THROUGHPUT_MEDIUM == 500
        assert PerformanceConstants.THROUGHPUT_HIGH == 1000
        assert PerformanceConstants.THROUGHPUT_VERY_HIGH == 5000
        # 验证递增顺序
        throughputs = [
            PerformanceConstants.THROUGHPUT_LOW,
            PerformanceConstants.THROUGHPUT_MEDIUM,
            PerformanceConstants.THROUGHPUT_HIGH,
            PerformanceConstants.THROUGHPUT_VERY_HIGH,
        ]
        assert throughputs == sorted(throughputs)
    
    def test_code_quality_targets(self):
        """测试代码质量目标常量"""
        assert PerformanceConstants.CODE_COVERAGE_TARGET == 85.0
        assert PerformanceConstants.DUPLICATE_CODE_THRESHOLD == 5.0
        assert PerformanceConstants.COMPLEXITY_THRESHOLD == 15
        assert PerformanceConstants.MAX_FUNCTION_LENGTH == 50
        assert PerformanceConstants.MAX_CLASS_LENGTH == 300
        assert PerformanceConstants.MAX_PARAMETERS == 5


class TestSizeConstants:
    """大小常量测试"""
    
    def test_basic_units(self):
        """测试基础单位常量"""
        assert SizeConstants.BYTE == 1
        assert SizeConstants.KB == 1024
        assert SizeConstants.MB == 1024 * 1024
        assert SizeConstants.GB == 1024 * 1024 * 1024
        assert SizeConstants.TB == 1024 * 1024 * 1024 * 1024
        # 验证递增关系
        assert SizeConstants.BYTE < SizeConstants.KB
        assert SizeConstants.KB < SizeConstants.MB
        assert SizeConstants.MB < SizeConstants.GB
        assert SizeConstants.GB < SizeConstants.TB
    
    def test_unit_conversion(self):
        """测试单位转换"""
        assert SizeConstants.KB == 1024 * SizeConstants.BYTE
        assert SizeConstants.MB == 1024 * SizeConstants.KB
        assert SizeConstants.GB == 1024 * SizeConstants.MB
        assert SizeConstants.TB == 1024 * SizeConstants.GB
    
    def test_cache_sizes(self):
        """测试缓存大小常量"""
        assert SizeConstants.CACHE_SIZE_TINY == 64
        assert SizeConstants.CACHE_SIZE_SMALL == 1024
        assert SizeConstants.CACHE_SIZE_MEDIUM == 10240
        assert SizeConstants.CACHE_SIZE_LARGE == 102400
        assert SizeConstants.CACHE_SIZE_XLARGE == 1048576
        # 验证递增顺序
        sizes = [
            SizeConstants.CACHE_SIZE_TINY,
            SizeConstants.CACHE_SIZE_SMALL,
            SizeConstants.CACHE_SIZE_MEDIUM,
            SizeConstants.CACHE_SIZE_LARGE,
            SizeConstants.CACHE_SIZE_XLARGE,
        ]
        assert sizes == sorted(sizes)
    
    def test_file_size_limits(self):
        """测试文件大小限制常量"""
        assert SizeConstants.MAX_UPLOAD_SIZE == 10 * SizeConstants.MB
        assert SizeConstants.MAX_CONFIG_FILE_SIZE == 10 * SizeConstants.MB
        assert SizeConstants.MAX_LOG_FILE_SIZE == 100 * SizeConstants.MB
        assert SizeConstants.MAX_BACKUP_SIZE == 1 * SizeConstants.GB
        # 验证合理性
        assert SizeConstants.MAX_CONFIG_FILE_SIZE < SizeConstants.MAX_LOG_FILE_SIZE
        assert SizeConstants.MAX_LOG_FILE_SIZE < SizeConstants.MAX_BACKUP_SIZE
    
    def test_queue_sizes(self):
        """测试队列大小常量"""
        assert SizeConstants.QUEUE_SIZE_SMALL == 100
        assert SizeConstants.QUEUE_SIZE_MEDIUM == 1000
        assert SizeConstants.QUEUE_SIZE_LARGE == 10000
        assert SizeConstants.QUEUE_SIZE_XLARGE == 100000
        # 验证递增顺序（10倍递增）
        assert SizeConstants.QUEUE_SIZE_MEDIUM == SizeConstants.QUEUE_SIZE_SMALL * 10
        assert SizeConstants.QUEUE_SIZE_LARGE == SizeConstants.QUEUE_SIZE_MEDIUM * 10
    
    def test_batch_sizes(self):
        """测试批处理大小常量"""
        assert SizeConstants.BATCH_SIZE_SMALL == 10
        assert SizeConstants.BATCH_SIZE_MEDIUM == 50
        assert SizeConstants.BATCH_SIZE_LARGE == 100
        assert SizeConstants.BATCH_SIZE_XLARGE == 500
        assert SizeConstants.BATCH_SIZE_XXLARGE == 1000
    
    def test_database_connection_constants(self):
        """测试数据库连接常量"""
        assert SizeConstants.DB_MAX_CONNECTIONS == 100
        assert SizeConstants.DB_MIN_CONNECTIONS == 5
        assert SizeConstants.DB_DEFAULT_CONNECTIONS == 20
        assert SizeConstants.DB_MIN_CONNECTIONS <= SizeConstants.DB_DEFAULT_CONNECTIONS <= SizeConstants.DB_MAX_CONNECTIONS
    
    def test_thread_pool_constants(self):
        """测试线程池常量"""
        assert SizeConstants.THREAD_POOL_MIN == 2
        assert SizeConstants.THREAD_POOL_MAX == 32
        assert SizeConstants.THREAD_POOL_DEFAULT == 10
        assert SizeConstants.THREAD_POOL_MIN <= SizeConstants.THREAD_POOL_DEFAULT <= SizeConstants.THREAD_POOL_MAX
    
    def test_page_size_constants(self):
        """测试分页大小常量"""
        assert SizeConstants.PAGE_SIZE_SMALL == 10
        assert SizeConstants.PAGE_SIZE_MEDIUM == 20
        assert SizeConstants.PAGE_SIZE_LARGE == 50
        assert SizeConstants.PAGE_SIZE_XLARGE == 100
        assert SizeConstants.PAGE_SIZE_MAX == 1000
        # 验证最大分页不超过合理范围
        assert SizeConstants.PAGE_SIZE_MAX <= 10000
    
    def test_string_length_limits(self):
        """测试字符串长度限制常量"""
        assert SizeConstants.MAX_STRING_LENGTH_SHORT == 50
        assert SizeConstants.MAX_STRING_LENGTH_MEDIUM == 255
        assert SizeConstants.MAX_STRING_LENGTH_LONG == 1000
        assert SizeConstants.MAX_STRING_LENGTH_XLARGE == 10000


class TestConstantsConsistency:
    """常量一致性测试"""
    
    def test_cache_size_consistency_across_modules(self):
        """测试跨模块的缓存大小一致性"""
        # ConfigConstants和SizeConstants的缓存大小应该一致
        assert ConfigConstants.DEFAULT_CACHE_SIZE == SizeConstants.CACHE_SIZE_SMALL
        assert ConfigConstants.MAX_CACHE_SIZE == SizeConstants.CACHE_SIZE_XLARGE
    
    def test_thread_pool_consistency(self):
        """测试线程池常量一致性"""
        # ConfigConstants和SizeConstants的线程池大小应该一致
        assert ConfigConstants.MIN_THREAD_POOL_SIZE == SizeConstants.THREAD_POOL_MIN
        assert ConfigConstants.MAX_THREAD_POOL_SIZE == SizeConstants.THREAD_POOL_MAX
        assert ConfigConstants.DEFAULT_THREAD_POOL_SIZE == SizeConstants.THREAD_POOL_DEFAULT
    
    def test_batch_size_consistency(self):
        """测试批处理大小一致性"""
        # 验证批处理大小在合理范围内
        assert PerformanceConstants.MIN_BATCH_SIZE == SizeConstants.BATCH_SIZE_SMALL
        assert PerformanceConstants.OPTIMAL_BATCH_SIZE == SizeConstants.BATCH_SIZE_LARGE
        assert PerformanceConstants.MAX_BATCH_SIZE == SizeConstants.BATCH_SIZE_XXLARGE
    
    def test_reasonable_default_values(self):
        """测试默认值的合理性"""
        # 缓存TTL应该合理
        assert 60 <= ConfigConstants.DEFAULT_TTL <= 7200  # 1分钟到2小时
        
        # 线程池大小应该合理
        assert 1 <= ConfigConstants.DEFAULT_THREAD_POOL_SIZE <= 100
        
        # 请求超时应该合理
        assert 5 <= ConfigConstants.REQUEST_TIMEOUT <= 300  # 5秒到5分钟
        
        # 批处理大小应该合理
        assert 1 <= PerformanceConstants.OPTIMAL_BATCH_SIZE <= 10000
    
    def test_performance_thresholds_ordering(self):
        """测试性能阈值的顺序"""
        # 延迟百分位应该递增
        assert PerformanceConstants.LATENCY_P50 < PerformanceConstants.LATENCY_P90
        assert PerformanceConstants.LATENCY_P90 < PerformanceConstants.LATENCY_P95
        assert PerformanceConstants.LATENCY_P95 < PerformanceConstants.LATENCY_P99
        
        # 吞吐量级别应该递增
        assert PerformanceConstants.THROUGHPUT_LOW < PerformanceConstants.THROUGHPUT_MEDIUM
        assert PerformanceConstants.THROUGHPUT_MEDIUM < PerformanceConstants.THROUGHPUT_HIGH
        assert PerformanceConstants.THROUGHPUT_HIGH < PerformanceConstants.THROUGHPUT_VERY_HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

