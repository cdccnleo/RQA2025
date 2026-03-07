"""
测试日志核心常量定义

覆盖 constants.py 中的基本常量
"""

from src.infrastructure.logging.core.constants import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
    LOG_LEVEL_VALUES,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_DIR,
    LOG_FILE_EXTENSION,
    DEFAULT_ENCODING,
    HTTP_OK,
    DEFAULT_PAGE_SIZE
)


class TestConstants:
    """常量测试"""

    def test_log_level_constants(self):
        """测试日志级别常量"""
        assert LOG_LEVEL_DEBUG == "DEBUG"
        assert LOG_LEVEL_INFO == "INFO"
        assert LOG_LEVEL_WARNING == "WARNING"
        assert LOG_LEVEL_ERROR == "ERROR"
        assert LOG_LEVEL_CRITICAL == "CRITICAL"

    def test_log_level_values(self):
        """测试日志级别值映射"""
        assert isinstance(LOG_LEVEL_VALUES, dict)
        assert LOG_LEVEL_VALUES[LOG_LEVEL_DEBUG] == 10
        assert LOG_LEVEL_VALUES[LOG_LEVEL_INFO] == 20

    def test_file_constants(self):
        """测试文件常量"""
        assert DEFAULT_LOG_FILENAME == "app.log"
        assert DEFAULT_LOG_DIR == "logs"
        assert LOG_FILE_EXTENSION == ".log"

    def test_encoding_constant(self):
        """测试编码常量"""
        assert DEFAULT_ENCODING == "utf-8"

    def test_http_constant(self):
        """测试HTTP常量"""
        assert HTTP_OK == 200

    def test_pagination_constant(self):
        """测试分页常量"""
        assert DEFAULT_PAGE_SIZE == 100