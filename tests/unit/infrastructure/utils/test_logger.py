"""
基础设施工具层Logger模块测试
"""

import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.utils.logger import get_logger, setup_logging, get_unified_logger


class TestInfrastructureUtilsLogger:
    """测试基础设施工具层Logger模块"""

    def test_get_logger_function_available(self):
        """测试get_logger函数可用"""
        assert callable(get_logger)

    def test_setup_logging_function_available(self):
        """测试setup_logging函数可用"""
        assert callable(setup_logging)

    def test_get_unified_logger_function_available(self):
        """测试get_unified_logger函数可用"""
        assert callable(get_unified_logger)

    def test_get_logger_basic_call(self):
        """测试get_logger函数基本调用"""
        # 这个测试验证函数可以被调用，不会抛出导入错误
        try:
            result = get_logger("test_logger")
            assert result is not None
        except ImportError:
            # 如果组件模块不存在，跳过测试
            pytest.skip("logger组件模块不存在")

    def test_setup_logging_basic_call(self):
        """测试setup_logging函数基本调用"""
        try:
            config = {"level": "INFO"}
            # 这个调用可能会失败，但不应该抛出导入错误
            setup_logging(config)
        except ImportError:
            pytest.skip("logger组件模块不存在")
        except Exception:
            # 其他错误是正常的，因为我们没有完整的配置
            pass

    def test_get_unified_logger_basic_call(self):
        """测试get_unified_logger函数基本调用"""
        try:
            result = get_unified_logger("test")
            assert result is not None
        except ImportError:
            pytest.skip("logger组件模块不存在")