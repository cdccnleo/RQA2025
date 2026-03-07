"""
测试交易服务测试构建器

覆盖 builders/trading_service_builder.py 中的 TradingServiceTestBuilder 类
"""

import pytest
from src.infrastructure.api.test_generation.builders.trading_service_builder import TradingServiceTestBuilder
from src.infrastructure.api.test_generation.builders.base_builder import TestSuite


class TestTradingServiceTestBuilder:
    """TradingServiceTestBuilder 类测试"""

    def test_initialization(self):
        """测试初始化"""
        builder = TradingServiceTestBuilder()

        assert isinstance(builder, TradingServiceTestBuilder)
        assert hasattr(builder, 'build_test_suite')
        assert hasattr(builder, 'get_supported_operations')

    def test_build_test_suite(self):
        """测试构建测试套件"""
        builder = TradingServiceTestBuilder()
        suite = builder.build_test_suite()

        assert suite is not None
        assert isinstance(suite, TestSuite)
        assert suite.id == "trading_service_tests"
        assert suite.name == "交易服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_supported_operations(self):
        """测试获取支持的操作"""
        builder = TradingServiceTestBuilder()
        operations = builder.get_supported_operations()

        assert isinstance(operations, list)
        assert len(operations) > 0
        assert "order_placement" in operations
        assert "order_cancellation" in operations
