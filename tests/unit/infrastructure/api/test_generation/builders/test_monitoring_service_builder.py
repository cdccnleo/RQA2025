"""
测试监控服务测试构建器

覆盖 builders/monitoring_service_builder.py 中的 MonitoringServiceTestBuilder 类
"""

import pytest
from src.infrastructure.api.test_generation.builders.monitoring_service_builder import MonitoringServiceTestBuilder
from src.infrastructure.api.test_generation.builders.base_builder import TestSuite


class TestMonitoringServiceTestBuilder:
    """MonitoringServiceTestBuilder 类测试"""

    def test_initialization(self):
        """测试初始化"""
        builder = MonitoringServiceTestBuilder()

        assert isinstance(builder, MonitoringServiceTestBuilder)
        assert hasattr(builder, 'build_test_suite')
        assert hasattr(builder, 'get_supported_operations')

    def test_build_test_suite(self):
        """测试构建测试套件"""
        builder = MonitoringServiceTestBuilder()
        suite = builder.build_test_suite()

        assert suite is not None
        assert isinstance(suite, TestSuite)
        assert suite.id == "monitoring_service_tests"
        assert suite.name == "监控服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_supported_operations(self):
        """测试获取支持的操作"""
        builder = MonitoringServiceTestBuilder()
        operations = builder.get_supported_operations()

        assert isinstance(operations, list)
        assert len(operations) > 0
        assert "health_check" in operations
        assert "metrics_query" in operations
