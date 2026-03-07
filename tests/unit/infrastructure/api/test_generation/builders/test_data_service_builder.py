"""
测试数据服务测试构建器

覆盖 builders/data_service_builder.py 中的 DataServiceTestBuilder 类
"""

import pytest
from src.infrastructure.api.test_generation.builders.data_service_builder import DataServiceTestBuilder
from src.infrastructure.api.test_generation.builders.base_builder import TestSuite


class TestDataServiceTestBuilder:
    """DataServiceTestBuilder 类测试"""

    def test_build_test_suite(self):
        """测试构建测试套件"""
        builder = DataServiceTestBuilder()
        suite = builder.build_test_suite()

        assert suite is not None
        assert isinstance(suite, TestSuite)
        assert suite.id == "data_service_tests"
        assert suite.name == "数据服务API测试"
        assert len(suite.scenarios) > 0