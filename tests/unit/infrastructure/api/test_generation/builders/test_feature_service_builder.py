"""
测试特征服务测试构建器

覆盖 builders/feature_service_builder.py 中的 FeatureServiceTestBuilder 类
"""

import pytest
from src.infrastructure.api.test_generation.builders.feature_service_builder import FeatureServiceTestBuilder
from src.infrastructure.api.test_generation.builders.base_builder import TestSuite


class TestFeatureServiceTestBuilder:
    """FeatureServiceTestBuilder 类测试"""

    def test_build_test_suite(self):
        """测试构建测试套件"""
        builder = FeatureServiceTestBuilder()
        suite = builder.build_test_suite()

        assert suite is not None
        assert isinstance(suite, TestSuite)
        assert suite.id == "feature_service_tests"
        assert suite.name == "特征服务API测试"
        assert len(suite.scenarios) > 0
