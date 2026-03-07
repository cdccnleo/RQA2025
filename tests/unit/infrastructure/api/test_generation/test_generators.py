"""
测试服务测试生成器

覆盖 generators.py 中的各种测试生成器类
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.test_generation.generators import (
    DataServiceTestGenerator,
    FeatureServiceTestGenerator,
    TradingServiceTestGenerator,
    MonitoringServiceTestGenerator
)
from src.infrastructure.api.test_generation.models import TestSuite


class TestDataServiceTestGenerator:
    """DataServiceTestGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        template_manager = Mock()
        generator = DataServiceTestGenerator(template_manager)

        assert isinstance(generator, DataServiceTestGenerator)
        assert generator.template_manager == template_manager

    def test_create_test_suite(self):
        """测试创建测试套件"""
        template_manager = Mock()
        # 设置Mock返回值
        template_manager.get_category_templates.return_value = {
            "basic_auth": {
                "description": "基本认证测试",
                "expected_status": 200,
                "template": "auth_template"
            }
        }
        template_manager.get_template.return_value = {"description": "模板描述"}
        generator = DataServiceTestGenerator(template_manager)

        suite = generator.create_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id == "data_service_tests"
        assert suite.name == "数据服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_service_type(self):
        """测试获取服务类型"""
        template_manager = Mock()
        generator = DataServiceTestGenerator(template_manager)

        service_type = generator.get_service_type()

        assert service_type == "data"


class TestFeatureServiceTestGenerator:
    """FeatureServiceTestGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        template_manager = Mock()
        generator = FeatureServiceTestGenerator(template_manager)

        assert isinstance(generator, FeatureServiceTestGenerator)
        assert generator.template_manager == template_manager

    def test_create_test_suite(self):
        """测试创建测试套件"""
        template_manager = Mock()
        # 设置Mock返回值
        template_manager.get_category_templates.return_value = {
            "basic_auth": {
                "description": "基本认证测试",
                "expected_status": 200,
                "template": "auth_template"
            }
        }
        template_manager.get_template.return_value = {"description": "模板描述"}
        generator = FeatureServiceTestGenerator(template_manager)

        suite = generator.create_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id == "feature_service_tests"
        assert suite.name == "特征工程服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_service_type(self):
        """测试获取服务类型"""
        template_manager = Mock()
        generator = FeatureServiceTestGenerator(template_manager)

        service_type = generator.get_service_type()

        assert service_type == "feature"


class TestTradingServiceTestGenerator:
    """TradingServiceTestGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        template_manager = Mock()
        generator = TradingServiceTestGenerator(template_manager)

        assert isinstance(generator, TradingServiceTestGenerator)
        assert generator.template_manager == template_manager

    def test_create_test_suite(self):
        """测试创建测试套件"""
        template_manager = Mock()
        generator = TradingServiceTestGenerator(template_manager)

        suite = generator.create_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id == "trading_service_tests"
        assert suite.name == "交易服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_service_type(self):
        """测试获取服务类型"""
        template_manager = Mock()
        generator = TradingServiceTestGenerator(template_manager)

        service_type = generator.get_service_type()

        assert service_type == "trading"


class TestMonitoringServiceTestGenerator:
    """MonitoringServiceTestGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        template_manager = Mock()
        generator = MonitoringServiceTestGenerator(template_manager)

        assert isinstance(generator, MonitoringServiceTestGenerator)
        assert generator.template_manager == template_manager

    def test_create_test_suite(self):
        """测试创建测试套件"""
        template_manager = Mock()
        generator = MonitoringServiceTestGenerator(template_manager)

        suite = generator.create_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id == "monitoring_service_tests"
        assert suite.name == "监控服务API测试"
        assert len(suite.scenarios) > 0

    def test_get_service_type(self):
        """测试获取服务类型"""
        template_manager = Mock()
        generator = MonitoringServiceTestGenerator(template_manager)

        service_type = generator.get_service_type()

        assert service_type == "monitoring"


class TestGeneratorsIntegration:
    """生成器集成测试"""

    def test_all_generators_create_valid_suites(self):
        """测试所有生成器都能创建有效的测试套件"""
        template_manager = Mock()
        # 设置Mock返回值
        template_manager.get_category_templates.return_value = {
            "basic_auth": {
                "description": "基本认证测试",
                "expected_status": 200,
                "template": "auth_template"
            }
        }
        template_manager.get_template.return_value = {"description": "模板描述"}

        generators = [
            DataServiceTestGenerator(template_manager),
            FeatureServiceTestGenerator(template_manager),
            TradingServiceTestGenerator(template_manager),
            MonitoringServiceTestGenerator(template_manager)
        ]

        for generator in generators:
            suite = generator.create_test_suite()

            # 验证套件基本属性
            assert isinstance(suite, TestSuite)
            assert suite.id.endswith("_tests")
            assert "API测试" in suite.name
            assert len(suite.scenarios) > 0

            # 验证每个场景都有测试用例
            for scenario in suite.scenarios:
                assert len(scenario.test_cases) > 0
                assert all(tc.id for tc in scenario.test_cases)

    def test_service_types_are_unique(self):
        """测试服务类型都是唯一的"""
        template_manager = Mock()

        generators = [
            DataServiceTestGenerator(template_manager),
            FeatureServiceTestGenerator(template_manager),
            TradingServiceTestGenerator(template_manager),
            MonitoringServiceTestGenerator(template_manager)
        ]

        service_types = [gen.get_service_type() for gen in generators]
        assert len(service_types) == len(set(service_types))  # 所有类型都唯一

    def test_template_manager_integration(self):
        """测试模板管理器集成"""
        template_manager = Mock()
        # 设置Mock返回值
        template_manager.get_category_templates.return_value = {
            "basic_auth": {
                "description": "基本认证测试",
                "expected_status": 200,
                "template": "auth_template"
            }
        }
        template_manager.get_template.return_value = {"description": "模板描述"}

        generator = DataServiceTestGenerator(template_manager)
        suite = generator.create_test_suite()

        # 验证生成了套件（即使模板返回mock数据）
        assert isinstance(suite, TestSuite)
        assert len(suite.scenarios) > 0

    def test_generator_inheritance(self):
        """测试生成器继承关系"""
        from src.infrastructure.api.test_generation.test_case_builder import TestCaseBuilder

        template_manager = Mock()

        generators = [
            DataServiceTestGenerator(template_manager),
            FeatureServiceTestGenerator(template_manager),
            TradingServiceTestGenerator(template_manager),
            MonitoringServiceTestGenerator(template_manager)
        ]

        for generator in generators:
            assert isinstance(generator, TestCaseBuilder)
            assert hasattr(generator, 'template_manager')
            assert hasattr(generator, 'create_test_suite')
            assert hasattr(generator, 'get_service_type')
