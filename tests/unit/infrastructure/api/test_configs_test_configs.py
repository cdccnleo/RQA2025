"""
测试配置测试
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.api.configs.test_configs import (
    TestCaseConfig,
    TestScenarioConfig,
    TestSuiteConfig,
    TestExportConfig
)
from src.infrastructure.api.configs.base_config import Priority


class TestTestCaseConfig:
    """测试测试用例配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = TestCaseConfig(
            case_id="TC001",
            title="Login Test",
            description="Test user login",
            category="functional"
        )

        assert config.case_id == "TC001"
        assert config.title == "Login Test"
        assert config.description == "Test user login"
        assert config.category == "functional"
        assert config.priority == Priority.MEDIUM
        assert config.timeout == 30
        assert config.retry_count == 0

    def test_init_complete(self):
        """测试完整初始化"""
        preconditions = ["User is registered", "System is running"]
        test_steps = [
            {"action": "Navigate to login page", "expected": "Login form displayed"},
            {"action": "Enter credentials", "expected": "Credentials accepted"}
        ]
        expected_results = ["User logged in successfully"]

        config = TestCaseConfig(
            case_id="TC002",
            title="Complete Login Test",
            description="Complete user login test scenario",
            category="integration",
            priority=Priority.HIGH,
            preconditions=preconditions,
            test_steps=test_steps,
            expected_results=expected_results,
            timeout=60,
            retry_count=2,
            tags=["login", "smoke"]
        )

        assert config.case_id == "TC002"
        assert config.title == "Complete Login Test"
        assert config.category == "integration"
        assert config.priority == Priority.HIGH
        assert config.preconditions == preconditions
        assert config.test_steps == test_steps
        assert config.expected_results == expected_results
        assert config.timeout == 60
        assert config.retry_count == 2
        assert config.tags == ["login", "smoke"]

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = TestCaseConfig(
            case_id="TC001",
            title="Test",
            description="Description",
            category="functional"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_id(self):
        """测试验证缺失ID"""
        config = TestCaseConfig(
            case_id="",
            title="Test",
            description="Description",
            category="functional"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "测试用例ID不能为空" in result.errors[0]


class TestTestScenarioConfig:
    """测试测试场景配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = TestScenarioConfig(
            scenario_id="SC001",
            name="User Registration",
            endpoint="/api/users",
            method="POST"
        )

        assert config.scenario_id == "SC001"
        assert config.name == "User Registration"
        assert config.endpoint == "/api/users"
        assert config.method == "POST"
        assert config.test_cases == []

    def test_init_complete(self):
        """测试完整初始化"""
        test_cases = [
            TestCaseConfig(
                case_id="TC001",
                title="Valid Registration",
                description="Test valid user registration",
                category="functional"
            )
        ]

        config = TestScenarioConfig(
            scenario_id="SC002",
            name="Complete Registration Flow",
            endpoint="/api/users",
            method="POST",
            test_cases=test_cases,
            setup_steps=["Initialize database", "Start services"],
            teardown_steps=["Cleanup data", "Stop services"]
        )

        assert config.scenario_id == "SC002"
        assert config.name == "Complete Registration Flow"
        assert len(config.test_cases) == 1
        assert config.setup_steps == ["Initialize database", "Start services"]
        assert config.teardown_steps == ["Cleanup data", "Stop services"]

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = TestScenarioConfig(
            scenario_id="SC001",
            name="Test Scenario",
            endpoint="/api/test",
            method="GET"
        )

        result = config.validate()
        assert result.is_valid is True


class TestTestSuiteConfig:
    """测试测试套件配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = TestSuiteConfig(
            suite_id="SUITE001",
            name="API Tests",
            description="Complete API test suite"
        )

        assert config.suite_id == "SUITE001"
        assert config.name == "API Tests"
        assert config.description == "Complete API test suite"
        assert config.scenarios == []

    def test_init_complete(self):
        """测试完整初始化"""
        scenarios = [
            TestScenarioConfig(
                scenario_id="SC001",
                name="User API",
                endpoint="/api/users",
                method="GET"
            )
        ]

        config = TestSuiteConfig(
            suite_id="SUITE002",
            name="Complete API Suite",
            description="Full API test coverage",
            scenarios=scenarios
        )

        assert config.suite_id == "SUITE002"
        assert config.name == "Complete API Suite"
        assert len(config.scenarios) == 1

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = TestSuiteConfig(
            suite_id="SUITE001",
            name="Test Suite",
            description="Test suite description"
        )

        result = config.validate()
        assert result.is_valid is True


class TestTestExportConfig:
    """测试测试导出配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = TestExportConfig(
            output_format="json",
            output_dir="./tests"
        )

        assert config.output_format == "json"
        assert config.output_dir == "./tests"
        assert config.include_performance_tests is True
        assert config.include_security_tests is True

    def test_init_complete(self):
        """测试完整初始化"""
        config = TestExportConfig(
            output_format="yaml",
            output_dir="/tmp/tests",
            include_performance_tests=False,
            include_security_tests=False,
            custom_templates={}
        )

        assert config.output_format == "yaml"
        assert config.output_dir == "/tmp/tests"
        assert config.include_performance_tests is False
        assert config.include_security_tests is False
        assert config.custom_templates == {}

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = TestExportConfig(
            output_format="json",
            output_dir="./tests"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_invalid_format(self):
        """测试验证无效格式"""
        config = TestExportConfig(
            output_format="invalid",
            output_dir="./tests"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "不支持的输出格式" in result.errors[0]
