"""
测试测试构建器基类

覆盖 builders/base_builder.py 中的 TestCase、TestScenario、TestSuite 和 BaseTestBuilder 类
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.test_generation.builders.base_builder import (
    TestCase,
    TestScenario,
    TestSuite,
    BaseTestBuilder
)

class TestTestSuite:
    """TestSuite 数据类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        suite = TestSuite(
            id="TS001",
            name="测试套件",
            description="测试套件描述"
        )

        assert suite.id == "TS001"
        assert suite.name == "测试套件"
        assert suite.description == "测试套件描述"
        assert suite.scenarios == []
        assert isinstance(suite.created_at, str)
        assert isinstance(suite.updated_at, str)

    def test_initialization_with_scenarios(self):
        """测试带场景初始化"""
        scenario = TestScenario(
            id="SC001",
            name="登录场景",
            description="用户登录测试场景",
            endpoint="/login",
            method="POST",
            test_cases=[]
        )

        suite = TestSuite(
            id="TS002",
            name="用户API测试",
            description="用户API完整测试",
            scenarios=[scenario]
        )

        assert suite.id == "TS002"
        assert len(suite.scenarios) == 1
        assert suite.scenarios[0] == scenario

class TestTestScenario:
    """TestScenario 数据类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        scenario = TestScenario(
            id="SC001",
            name="登录场景",
            description="用户登录测试场景",
            endpoint="/login",
            method="POST"
        )

        assert scenario.id == "SC001"
        assert scenario.name == "登录场景"
        assert scenario.description == "用户登录测试场景"
        assert scenario.endpoint == "/login"
        assert scenario.method == "POST"
        assert scenario.test_cases == []
        assert scenario.setup_steps == []
        assert scenario.teardown_steps == []
        assert scenario.variables == {}

    def test_initialization_complete(self):
        """测试完整初始化"""
        test_case = TestCase("TC001", "登录测试", "测试登录功能")
        setup_steps = ["初始化数据库", "启动服务"]
        teardown_steps = ["清理数据", "停止服务"]
        variables = {"base_url": "http://localhost:8000", "timeout": 30}

        scenario = TestScenario(
            id="SC002",
            name="完整场景",
            description="包含所有元素的场景",
            endpoint="/api/test",
            method="GET",
            test_cases=[test_case],
            setup_steps=setup_steps,
            teardown_steps=teardown_steps,
            variables=variables
        )

        assert scenario.id == "SC002"
        assert scenario.endpoint == "/api/test"
        assert scenario.method == "GET"
        assert len(scenario.test_cases) == 1
        assert scenario.test_cases[0] == test_case
        assert scenario.setup_steps == setup_steps
        assert scenario.teardown_steps == teardown_steps
        assert scenario.variables == variables

class TestTestCase:
    """TestCase 数据类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        test_case = TestCase(
            id="TC001",
            title="用户登录测试",
            description="测试用户登录功能"
        )

        assert test_case.id == "TC001"
        assert test_case.title == "用户登录测试"
        assert test_case.description == "测试用户登录功能"
        assert test_case.priority == "medium"
        assert test_case.category == "functional"
        assert test_case.preconditions == []
        assert test_case.test_steps == []
        assert test_case.expected_results == []
        assert test_case.actual_results is None
        assert test_case.status == "pending"
        assert test_case.execution_time is None
        assert test_case.environment == "test"
        assert test_case.tags == []

    def test_initialization_custom(self):
        """测试自定义初始化"""
        preconditions = ["用户已注册", "邮箱已验证"]
        test_steps = [
            {"action": "输入用户名", "data": "test@example.com"},
            {"action": "输入密码", "data": "password123"}
        ]
        expected_results = ["登录成功", "跳转到首页"]
        tags = ["smoke", "authentication"]

        test_case = TestCase(
            id="TC002",
            title="用户登录测试",
            description="测试用户登录功能",
            priority="high",
            category="smoke",
            preconditions=preconditions,
            test_steps=test_steps,
            expected_results=expected_results,
            actual_results=["登录成功", "跳转到首页"],
            status="passed",
            execution_time=2.5,
            environment="staging",
            tags=tags
        )

        assert test_case.id == "TC002"
        assert test_case.title == "用户登录测试"
        assert test_case.description == "测试用户登录功能"
        assert test_case.priority == "high"
        assert test_case.category == "smoke"
        assert test_case.preconditions == preconditions
        assert test_case.test_steps == test_steps
        assert test_case.expected_results == expected_results
        assert test_case.actual_results == ["登录成功", "跳转到首页"]
        assert test_case.status == "passed"
        assert test_case.execution_time == 2.5
        assert test_case.environment == "staging"
        assert test_case.tags == tags

    def test_equality(self):
        """测试相等性"""
        test_case1 = TestCase(
            id="TC001",
            title="测试用例1",
            description="描述1"
        )
        test_case2 = TestCase(
            id="TC001",
            title="测试用例1",
            description="描述1"
        )
        test_case3 = TestCase(
            id="TC002",
            title="测试用例2",
            description="描述2"
        )

        assert test_case1 == test_case2
        assert test_case1 != test_case3

    def test_list_modification(self):
        """测试列表字段的可修改性"""
        test_case = TestCase(
            id="TC001",
            title="测试用例",
            description="描述",
            preconditions=["条件1"],
            test_steps=[{"action": "步骤1"}],
            expected_results=["结果1"],
            tags=["标签1"]
        )

        # 验证列表字段的修改
        original_preconditions = test_case.preconditions.copy()
        test_case.preconditions.append("条件2")
        assert test_case.preconditions == original_preconditions + ["条件2"]


