"""
测试测试用例构建器基类

覆盖 test_case_builder.py 中的 TestCaseBuilder 类
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.test_generation.test_case_builder import TestCaseBuilder
from src.infrastructure.api.test_generation.models import TestCase, TestScenario


class TestTestCaseBuilder:
    """TestCaseBuilder 类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        builder = TestCaseBuilder()

        assert builder.template_manager is not None
        assert builder._test_case_counter == 0
        assert builder._scenario_counter == 0

    def test_initialization_with_template_manager(self):
        """测试带模板管理器初始化"""
        template_manager = Mock()
        builder = TestCaseBuilder(template_manager)

        assert builder.template_manager == template_manager
        assert builder._test_case_counter == 0
        assert builder._scenario_counter == 0

    def test_create_test_case_basic(self):
        """测试创建基本测试用例"""
        builder = TestCaseBuilder()

        test_case = builder.create_test_case(
            title="用户登录测试",
            description="测试用户登录功能"
        )

        assert isinstance(test_case, TestCase)
        assert test_case.title == "用户登录测试"
        assert test_case.description == "测试用户登录功能"
        assert test_case.priority == "medium"
        assert test_case.category == "functional"
        assert test_case.id.startswith("TC_")

    def test_create_test_case_custom(self):
        """测试创建自定义测试用例"""
        builder = TestCaseBuilder()

        preconditions = ["用户已注册"]
        test_steps = [{"action": "输入用户名", "data": "test@example.com"}]
        expected_results = ["登录成功"]
        tags = ["smoke", "authentication"]

        test_case = builder.create_test_case(
            title="用户登录测试",
            description="测试用户登录功能",
            priority="high",
            category="smoke",
            preconditions=preconditions,
            test_steps=test_steps,
            expected_results=expected_results,
            tags=tags
        )

        assert test_case.priority == "high"
        assert test_case.category == "smoke"
        assert test_case.preconditions == preconditions
        assert test_case.test_steps == test_steps
        assert test_case.expected_results == expected_results
        assert test_case.tags == tags

    def test_create_scenario_basic(self):
        """测试创建基本场景"""
        builder = TestCaseBuilder()

        scenario = builder.create_scenario(
            name="登录场景",
            description="用户登录测试场景",
            endpoint="/login",
            method="POST"
        )

        assert isinstance(scenario, TestScenario)
        assert scenario.name == "登录场景"
        assert scenario.description == "用户登录测试场景"
        assert scenario.endpoint == "/login"
        assert scenario.method == "POST"
        assert scenario.id.startswith("SC_")
        assert scenario.test_cases == []

    def test_create_scenario_with_test_cases(self):
        """测试创建包含测试用例的场景"""
        builder = TestCaseBuilder()

        test_case1 = builder.create_test_case("测试1", "描述1")
        test_case2 = builder.create_test_case("测试2", "描述2")

        scenario = builder.create_scenario(
            name="完整场景",
            description="包含多个测试用例的场景",
            endpoint="/api/test",
            method="GET",
            test_cases=[test_case1, test_case2],
            setup_steps=["初始化数据"],
            teardown_steps=["清理数据"],
            variables={"timeout": 30}
        )

        assert scenario.endpoint == "/api/test"
        assert scenario.method == "GET"
        assert len(scenario.test_cases) == 2
        assert scenario.setup_steps == ["初始化数据"]
        assert scenario.teardown_steps == ["清理数据"]
        assert scenario.variables == {"timeout": 30}

    def test_get_next_test_case_id(self):
        """测试获取下一个测试用例ID"""
        builder = TestCaseBuilder()

        # 初始计数器为0
        id1 = builder._get_next_test_case_id()
        assert id1 == "TC_001"

        # 计数器递增
        id2 = builder._get_next_test_case_id()
        assert id2 == "TC_002"

        # 验证计数器更新
        assert builder._test_case_counter == 2

    def test_get_next_scenario_id(self):
        """测试获取下一个场景ID"""
        builder = TestCaseBuilder()

        id1 = builder._get_next_scenario_id()
        assert id1 == "SC_001"

        id2 = builder._get_next_scenario_id()
        assert id2 == "SC_002"

        assert builder._scenario_counter == 2

    def test_build_test_case_from_template(self):
        """测试从模板构建测试用例"""
        template_manager = Mock()
        template_manager.get_template.return_value = {
            "description": "模板描述",
            "priority": "high",
            "category": "security",
            "test_steps": [{"action": "安全检查"}],
            "expected_results": ["安全验证通过"]
        }

        builder = TestCaseBuilder(template_manager)

        test_case = builder.build_test_case_from_template(
            category="authentication",
            template_name="login",
            title="登录安全测试",
            priority="high",
            category_override="security"
        )

        assert test_case.title == "登录安全测试"
        assert test_case.description == "模板描述"
        assert test_case.priority == "high"
        assert test_case.category == "security"
        assert test_case.test_steps == [{"action": "安全检查"}]
        assert test_case.expected_results == ["安全验证通过"]

        # 验证模板管理器被调用
        template_manager.get_template.assert_called_once_with("authentication", "login")

    def test_build_test_case_from_template_not_found(self):
        """测试从不存在的模板构建测试用例"""
        template_manager = Mock()
        template_manager.get_template.return_value = None

        builder = TestCaseBuilder(template_manager)

        test_case = builder.build_test_case_from_template(
            category="unknown",
            template_name="missing",
            title="测试标题"
        )

        # 应该返回基本测试用例
        assert test_case.title == "测试标题"
        assert test_case.description == ""
        assert test_case.priority == "medium"

    def test_add_test_case_to_scenario(self):
        """测试向场景添加测试用例"""
        builder = TestCaseBuilder()

        scenario = builder.create_scenario("测试场景", "描述", "/test", "GET")
        test_case = builder.create_test_case("测试用例", "描述")

        result = builder.add_test_case_to_scenario(scenario, test_case)

        assert result == scenario  # 返回场景本身
        assert len(scenario.test_cases) == 1
        assert scenario.test_cases[0] == test_case

    def test_add_multiple_test_cases_to_scenario(self):
        """测试向场景添加多个测试用例"""
        builder = TestCaseBuilder()

        scenario = builder.create_scenario("测试场景", "描述", "/test", "GET")
        test_cases = [
            builder.create_test_case("测试1", "描述1"),
            builder.create_test_case("测试2", "描述2"),
            builder.create_test_case("测试3", "描述3")
        ]

        for tc in test_cases:
            builder.add_test_case_to_scenario(scenario, tc)

        assert len(scenario.test_cases) == 3
        assert scenario.test_cases == test_cases

    def test_create_scenario_with_setup_teardown(self):
        """测试创建包含设置和清理的场景"""
        builder = TestCaseBuilder()

        setup_steps = ["连接数据库", "初始化测试数据"]
        teardown_steps = ["清理测试数据", "断开连接"]

        scenario = builder.create_scenario(
            name="数据库测试场景",
            description="测试数据库操作",
            endpoint="/api/db",
            method="POST",
            setup_steps=setup_steps,
            teardown_steps=teardown_steps
        )

        assert scenario.setup_steps == setup_steps
        assert scenario.teardown_steps == teardown_steps

    def test_template_manager_property(self):
        """测试模板管理器属性"""
        template_manager = Mock()
        builder = TestCaseBuilder(template_manager)

        assert builder.template_manager == template_manager

        # 测试默认模板管理器
        builder2 = TestCaseBuilder()
        assert builder2.template_manager is not None

    def test_counter_independence(self):
        """测试计数器独立性"""
        builder1 = TestCaseBuilder()
        builder2 = TestCaseBuilder()

        # builder1 生成一些ID
        builder1._get_next_test_case_id()
        builder1._get_next_test_case_id()
        builder1._get_next_scenario_id()

        # builder2 应该从头开始
        assert builder2._test_case_counter == 0
        assert builder2._scenario_counter == 0

        tc_id = builder2._get_next_test_case_id()
        sc_id = builder2._get_next_scenario_id()

        assert tc_id == "TC_001"
        assert sc_id == "SC_001"
