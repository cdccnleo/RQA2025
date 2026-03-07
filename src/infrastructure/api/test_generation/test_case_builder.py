"""
测试用例构建器基类

职责: 提供测试用例和场景的构建基础功能
"""

from typing import List, Dict, Any, Optional
from .models import TestCase, TestScenario
from .template_manager import TestTemplateManager


class TestCaseBuilder:
    """测试用例构建器基类"""
    
    def __init__(self, template_manager: Optional[TestTemplateManager] = None):
        """
        初始化构建器
        
        Args:
            template_manager: 模板管理器实例
        """
        self.template_manager = template_manager or TestTemplateManager()
        self._test_case_counter = 0
        self._scenario_counter = 0

    def _get_next_test_case_id(self) -> str:
        """获取下一个测试用例ID"""
        self._test_case_counter += 1
        return f"TC_{self._test_case_counter:03d}"

    def _get_next_scenario_id(self) -> str:
        """获取下一个场景ID"""
        self._scenario_counter += 1
        return f"SC_{self._scenario_counter:03d}"
    
    def create_test_case(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        category: str = "functional",
        preconditions: List[str] = None,
        test_steps: List[Dict[str, Any]] = None,
        expected_results: List[str] = None,
        tags: List[str] = None
    ) -> TestCase:
        """
        创建测试用例

        Args:
            title: 测试标题
            description: 测试描述
            priority: 优先级
            category: 类别
            preconditions: 前置条件
            test_steps: 测试步骤
            expected_results: 预期结果
            tags: 标签

        Returns:
            TestCase实例
        """
        return TestCase(
            id=self._get_next_test_case_id(),
            title=title,
            description=description,
            priority=priority,
            category=category,
            preconditions=preconditions or [],
            test_steps=test_steps or [],
            expected_results=expected_results or [],
            tags=tags or []
        )
    
    def create_scenario(
        self,
        name: str,
        description: str,
        endpoint: str,
        method: str,
        setup_steps: List[str] = None,
        teardown_steps: List[str] = None,
        variables: Dict[str, Any] = None,
        test_cases: List[TestCase] = None
    ) -> TestScenario:
        """
        创建测试场景

        Args:
            name: 场景名称
            description: 场景描述
            endpoint: API端点
            method: HTTP方法
            setup_steps: 准备步骤
            teardown_steps: 清理步骤
            variables: 变量字典
            test_cases: 测试用例列表

        Returns:
            TestScenario实例
        """
        return TestScenario(
            id=self._get_next_scenario_id(),
            name=name,
            description=description,
            endpoint=endpoint,
            method=method,
            setup_steps=setup_steps or [],
            teardown_steps=teardown_steps or [],
            variables=variables or {},
            test_cases=test_cases or []
        )

    def build_test_case_from_template(
        self,
        category: str,
        template_name: str,
        title: str,
        description: str = None,
        priority: str = "medium",
        category_override: str = None
    ) -> TestCase:
        """
        从模板构建测试用例

        Args:
            category: 模板类别
            template_name: 模板名称
            title: 测试用例标题
            description: 测试描述（如果为None则使用模板描述）
            priority: 优先级
            category_override: 类别覆盖

        Returns:
            TestCase实例
        """
        template = self.template_manager.get_template(category, template_name)

        if template is None:
            # 如果模板不存在，返回基本测试用例
            return TestCase(
                id=self._get_next_test_case_id(),
                title=title,
                description=description or "",
                priority=priority,
                category=category_override or category,
                preconditions=[],
                test_steps=[],
                expected_results=[],
                tags=[category, template_name]
            )

        # 使用模板信息构建测试用例
        return TestCase(
            id=self._get_next_test_case_id(),
            title=title,
            description=description or template.get("description", ""),
            priority=priority,
            category=category_override or category,
            preconditions=template.get("setup", []),
            test_steps=template.get("test_steps", []),
            expected_results=template.get("expected_results", []),
            tags=[category, template_name]
        )

    def add_test_case_to_scenario(self, scenario: TestScenario, test_case: TestCase) -> TestScenario:
        """
        将测试用例添加到场景中

        Args:
            scenario: 测试场景
            test_case: 测试用例

        Returns:
            TestScenario: 修改后的场景
        """
        scenario.test_cases.append(test_case)
        return scenario
    
    def create_authentication_tests(self, endpoint: str, method: str) -> List[TestCase]:
        """
        创建认证相关测试用例
        
        Args:
            endpoint: API端点
            method: HTTP方法
        
        Returns:
            认证测试用例列表
        """
        tests = []
        auth_templates = self.template_manager.get_category_templates("authentication")
        
        for template_name, template in auth_templates.items():
            test_case = self.create_test_case(
                title=f"{template_name}认证测试 - {endpoint}",
                description=template['description'],
                category="security",
                priority="high",
                test_steps=[
                    {"step": 1, "action": f"发送{method}请求到{endpoint}"},
                    {"step": 2, "action": f"使用{template_name}认证"},
                    {"step": 3, "action": "验证响应状态码"}
                ],
                expected_results=[
                    f"状态码应为{template['expected_status']}"
                ],
                tags=["authentication", template_name]
            )
            tests.append(test_case)
        
        return tests
    
    def create_validation_tests(self, endpoint: str, method: str) -> List[TestCase]:
        """
        创建验证相关测试用例
        
        Args:
            endpoint: API端点
            method: HTTP方法
        
        Returns:
            验证测试用例列表
        """
        tests = []
        validation_templates = self.template_manager.get_category_templates("validation")
        
        for template_name, template in validation_templates.items():
            test_case = self.create_test_case(
                title=f"{template_name}验证测试 - {endpoint}",
                description=template['description'],
                category="functional",
                priority="high",
                test_steps=[
                    {"step": 1, "action": f"准备测试数据"},
                    {"step": 2, "action": f"发送{method}请求到{endpoint}"},
                    {"step": 3, "action": "验证响应"}
                ],
                expected_results=[
                    f"状态码应为{template['expected_status']}"
                ],
                tags=["validation", template_name]
            )
            tests.append(test_case)
        
        return tests
    
    def create_security_tests(self, endpoint: str, method: str) -> List[TestCase]:
        """
        创建安全相关测试用例
        
        Args:
            endpoint: API端点
            method: HTTP方法
        
        Returns:
            安全测试用例列表
        """
        tests = []
        security_templates = self.template_manager.get_category_templates("security")
        
        for template_name, template in security_templates.items():
            test_case = self.create_test_case(
                title=f"{template_name}安全测试 - {endpoint}",
                description=template['description'],
                category="security",
                priority="critical",
                test_steps=[
                    {"step": 1, "action": "准备攻击测试数据"},
                    {"step": 2, "action": f"发送{method}请求到{endpoint}"},
                    {"step": 3, "action": "验证系统防护"}
                ],
                expected_results=[
                    "系统应正确拦截恶意请求",
                    f"返回安全的错误响应"
                ],
                tags=["security", template_name]
            )
            tests.append(test_case)
        
        return tests

