"""
测试构建器基类

提供测试套件构建的通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class TestCase:
    """测试用例"""
    id: str
    title: str
    description: str
    priority: str = "medium"
    category: str = "functional"
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    actual_results: Optional[List[str]] = None
    status: str = "pending"
    execution_time: Optional[float] = None
    environment: str = "test"
    tags: List[str] = field(default_factory=list)


@dataclass
class TestScenario:
    """测试场景"""
    id: str
    name: str
    description: str
    endpoint: str
    method: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """测试套件"""
    id: str
    name: str
    description: str
    scenarios: List[TestScenario] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseTestBuilder(ABC):
    """
    测试构建器基类
    
    定义测试套件构建的标准流程和通用方法
    """
    
    def __init__(self, template_manager=None):
        """
        初始化测试构建器
        
        Args:
            template_manager: 模板管理器实例
        """
        self.template_manager = template_manager
    
    @abstractmethod
    def build_test_suite(self) -> TestSuite:
        """
        构建测试套件（子类必须实现）
        
        Returns:
            TestSuite: 构建的测试套件
        """
        pass
    
    def _create_test_case(
        self,
        case_id: str,
        title: str,
        description: str,
        priority: str = "medium",
        category: str = "functional",
        preconditions: Optional[List[str]] = None,
        test_steps: Optional[List[Dict[str, Any]]] = None,
        expected_results: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> TestCase:
        """
        创建测试用例的辅助方法
        
        Args:
            case_id: 测试用例ID
            title: 测试用例标题
            description: 测试用例描述
            priority: 优先级
            category: 类别
            preconditions: 前置条件
            test_steps: 测试步骤
            expected_results: 预期结果
            tags: 标签
        
        Returns:
            TestCase: 创建的测试用例
        """
        return TestCase(
            id=case_id,
            title=title,
            description=description,
            priority=priority,
            category=category,
            preconditions=preconditions or [],
            test_steps=test_steps or [],
            expected_results=expected_results or [],
            tags=tags or []
        )
    
    def _create_test_scenario(
        self,
        scenario_id: str,
        name: str,
        description: str,
        endpoint: str,
        method: str,
        variables: Optional[Dict[str, Any]] = None,
        setup_steps: Optional[List[str]] = None,
        teardown_steps: Optional[List[str]] = None
    ) -> TestScenario:
        """
        创建测试场景的辅助方法
        
        Args:
            scenario_id: 场景ID
            name: 场景名称
            description: 场景描述
            endpoint: API端点
            method: HTTP方法
            variables: 场景变量
            setup_steps: 设置步骤
            teardown_steps: 清理步骤
        
        Returns:
            TestScenario: 创建的测试场景
        """
        return TestScenario(
            id=scenario_id,
            name=name,
            description=description,
            endpoint=endpoint,
            method=method,
            variables=variables or {},
            setup_steps=setup_steps or [],
            teardown_steps=teardown_steps or []
        )
    
    def _get_template(self, category: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        获取模板的辅助方法
        
        Args:
            category: 模板类别
            template_name: 模板名称
        
        Returns:
            模板字典或None
        """
        if self.template_manager:
            return self.template_manager.get_template(category, template_name)
        return None
    
    def _create_auth_test_cases(
        self,
        scenario: TestScenario,
        auth_types: Optional[List[str]] = None
    ) -> List[TestCase]:
        """
        创建认证测试用例
        
        Args:
            scenario: 测试场景
            auth_types: 认证类型列表
        
        Returns:
            认证测试用例列表
        """
        test_cases = []
        auth_types = auth_types or ['bearer_token']
        
        for auth_type in auth_types:
            template = self._get_template('authentication', auth_type)
            if not template:
                continue
            
            # 有效认证测试
            test_cases.append(self._create_test_case(
                case_id=f"{scenario.id}_auth_{auth_type}_valid",
                title=f"有效{auth_type}认证",
                description=f"使用有效的{auth_type}进行认证",
                priority="high",
                category="security",
                expected_results=[f"返回状态码200", "成功通过认证"],
                tags=[auth_type, "authentication", "valid"]
            ))
            
            # 无效认证测试
            test_cases.append(self._create_test_case(
                case_id=f"{scenario.id}_auth_{auth_type}_invalid",
                title=f"无效{auth_type}认证",
                description=f"使用无效的{auth_type}进行认证",
                priority="high",
                category="security",
                expected_results=["返回状态码401", "认证失败"],
                tags=[auth_type, "authentication", "invalid"]
            ))
        
        return test_cases
    
    def _create_validation_test_cases(
        self,
        scenario: TestScenario,
        validation_types: Optional[List[str]] = None
    ) -> List[TestCase]:
        """
        创建验证测试用例
        
        Args:
            scenario: 测试场景
            validation_types: 验证类型列表
        
        Returns:
            验证测试用例列表
        """
        test_cases = []
        validation_types = validation_types or ['required_fields', 'data_types', 'constraints']
        
        for val_type in validation_types:
            template = self._get_template('validation', val_type)
            if not template:
                continue
            
            test_cases.append(self._create_test_case(
                case_id=f"{scenario.id}_validation_{val_type}",
                title=f"{template.get('description', val_type)}",
                description=f"验证{val_type}的正确性",
                priority="high",
                category="functional",
                expected_results=["返回状态码400", "包含错误信息"],
                tags=[val_type, "validation"]
            ))
        
        return test_cases
    
    def _create_error_handling_test_cases(
        self,
        scenario: TestScenario,
        error_types: Optional[List[str]] = None
    ) -> List[TestCase]:
        """
        创建错误处理测试用例
        
        Args:
            scenario: 测试场景
            error_types: 错误类型列表
        
        Returns:
            错误处理测试用例列表
        """
        test_cases = []
        error_types = error_types or ['not_found', 'server_error', 'rate_limit']
        
        for error_type in error_types:
            template = self._get_template('error_handling', error_type)
            if not template:
                continue
            
            test_cases.append(self._create_test_case(
                case_id=f"{scenario.id}_error_{error_type}",
                title=f"{template.get('description', error_type)}",
                description=f"测试{error_type}错误处理",
                priority="medium",
                category="functional",
                expected_results=[
                    f"返回状态码{template.get('expected_status')}",
                    "包含正确的错误码和错误信息"
                ],
                tags=[error_type, "error_handling"]
            ))
        
        return test_cases

