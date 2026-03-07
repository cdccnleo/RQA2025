"""
测试用例生成相关配置

提供测试用例生成所需的各类配置对象
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig, ValidationResult, Priority, ExportFormat


@dataclass
class TestExportConfig(BaseConfig):
    """测试用例导出配置"""
    
    output_format: str
    output_dir: str
    
    # 导出选项
    include_metadata: bool = False
    include_statistics: bool = True
    include_execution_info: bool = False
    include_performance_tests: bool = True
    include_security_tests: bool = True
    
    # 自定义模板
    custom_templates: Dict[str, Any] = field(default_factory=dict)
    
    # 格式化选项
    pretty_print: bool = True
    indent: int = 2
    encoding: str = "utf-8"
    
    def __post_init__(self):
        if isinstance(self.output_format, ExportFormat):
            self.output_format = self.output_format.value
        if isinstance(self.output_format, str):
            self.output_format = self.output_format.lower()
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证导出配置"""
        if not self.output_dir:
            result.add_error("输出路径不能为空")
        
        valid_formats = [f.value for f in ExportFormat]
        if self.output_format not in valid_formats:
            result.add_error(f"不支持的输出格式: {self.output_format}. 允许值: {valid_formats}")
        
        if self.indent <= 0:
            result.add_error("缩进空格数必须大于0")

    @property
    def format(self) -> str:
        """向后兼容的格式属性"""
        return self.output_format

    @property
    def output_path(self) -> str:
        """向后兼容的输出路径属性"""
        return self.output_dir

    @output_path.setter
    def output_path(self, value: str):
        self.output_dir = value


@dataclass
class TestCaseConfig(BaseConfig):
    """测试用例配置"""
    
    case_id: str
    title: str
    description: str
    category: str  # functional, performance, security, integration
    priority: Priority = Priority.MEDIUM
    
    # 测试步骤
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, str]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    
    # 测试数据
    test_data: Dict[str, Any] = field(default_factory=dict)
    
    # 断言配置
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 其他配置
    timeout: int = 30  # 秒
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _validate_impl(self, result: ValidationResult):
        """验证测试用例配置"""
        if not self.case_id:
            result.add_error("测试用例ID不能为空")
        
        if not self.title:
            result.add_error("测试用例标题不能为空")
        
        valid_categories = ['functional', 'performance', 'security', 'integration', 'unit']
        if self.category not in valid_categories:
            result.add_error(f"测试类别必须是 {valid_categories} 之一")
        
        if self.timeout <= 0:
            result.add_error("超时时间必须大于0")
        
        if self.retry_count < 0:
            result.add_error("重试次数不能为负数")
        
        if not self.test_steps:
            result.add_warning("测试用例没有定义测试步骤")


@dataclass
class TestScenarioConfig(BaseConfig):
    """测试场景配置"""
    
    scenario_id: str
    name: str
    endpoint: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    description: str = ""
    
    # 场景变量
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # 场景中的测试用例
    test_cases: List[TestCaseConfig] = field(default_factory=list)
    
    # 场景级配置
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    setup_config: Optional[Dict[str, Any]] = None
    teardown_config: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.method = (self.method or "").upper()
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证测试场景配置"""
        if not self.scenario_id:
            result.add_error("场景ID不能为空")
        
        if not self.name:
            result.add_error("场景名称不能为空")
        
        if not self.endpoint:
            result.add_error("端点路径不能为空")
        
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if self.method.upper() not in valid_methods:
            result.add_error(f"HTTP方法必须是 {valid_methods} 之一")
        
        # 验证测试用例
        if not self.test_cases:
            result.add_warning("测试场景没有包含任何测试用例")
        
        for test_case in self.test_cases:
            case_result = test_case.validate()
            result.merge(case_result)
    
    def add_test_case(self, test_case: TestCaseConfig):
        """添加测试用例"""
        self.test_cases.append(test_case)


@dataclass
class TestSuiteConfig(BaseConfig):
    """测试套件配置"""
    
    suite_id: str
    name: str
    service_type: str = "custom"  # data_service, trading_service, feature_service, monitoring_service
    description: str = ""
    
    # 测试场景列表
    scenarios: List[TestScenarioConfig] = field(default_factory=list)
    
    # 套件级配置
    base_url: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # 执行配置
    parallel_execution: bool = False
    max_workers: int = 4
    stop_on_failure: bool = False
    
    # 报告配置
    generate_report: bool = True
    report_format: ExportFormat = ExportFormat.JSON
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证测试套件配置"""
        if not self.suite_id:
            result.add_error("测试套件ID不能为空")
        
        if not self.name:
            result.add_error("测试套件名称不能为空")
        
        valid_service_types = [
            'data_service', 'trading_service', 'feature_service', 
            'monitoring_service', 'custom'
        ]
        if self.service_type not in valid_service_types:
            result.add_error(f"服务类型必须是 {valid_service_types} 之一")
        
        if self.max_workers <= 0:
            result.add_error("最大工作线程数必须大于0")
        
        # 验证场景
        if not self.scenarios:
            result.add_warning("测试套件没有包含任何测试场景")
        
        for scenario in self.scenarios:
            scenario_result = scenario.validate()
            result.merge(scenario_result)
        
        # 如果启用并行执行，检查是否合理
        if self.parallel_execution and len(self.scenarios) < self.max_workers:
            result.add_warning(
                f"并行执行的工作线程数({self.max_workers})大于场景数({len(self.scenarios)})"
            )
    
    def add_scenario(self, scenario: TestScenarioConfig):
        """添加测试场景"""
        self.scenarios.append(scenario)
    
    def get_scenario(self, scenario_id: str) -> Optional[TestScenarioConfig]:
        """获取测试场景"""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None
    
    def get_all_test_cases(self) -> List[TestCaseConfig]:
        """获取所有测试用例"""
        all_cases = []
        for scenario in self.scenarios:
            all_cases.extend(scenario.test_cases)
        return all_cases
    
    def count_test_cases(self) -> Dict[str, int]:
        """统计测试用例数量"""
        stats = {
            'total': 0,
            'by_category': {},
            'by_priority': {}
        }
        
        for test_case in self.get_all_test_cases():
            stats['total'] += 1
            
            # 按类别统计
            category = test_case.category
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # 按优先级统计
            priority = test_case.priority
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
        
        return stats



