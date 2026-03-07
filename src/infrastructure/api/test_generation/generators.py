"""
服务测试生成器

包含各个服务的专门测试生成器：
- DataServiceTestGenerator: 数据服务
- FeatureServiceTestGenerator: 特征服务
- TradingServiceTestGenerator: 交易服务
- MonitoringServiceTestGenerator: 监控服务
"""

from typing import List
from .models import TestSuite, TestScenario, TestCase
from .test_case_builder import TestCaseBuilder
from .template_manager import TestTemplateManager


class DataServiceTestGenerator(TestCaseBuilder):
    """数据服务测试生成器"""
    
    def __init__(self, template_manager: TestTemplateManager):
        """初始化数据服务测试生成器"""
        super().__init__(template_manager)
    
    def create_test_suite(self) -> TestSuite:
        """创建数据服务测试套件"""
        suite = TestSuite(
            id="data_service_tests",
            name="数据服务API测试",
            description="RQA2025数据服务的完整API测试套件"
        )
        
        # 添加各种测试场景
        suite.scenarios.append(self._create_market_data_scenario())
        suite.scenarios.append(self._create_historical_data_scenario())
        suite.scenarios.append(self._create_realtime_data_scenario())
        suite.scenarios.append(self._create_data_validation_scenario())
        
        return suite
    
    def _create_market_data_scenario(self) -> TestScenario:
        """创建市场数据获取场景"""
        scenario = self.create_scenario(
            name="市场数据获取",
            description="测试市场数据API的获取功能",
            endpoint="/api/v1/data/market",
            method="GET"
        )
        
        # 添加测试用例
        scenario.test_cases.extend(self.create_authentication_tests(scenario.endpoint, scenario.method))
        scenario.test_cases.extend(self.create_validation_tests(scenario.endpoint, scenario.method))
        
        # 添加特定测试用例
        scenario.test_cases.append(self.create_test_case(
            title="获取市场数据 - 正常流程",
            description="测试正常获取市场数据",
            priority="high",
            test_steps=[
                {"step": 1, "action": "准备有效的股票代码"},
                {"step": 2, "action": "发送GET请求"},
                {"step": 3, "action": "验证返回数据格式"}
            ],
            expected_results=[
                "状态码200",
                "返回包含价格、成交量等字段",
                "数据时间戳正确"
            ],
            tags=["market_data", "normal_flow"]
        ))
        
        return scenario
    
    def _create_historical_data_scenario(self) -> TestScenario:
        """创建历史数据查询场景"""
        scenario = self.create_scenario(
            name="历史数据查询",
            description="测试历史数据查询API",
            endpoint="/api/v1/data/historical",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="查询历史数据 - 时间范围查询",
            description="测试指定时间范围的历史数据查询",
            priority="high",
            test_steps=[
                {"step": 1, "action": "准备查询参数(股票代码、开始时间、结束时间)"},
                {"step": 2, "action": "发送POST请求"},
                {"step": 3, "action": "验证返回数据"}
            ],
            expected_results=[
                "返回指定时间范围内的数据",
                "数据按时间排序",
                "包含OHLCV字段"
            ],
            tags=["historical_data", "query"]
        ))
        
        return scenario
    
    def _create_realtime_data_scenario(self) -> TestScenario:
        """创建实时数据订阅场景"""
        scenario = self.create_scenario(
            name="实时数据订阅",
            description="测试实时数据订阅功能",
            endpoint="/api/v1/data/realtime",
            method="WS"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="订阅实时行情",
            description="测试WebSocket实时行情订阅",
            priority="high",
            category="integration",
            test_steps=[
                {"step": 1, "action": "建立WebSocket连接"},
                {"step": 2, "action": "发送订阅请求"},
                {"step": 3, "action": "接收实时数据"},
                {"step": 4, "action": "验证数据更新"}
            ],
            expected_results=[
                "连接成功建立",
                "接收到实时行情推送",
                "数据更新及时(延迟<100ms)"
            ],
            tags=["realtime", "websocket"]
        ))
        
        return scenario
    
    def _create_data_validation_scenario(self) -> TestScenario:
        """创建数据验证场景"""
        scenario = self.create_scenario(
            name="数据验证",
            description="测试数据验证和清洗功能",
            endpoint="/api/v1/data/validate",
            method="POST"
        )
        
        scenario.test_cases.extend(self.create_validation_tests(scenario.endpoint, scenario.method))
        
        return scenario

    def get_service_type(self) -> str:
        """获取服务类型"""
        return "data"


class FeatureServiceTestGenerator(TestCaseBuilder):
    """特征服务测试生成器"""
    
    def __init__(self, template_manager: TestTemplateManager):
        """初始化特征服务测试生成器"""
        super().__init__(template_manager)
    
    def create_test_suite(self) -> TestSuite:
        """创建特征服务测试套件"""
        suite = TestSuite(
            id="feature_service_tests",
            name="特征工程服务API测试",
            description="特征工程服务的完整API测试套件"
        )
        
        suite.scenarios.append(self._create_feature_extraction_scenario())
        suite.scenarios.append(self._create_feature_calculation_scenario())
        suite.scenarios.append(self._create_feature_normalization_scenario())
        
        return suite
    
    def _create_feature_extraction_scenario(self) -> TestScenario:
        """创建特征提取场景"""
        scenario = self.create_scenario(
            name="特征提取",
            description="测试特征提取功能",
            endpoint="/api/v1/features/extract",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="提取技术指标",
            description="测试技术指标的提取",
            priority="high",
            test_steps=[
                {"step": 1, "action": "准备市场数据"},
                {"step": 2, "action": "指定要提取的技术指标"},
                {"step": 3, "action": "调用特征提取API"},
                {"step": 4, "action": "验证特征计算结果"}
            ],
            expected_results=[
                "成功提取指定的技术指标",
                "特征值计算正确",
                "返回格式符合规范"
            ],
            tags=["feature_extraction", "technical_indicators"]
        ))
        
        return scenario
    
    def _create_feature_calculation_scenario(self) -> TestScenario:
        """创建特征计算场景"""
        scenario = self.create_scenario(
            name="特征计算",
            description="测试特征计算功能",
            endpoint="/api/v1/features/calculate",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="计算多维特征",
            description="测试同时计算多个特征",
            priority="medium",
            test_steps=[
                {"step": 1, "action": "准备输入数据"},
                {"step": 2, "action": "指定特征列表"},
                {"step": 3, "action": "调用计算API"}
            ],
            expected_results=[
                "所有特征都被计算",
                "结果准确性验证通过"
            ],
            tags=["feature_calculation", "batch"]
        ))
        
        return scenario
    
    def _create_feature_normalization_scenario(self) -> TestScenario:
        """创建特征归一化场景"""
        scenario = self.create_scenario(
            name="特征归一化",
            description="测试特征归一化功能",
            endpoint="/api/v1/features/normalize",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="Z-score归一化",
            description="测试Z-score归一化",
            priority="medium",
            test_steps=[
                {"step": 1, "action": "准备特征数据"},
                {"step": 2, "action": "调用归一化API"},
                {"step": 3, "action": "验证归一化结果"}
            ],
            expected_results=[
                "归一化后均值接近0",
                "归一化后标准差接近1"
            ],
            tags=["normalization", "zscore"]
        ))
        
        return scenario

    def get_service_type(self) -> str:
        """获取服务类型"""
        return "feature"


class TradingServiceTestGenerator(TestCaseBuilder):
    """交易服务测试生成器"""
    
    def __init__(self, template_manager: TestTemplateManager):
        """初始化交易服务测试生成器"""
        super().__init__(template_manager)
    
    def create_test_suite(self) -> TestSuite:
        """创建交易服务测试套件"""
        suite = TestSuite(
            id="trading_service_tests",
            name="交易服务API测试",
            description="交易服务的完整API测试套件"
        )
        
        suite.scenarios.append(self._create_order_management_scenario())
        suite.scenarios.append(self._create_position_management_scenario())
        suite.scenarios.append(self._create_risk_control_scenario())
        
        return suite
    
    def _create_order_management_scenario(self) -> TestScenario:
        """创建订单管理场景"""
        scenario = self.create_scenario(
            name="订单管理",
            description="测试订单的创建、修改、取消",
            endpoint="/api/v1/trading/orders",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="创建订单 - 市价单",
            description="测试创建市价订单",
            priority="critical",
            test_steps=[
                {"step": 1, "action": "准备订单参数"},
                {"step": 2, "action": "提交订单"},
                {"step": 3, "action": "验证订单状态"}
            ],
            expected_results=[
                "订单创建成功",
                "返回订单ID",
                "订单状态为pending或filled"
            ],
            tags=["order", "market_order"]
        ))
        
        return scenario
    
    def _create_position_management_scenario(self) -> TestScenario:
        """创建持仓管理场景"""
        scenario = self.create_scenario(
            name="持仓管理",
            description="测试持仓查询和管理",
            endpoint="/api/v1/trading/positions",
            method="GET"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="查询当前持仓",
            description="测试查询用户当前持仓",
            priority="high",
            test_steps=[
                {"step": 1, "action": "发送持仓查询请求"},
                {"step": 2, "action": "验证返回数据"}
            ],
            expected_results=[
                "返回所有持仓信息",
                "包含持仓数量、成本、市值等"
            ],
            tags=["position", "query"]
        ))
        
        return scenario
    
    def _create_risk_control_scenario(self) -> TestScenario:
        """创建风控场景"""
        scenario = self.create_scenario(
            name="风险控制",
            description="测试风控规则和限制",
            endpoint="/api/v1/trading/risk-check",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="风险检查 - 仓位限制",
            description="测试仓位限制风控",
            priority="critical",
            test_steps=[
                {"step": 1, "action": "准备超限订单"},
                {"step": 2, "action": "提交风险检查"},
                {"step": 3, "action": "验证被拦截"}
            ],
            expected_results=[
                "风险检查失败",
                "返回拒绝原因",
                "订单未被执行"
            ],
            tags=["risk_control", "position_limit"]
        ))
        
        return scenario

    def get_service_type(self) -> str:
        """获取服务类型"""
        return "trading"


class MonitoringServiceTestGenerator(TestCaseBuilder):
    """监控服务测试生成器"""
    
    def __init__(self, template_manager: TestTemplateManager):
        """初始化监控服务测试生成器"""
        super().__init__(template_manager)
    
    def create_test_suite(self) -> TestSuite:
        """创建监控服务测试套件"""
        suite = TestSuite(
            id="monitoring_service_tests",
            name="监控服务API测试",
            description="监控服务的完整API测试套件"
        )
        
        suite.scenarios.append(self._create_metrics_collection_scenario())
        suite.scenarios.append(self._create_health_check_scenario())
        suite.scenarios.append(self._create_alert_management_scenario())
        
        return suite
    
    def _create_metrics_collection_scenario(self) -> TestScenario:
        """创建指标收集场景"""
        scenario = self.create_scenario(
            name="指标收集",
            description="测试系统指标收集功能",
            endpoint="/api/v1/monitoring/metrics",
            method="GET"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="获取系统指标",
            description="测试获取系统性能指标",
            priority="high",
            test_steps=[
                {"step": 1, "action": "请求系统指标"},
                {"step": 2, "action": "验证指标数据"}
            ],
            expected_results=[
                "返回CPU、内存、磁盘等指标",
                "指标值在合理范围内"
            ],
            tags=["metrics", "system"]
        ))
        
        return scenario
    
    def _create_health_check_scenario(self) -> TestScenario:
        """创建健康检查场景"""
        scenario = self.create_scenario(
            name="健康检查",
            description="测试服务健康检查",
            endpoint="/api/v1/monitoring/health",
            method="GET"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="系统健康检查",
            description="测试系统整体健康状态",
            priority="critical",
            test_steps=[
                {"step": 1, "action": "发送健康检查请求"},
                {"step": 2, "action": "验证各组件状态"}
            ],
            expected_results=[
                "返回健康状态",
                "列出所有组件状态",
                "整体健康评分"
            ],
            tags=["health_check", "status"]
        ))
        
        return scenario
    
    def _create_alert_management_scenario(self) -> TestScenario:
        """创建告警管理场景"""
        scenario = self.create_scenario(
            name="告警管理",
            description="测试告警创建和管理",
            endpoint="/api/v1/monitoring/alerts",
            method="POST"
        )
        
        scenario.test_cases.append(self.create_test_case(
            title="创建告警规则",
            description="测试创建新的告警规则",
            priority="high",
            test_steps=[
                {"step": 1, "action": "准备告警规则配置"},
                {"step": 2, "action": "提交告警规则"},
                {"step": 3, "action": "验证规则生效"}
            ],
            expected_results=[
                "告警规则创建成功",
                "规则开始生效",
                "触发条件正确"
            ],
            tags=["alert", "rule"]
        ))

        return scenario

    def get_service_type(self) -> str:
        """获取服务类型"""
        return "monitoring"

