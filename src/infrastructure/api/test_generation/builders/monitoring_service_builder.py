"""
监控服务测试构建器

负责构建监控服务相关的测试套件
"""

from typing import List
from .base_builder import BaseTestBuilder
from ..models import TestSuite, TestScenario, TestCase


class MonitoringServiceTestBuilder(BaseTestBuilder):
    """
    监控服务测试构建器
    
    职责：
    - 构建监控服务API的测试场景
    - 生成健康检查、指标查询、告警管理等测试用例
    """
    
    def build_test_suite(self) -> TestSuite:
        """构建监控服务测试套件"""
        suite = TestSuite(
            id="monitoring_service_tests",
            name="监控服务API测试",
            description="RQA2025监控服务的完整API测试套件"
        )
        
        # 添加各类测试场景
        suite.scenarios.append(self._build_health_check_scenario())
        suite.scenarios.append(self._build_metrics_query_scenario())
        suite.scenarios.append(self._build_alert_management_scenario())
        
        return suite
    
    def _build_health_check_scenario(self) -> TestScenario:
        """构建健康检查测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="health_check",
            name="健康检查",
            description="测试系统健康检查功能",
            endpoint="/api/v1/health",
            method="GET",
            variables={}
        )
        
        # 正常健康检查
        scenario.test_cases.append(self._create_test_case(
            case_id="health_check_normal",
            title="正常健康检查",
            description="系统正常运行时的健康检查",
            priority="critical",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回系统健康状态",
                "包含各组件状态信息"
            ],
            tags=["monitoring", "health", "normal"]
        ))
        
        # 详细健康检查
        scenario.test_cases.append(self._create_test_case(
            case_id="health_check_detailed",
            title="详细健康检查",
            description="获取系统详细的健康状态信息",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "包含各服务的详细状态",
                "包含数据库、Redis、MQ等依赖状态"
            ],
            tags=["monitoring", "health", "detailed"]
        ))

        return scenario

    def get_supported_operations(self) -> List[str]:
        """获取支持的操作列表"""
        return ["health_check", "metrics_query", "alert_management"]
    
    def _build_metrics_query_scenario(self) -> TestScenario:
        """构建指标查询测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="metrics_query",
            name="性能指标查询",
            description="测试性能指标的查询功能",
            endpoint="/api/v1/metrics",
            method="GET",
            variables={
                "metric_name": "request_count",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-12-31T23:59:59Z"
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="metrics_query_request_count",
            title="查询请求计数指标",
            description="查询指定时间范围的请求计数",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回时间序列数据",
                "数据点包含时间戳和值"
            ],
            tags=["monitoring", "metrics", "query"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="metrics_query_aggregation",
            title="指标聚合查询",
            description="测试指标的聚合统计功能",
            priority="medium",
            category="functional",
            expected_results=[
                "返回状态码200",
                "支持sum、avg、max、min聚合",
                "聚合结果计算正确"
            ],
            tags=["monitoring", "metrics", "aggregation"]
        ))

        return scenario

    def get_supported_operations(self) -> List[str]:
        """获取支持的操作列表"""
        return ["health_check", "metrics_query", "alert_management"]
    
    def _build_alert_management_scenario(self) -> TestScenario:
        """构建告警管理测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="alert_management",
            name="告警管理",
            description="测试告警的创建、查询和管理功能",
            endpoint="/api/v1/alerts",
            method="GET",
            variables={
                "severity": "critical",
                "status": "active"
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="alert_query_active",
            title="查询活跃告警",
            description="查询系统当前的活跃告警",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回活跃告警列表",
                "包含告警级别、时间、描述"
            ],
            tags=["monitoring", "alert", "query"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="alert_acknowledge",
            title="确认告警",
            description="测试告警确认功能",
            priority="medium",
            category="functional",
            expected_results=[
                "返回状态码200",
                "告警状态更新为acknowledged",
                "记录确认人和时间"
            ],
            tags=["monitoring", "alert", "acknowledge"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="alert_resolve",
            title="解决告警",
            description="测试告警解决功能",
            priority="medium",
            category="functional",
            expected_results=[
                "返回状态码200",
                "告警状态更新为resolved",
                "记录解决方案和时间"
            ],
            tags=["monitoring", "alert", "resolve"]
        ))

        return scenario

    def get_supported_operations(self) -> List[str]:
        """获取支持的操作列表"""
        return ["health_check", "metrics_query", "alert_management"]

