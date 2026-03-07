"""
验收测试管理器

Acceptance Test Manager - manages test library and test plans.

Extracted from user_acceptance_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from .enums import UserRole, AcceptanceTestType, TestScenario
from .models import UserAcceptanceTest, TestExecutionResult, TestStatus

class AcceptanceTestManager:

    """验收测试管理器"""

    def __init__(self):

        self.acceptance_tests: List[UserAcceptanceTest] = []
        self.execution_results: List[TestExecutionResult] = []
        self.test_library = self._initialize_test_library()

        # 测试环境配置
        self.environments = {
            'development': {
                'url': 'http://dev.rqa2025.com',
                'database': 'dev_db',
                'features': ['basic_functionality']
            },
            'staging': {
                'url': 'http://staging.rqa2025.com',
                'database': 'staging_db',
                'features': ['full_functionality', 'performance_monitoring']
            },
            'production': {
                'url': 'http://rqa2025.com',
                'database': 'prod_db',
                'features': ['full_functionality', 'high_availability', 'security']
            }
        }

    def _initialize_test_library(self) -> Dict[str, List[UserAcceptanceTest]]:
        """初始化测试库"""
        return {
            'business_functionality': self._create_business_functionality_tests(),
            'user_interface': self._create_ui_tests(),
            'performance': self._create_performance_tests(),
            'security': self._create_security_tests(),
            'compliance': self._create_compliance_tests(),
            'integration': self._create_integration_tests()
        }

    def _create_business_functionality_tests(self) -> List[UserAcceptanceTest]:
        """创建业务功能测试"""
        return [
            UserAcceptanceTest(
                test_id="BF001",
                test_name="市场数据实时处理验证",
                test_type=AcceptanceTestType.BUSINESS_FUNCTIONALITY,
                scenario=TestScenario.MARKET_DATA_PROCESSING,
                target_users=[UserRole.TRADER, UserRole.QUANTITATIVE_ANALYST],
                prerequisites=["系统登录", "市场数据源连接"],
                test_steps=[
                    {"step": 1, "action": "登录系统", "expected": "成功进入主界面"},
                    {"step": 2, "action": "查看实时市场数据", "expected": "数据实时更新"},
                    {"step": 3, "action": "验证数据准确性", "expected": "数据与外部源一致"},
                    {"step": 4, "action": "测试数据导出功能", "expected": "成功导出数据"}
                ],
                expected_results=[
                    "市场数据延迟小于1秒",
                    "数据准确率大于99.9%",
                    "支持多种数据格式",
                    "实时告警功能正常"
                ],
                acceptance_criteria=[
                    "数据延迟<1秒",
                    "准确率>99.9%",
                    "支持5种以上数据源",
                    "用户界面响应<2秒"
                ],
                priority=1,
                estimated_duration=45
            ),
            UserAcceptanceTest(
                test_id="BF002",
                test_name="自动化交易执行验证",
                test_type=AcceptanceTestType.BUSINESS_FUNCTIONALITY,
                scenario=TestScenario.AUTOMATED_TRADING_EXECUTION,
                target_users=[UserRole.TRADER, UserRole.PORTFOLIO_MANAGER],
                prerequisites=["交易账户配置", "风控规则设置"],
                test_steps=[
                    {"step": 1, "action": "配置交易策略", "expected": "策略参数正确保存"},
                    {"step": 2, "action": "启动自动化交易", "expected": "系统开始执行交易"},
                    {"step": 3, "action": "监控交易执行", "expected": "交易按预期执行"},
                    {"step": 4, "action": "验证交易结果", "expected": "成交价格符合预期"}
                ],
                expected_results=[
                    "交易执行准确率>99%",
                    "平均执行时间<100ms",
                    "风控规则有效执行",
                    "实时监控数据更新"
                ],
                acceptance_criteria=[
                    "执行准确率>99%",
                    "执行时间<100ms",
                    "风控生效率100%",
                    "用户界面实时更新"
                ],
                priority=1,
                estimated_duration=60
            )
        ]

    def _create_ui_tests(self) -> List[UserAcceptanceTest]:
        """创建用户界面测试"""
        return [
            UserAcceptanceTest(
                test_id="UI001",
                test_name="交易界面操作验证",
                test_type=AcceptanceTestType.USER_INTERFACE,
                scenario=TestScenario.AUTOMATED_TRADING_EXECUTION,
                target_users=[UserRole.TRADER],
                prerequisites=["用户登录"],
                test_steps=[
                    {"step": 1, "action": "访问交易界面", "expected": "界面正常加载"},
                    {"step": 2, "action": "测试界面响应速度", "expected": "操作响应<2秒"},
                    {"step": 3, "action": "验证界面布局", "expected": "布局符合设计规范"},
                    {"step": 4, "action": "测试跨设备兼容性", "expected": "在不同设备上正常显示"}
                ],
                expected_results=[
                    "界面加载时间<3秒",
                    "操作响应时间<2秒",
                    "界面布局合理美观",
                    "支持多设备访问"
                ],
                acceptance_criteria=[
                    "加载时间<3秒",
                    "响应时间<2秒",
                    "无界面错误",
                    "兼容主流浏览器"
                ],
                priority=2,
                estimated_duration=30
            )
        ]

    def _create_performance_tests(self) -> List[UserAcceptanceTest]:
        """创建性能测试"""
        return [
            UserAcceptanceTest(
                test_id="PERF001",
                test_name="系统性能基准测试",
                test_type=AcceptanceTestType.PERFORMANCE_REQUIREMENTS,
                scenario=TestScenario.MARKET_DATA_PROCESSING,
                target_users=[UserRole.SYSTEM_ADMINISTRATOR],
                prerequisites=["系统部署完成"],
                test_steps=[
                    {"step": 1, "action": "准备测试数据", "expected": "测试数据准备完成"},
                    {"step": 2, "action": "执行性能测试", "expected": "测试脚本正常运行"},
                    {"step": 3, "action": "收集性能指标", "expected": "获得完整的性能数据"},
                    {"step": 4, "action": "分析测试结果", "expected": "生成性能分析报告"}
                ],
                expected_results=[
                    "响应时间满足SLA",
                    "系统吞吐量达标",
                    "资源使用率正常",
                    "无性能瓶颈"
                ],
                acceptance_criteria=[
                    "平均响应时间<500ms",
                    "并发用户数>1000",
                    "CPU使用率<70%",
                    "内存使用率<80%"
                ],
                priority=2,
                estimated_duration=90
            )
        ]

    def _create_security_tests(self) -> List[UserAcceptanceTest]:
        """创建安全测试"""
        return [
            UserAcceptanceTest(
                test_id="SEC001",
                test_name="用户认证安全验证",
                test_type=AcceptanceTestType.SECURITY_REQUIREMENTS,
                scenario=TestScenario.COMPLIANCE_CHECKING,
                target_users=[UserRole.SYSTEM_ADMINISTRATOR, UserRole.COMPLIANCE_OFFICER],
                prerequisites=["安全配置完成"],
                test_steps=[
                    {"step": 1, "action": "测试用户登录", "expected": "正常登录流程"},
                    {"step": 2, "action": "验证密码策略", "expected": "密码要求符合安全标准"},
                    {"step": 3, "action": "测试多因素认证", "expected": "MFA功能正常"},
                    {"step": 4, "action": "验证访问控制", "expected": "权限控制有效"}
                ],
                expected_results=[
                    "用户认证100 % 成功",
                    "密码策略符合要求",
                    "MFA功能正常工作",
                    "访问控制严格执行"
                ],
                acceptance_criteria=[
                    "认证成功率100%",
                    "密码复杂度达标",
                    "MFA验证通过",
                    "无越权访问风险"
                ],
                priority=1,
                estimated_duration=45
            )
        ]

    def _create_compliance_tests(self) -> List[UserAcceptanceTest]:
        """创建合规测试"""
        return [
            UserAcceptanceTest(
                test_id="COMP001",
                test_name="交易合规检查验证",
                test_type=AcceptanceTestType.COMPLIANCE_REQUIREMENTS,
                scenario=TestScenario.COMPLIANCE_CHECKING,
                target_users=[UserRole.COMPLIANCE_OFFICER, UserRole.RISK_MANAGER],
                prerequisites=["合规规则配置"],
                test_steps=[
                    {"step": 1, "action": "配置合规规则", "expected": "规则配置成功"},
                    {"step": 2, "action": "执行合规检查", "expected": "检查结果准确"},
                    {"step": 3, "action": "验证告警机制", "expected": "违规及时告警"},
                    {"step": 4, "action": "生成合规报告", "expected": "报告内容完整"}
                ],
                expected_results=[
                    "合规检查100 % 准确",
                    "告警响应及时",
                    "报告生成自动化",
                    "审计日志完整"
                ],
                acceptance_criteria=[
                    "检查准确率>99%",
                    "告警延迟<5秒",
                    "报告自动生成",
                    "审计追踪完整"
                ],
                priority=1,
                estimated_duration=60
            )
        ]

    def _create_integration_tests(self) -> List[UserAcceptanceTest]:
        """创建集成测试"""
        return [
            UserAcceptanceTest(
                test_id="INT001",
                test_name="系统集成场景验证",
                test_type=AcceptanceTestType.INTEGRATION_SCENARIOS,
                scenario=TestScenario.SYSTEM_BACKUP_RECOVERY,
                target_users=[UserRole.SYSTEM_ADMINISTRATOR],
                prerequisites=["系统各模块部署完成"],
                test_steps=[
                    {"step": 1, "action": "验证模块间通信", "expected": "通信正常"},
                    {"step": 2, "action": "测试数据流转", "expected": "数据正确传递"},
                    {"step": 3, "action": "验证接口兼容性", "expected": "接口调用成功"},
                    {"step": 4, "action": "测试系统集成", "expected": "整体功能正常"}
                ],
                expected_results=[
                    "模块间通信正常",
                    "数据流转无丢失",
                    "接口兼容性良好",
                    "系统集成无误"
                ],
                acceptance_criteria=[
                    "通信成功率100%",
                    "数据完整性100%",
                    "接口兼容性100%",
                    "集成测试通过"
                ],
                priority=2,
                estimated_duration=75
            )
        ]

    def get_test_plan(self, user_role: Optional[UserRole] = None,


                      test_type: Optional[AcceptanceTestType] = None) -> List[UserAcceptanceTest]:
        """获取测试计划"""
        tests = []

        for category, test_list in self.test_library.items():
            for test in test_list:
                # 按用户角色过滤
                if user_role and user_role not in test.target_users:
                    continue

                # 按测试类型过滤
                if test_type and test.test_type != test_type:
                    continue

                tests.append(test)

        # 按优先级排序
        tests.sort(key=lambda x: x.priority)

        return tests

    def assign_test_execution(self, test_id: str, tester: str, environment: str = "staging") -> bool:
        """分配测试执行"""
        for test in self.acceptance_tests:
            if test.test_id == test_id:
                test.assigned_tester = tester
                test.status = TestStatus.RUNNING

                # 创建执行结果记录
                execution_result = TestExecutionResult(
                    execution_id=f"exec_{test_id}_{int(time.time())}",
                    test_id=test_id,
                    start_time=datetime.now(),
                    end_time=None,
                    duration=None,
                    status=TestStatus.RUNNING,
                    executed_by=tester,
                    environment=environment,
                    results={},

                    defects_found=[],
                    user_feedback=[]
                )

                self.execution_results.append(execution_result)
                return True

        return False

    def record_test_result(self, execution_id: str, status: TestStatus,


                           results: Dict[str, Any], defects: List[Dict[str, Any]],
                           feedback: List[Dict[str, Any]]) -> bool:
        """记录测试结果"""
        for execution in self.execution_results:
            if execution.execution_id == execution_id:
                execution.end_time = datetime.now()
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
                execution.status = status
                execution.results = results
                execution.defects_found = defects
                execution.user_feedback = feedback

                # 更新测试状态
                for test in self.acceptance_tests:
                    if test.test_id == execution.test_id:
                        test.status = status
                        break

                return True

        return False

    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = len(self.acceptance_tests)
        completed_tests = sum(
            1 for test in self.acceptance_tests if test.status != TestStatus.PENDING)
        passed_tests = sum(1 for test in self.acceptance_tests if test.status == TestStatus.PASSED)
        failed_tests = sum(1 for test in self.acceptance_tests if test.status == TestStatus.FAILED)

        # 按类型统计
        type_stats = {}
        for test in self.acceptance_tests:
            test_type = test.test_type.value
            if test_type not in type_stats:
                type_stats[test_type] = {'total': 0, 'passed': 0, 'failed': 0, 'pending': 0}

            type_stats[test_type]['total'] += 1
            if test.status == TestStatus.PASSED:
                type_stats[test_type]['passed'] += 1
            elif test.status == TestStatus.FAILED:
                type_stats[test_type]['failed'] += 1
            else:
                type_stats[test_type]['pending'] += 1

        # 按用户角色统计
        role_stats = {}
        for test in self.acceptance_tests:
            for role in test.target_users:
                role_name = role.value
                if role_name not in role_stats:
                    role_stats[role_name] = {'total': 0, 'completed': 0}

                role_stats[role_name]['total'] += 1
        if test.status != TestStatus.PENDING:
            role_stats[role_name]['completed'] += 1

        # 缺陷统计
        total_defects = sum(len(execution.defects_found) for execution in self.execution_results)
        critical_defects = sum(
            len([d for d in execution.defects_found if d.get('severity') == 'critical'])
            for execution in self.execution_results
        )

        return {
            'summary': {
                'total_tests': total_tests,
                'completed_tests': completed_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'completion_rate': completed_tests / total_tests if total_tests > 0 else 0
            },
            'type_statistics': type_stats,
            'role_statistics': role_stats,
            'defect_summary': {
                'total_defects': total_defects,
                'critical_defects': critical_defects,
                'defects_per_test': total_defects / total_tests if total_tests > 0 else 0
            },
            'execution_summary': {
                'total_executions': len(self.execution_results),
                'avg_execution_time': np.mean([
                    execution.duration for execution in self.execution_results
                    if execution.duration
                ]) if self.execution_results else 0,
                'environments_used': list(set(
                    execution.environment for execution in self.execution_results
                ))
            },
            'test_details': [test.to_dict() for test in self.acceptance_tests],
            # 最近20个
            'execution_details': [execution.to_dict() for execution in self.execution_results[-20:]]
        }




__all__ = ['AcceptanceTestManager']
