#!/usr/bin/env python3
"""
RQA2025用户验收测试执行系统

构建专业的用户验收测试和业务验证平台
    创建时间: 2025年3月
"""

import sys
import os
import numpy as np
import logging
import time
import queue
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import concurrent.futures

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from testing.system_integration_tester import (
        TestResult, TestStatus, IntegrationTestType
    )
    print("✅ 系统集成测试器导入成功")
except ImportError as e:
    print(f"❌ 系统集成测试器导入失败: {e}")
    # 创建简化的替代类用于演示

    class TestResult:

        def __init__(self, **kwargs):

            for k, v in kwargs.items():
                setattr(self, k, v)

    class TestStatus(Enum):

        PASSED = "passed"
        FAILED = "failed"

    class IntegrationTestType(Enum):

        END_TO_END_TEST = "end_to_end_test"


class UserRole(Enum):

    """用户角色枚举"""
    TRADER = "trader"                    # 交易员
    RISK_MANAGER = "risk_manager"        # 风险经理
    PORTFOLIO_MANAGER = "portfolio_manager"  # 投资组合经理
    QUANTITATIVE_ANALYST = "quantitative_analyst"  # 量化分析师
    SYSTEM_ADMINISTRATOR = "system_administrator"  # 系统管理员
    COMPLIANCE_OFFICER = "compliance_officer"  # 合规官
    BUSINESS_ANALYST = "business_analyst"  # 业务分析师


class AcceptanceTestType(Enum):

    """验收测试类型枚举"""
    BUSINESS_FUNCTIONALITY = "business_functionality"    # 业务功能测试
    USER_INTERFACE = "user_interface"                    # 用户界面测试
    PERFORMANCE_REQUIREMENTS = "performance_requirements"  # 性能需求测试
    SECURITY_REQUIREMENTS = "security_requirements"        # 安全需求测试
    COMPLIANCE_REQUIREMENTS = "compliance_requirements"    # 合规需求测试
    INTEGRATION_SCENARIOS = "integration_scenarios"        # 集成场景测试
    BUSINESS_WORKFLOW = "business_workflow"                # 业务流程测试
    DATA_ACCURACY = "data_accuracy"                        # 数据准确性测试


class TestScenario(Enum):

    """测试场景枚举"""
    MARKET_DATA_PROCESSING = "market_data_processing"      # 市场数据处理
    MODEL_TRAINING_DEPLOYMENT = "model_training_deployment"  # 模型训练部署
    RISK_MONITORING_ALERT = "risk_monitoring_alert"        # 风险监控告警
    AUTOMATED_TRADING_EXECUTION = "automated_trading_execution"  # 自动化交易执行
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"        # 投资组合再平衡
    REPORT_GENERATION = "report_generation"                # 报告生成
    SYSTEM_BACKUP_RECOVERY = "system_backup_recovery"      # 系统备份恢复
    COMPLIANCE_CHECKING = "compliance_checking"            # 合规检查


@dataclass
class UserAcceptanceTest:

    """用户验收测试"""
    test_id: str
    test_name: str
    test_type: AcceptanceTestType
    scenario: TestScenario
    target_users: List[UserRole]
    prerequisites: List[str]
    test_steps: List[Dict[str, Any]]
    expected_results: List[str]
    acceptance_criteria: List[str]
    priority: int  # 1 - 5, 1最高
    estimated_duration: int  # 分钟
    status: TestStatus = TestStatus.PENDING
    assigned_tester: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'scenario': self.scenario.value,
            'target_users': [role.value for role in self.target_users],
            'prerequisites': self.prerequisites,
            'test_steps': self.test_steps,
            'expected_results': self.expected_results,
            'acceptance_criteria': self.acceptance_criteria,
            'priority': self.priority,
            'estimated_duration': self.estimated_duration,
            'status': self.status.value,
            'assigned_tester': self.assigned_tester
        }


@dataclass
class TestExecutionResult:

    """测试执行结果"""
    execution_id: str
    test_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: TestStatus
    executed_by: str
    environment: str
    results: Dict[str, Any]

    defects_found: List[Dict[str, Any]]
    user_feedback: List[Dict[str, Any]]
    screenshots: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'test_id': self.test_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'status': self.status.value,
            'executed_by': self.executed_by,
            'environment': self.environment,
            'results': self.results,
            'defects_found': self.defects_found,
            'user_feedback': self.user_feedback,
            'screenshots': self.screenshots
        }


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


class UserAcceptanceTestExecutor:

    """用户验收测试执行器"""

    def __init__(self):

        self.test_manager = AcceptanceTestManager()
        self.is_executing = False
        self.execution_queue = queue.Queue()
        self.executors = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        # 执行统计
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'defects_found': 0
        }

    def start_acceptance_testing(self, user_role: Optional[UserRole] = None,


                                 environment: str = "staging"):
        """启动验收测试"""
        if self.is_executing:
            logger.warning("验收测试已在执行中")
            return

        self.is_executing = True

        # 获取测试计划
        test_plan = self.test_manager.get_test_plan(user_role)

        print(f"📋 加载测试计划: {len(test_plan)} 个测试用例")

        # 分配测试执行
        for test in test_plan:
            if not self.is_executing:
                break

            # 自动分配测试人员
            tester = self._assign_tester(test)
            success = self.test_manager.assign_test_execution(
                test.test_id, tester, environment
            )

        if success:
            # 提交执行任务
            future = self.executors.submit(
                self._execute_single_test,
                test.test_id,
                tester,
                environment
            )

            print(f"✅ 已分配测试: {test.test_name} -> {tester}")

        logger.info("用户验收测试已启动")

    def stop_acceptance_testing(self):
        """停止验收测试"""
        self.is_executing = False
        self.executors.shutdown(wait=True)
        logger.info("用户验收测试已停止")

    def _assign_tester(self, test: UserAcceptanceTest) -> str:
        """分配测试人员"""
        # 简化的测试人员分配逻辑
        testers = {
            UserRole.TRADER: ["张交易员", "李交易员"],
            UserRole.RISK_MANAGER: ["王风险经理", "赵风险经理"],
            UserRole.PORTFOLIO_MANAGER: ["刘投资经理", "陈投资经理"],
            UserRole.QUANTITATIVE_ANALYST: ["杨量化分析师", "黄量化分析师"],
            UserRole.SYSTEM_ADMINISTRATOR: ["周系统管理员", "吴系统管理员"],
            UserRole.COMPLIANCE_OFFICER: ["郑合规官", "马合规官"],
            UserRole.BUSINESS_ANALYST: ["孙业务分析师", "胡业务分析师"]
        }

        for role in test.target_users:
            if role in testers and testers[role]:
                return testers[role][0]

        return "默认测试员"

    def _execute_single_test(self, test_id: str, tester: str, environment: str):
        """执行单个测试"""
        try:
            print(f"🎯 开始执行测试: {test_id} (测试员: {tester})")

            # 查找测试用例
            test = None
            for t in self.test_manager.acceptance_tests:
                if t.test_id == test_id:
                    test = t
                    break

            if not test:
                print(f"❌ 未找到测试用例: {test_id}")
                return

            # 查找执行记录
            execution = None
            for exec_result in self.test_manager.execution_results:
                if exec_result.test_id == test_id and exec_result.executed_by == tester:
                    execution = exec_result
                    break

            if not execution:
                print(f"❌ 未找到执行记录: {test_id}")
                return

            # 模拟测试执行
            execution_time = test.estimated_duration * 60 * np.random.uniform(0.8, 1.2)  # 分钟转秒
            time.sleep(min(execution_time, 30))  # 限制最大等待时间

            # 模拟测试结果
            success_probability = {
                AcceptanceTestType.BUSINESS_FUNCTIONALITY: 0.90,
                AcceptanceTestType.USER_INTERFACE: 0.95,
                AcceptanceTestType.PERFORMANCE_REQUIREMENTS: 0.85,
                AcceptanceTestType.SECURITY_REQUIREMENTS: 0.95,
                AcceptanceTestType.COMPLIANCE_REQUIREMENTS: 0.90,
                AcceptanceTestType.INTEGRATION_SCENARIOS: 0.80
            }

            prob = success_probability.get(test.test_type, 0.85)
            is_success = np.random.random() < prob

            if is_success:
                status = TestStatus.PASSED
                results = {
                    'steps_completed': len(test.test_steps),
                    'criteria_met': len(test.acceptance_criteria),
                    'user_satisfaction': np.random.uniform(4.0, 5.0),
                    'performance_rating': np.random.uniform(4.0, 5.0)
                }

                defects = []
                feedback = [{
                    'user': tester,
                    'rating': 5,
                    'comments': '功能符合预期，操作流畅'
                }]
            else:
                status = TestStatus.FAILED
                results = {
                    'steps_completed': np.random.randint(1, len(test.test_steps)),
                    'criteria_met': np.random.randint(0, len(test.acceptance_criteria)),
                    'user_satisfaction': np.random.uniform(1.0, 3.0),
                    'performance_rating': np.random.uniform(1.0, 3.0)
                }

                # 生成模拟缺陷

                defect_count = np.random.randint(1, 4)

                defects = []
                for i in range(defect_count):
                    severity = np.random.choice(['low', 'medium', 'high', 'critical'],
                                                p=[0.4, 0.3, 0.2, 0.1])

                    defects.append({
                        'defect_id': f"DEF_{test_id}_{i + 1}",
                        'title': f"测试缺陷 {i + 1}",
                        'severity': severity,
                        'description': f"在{test.test_name}中发现的{severity}级别缺陷",
                        'steps_to_reproduce': test.test_steps[:2],
                        'expected_behavior': test.expected_results[0],
                        'actual_behavior': '与预期不符',
                        'environment': environment
                    })

                feedback = [{
                    'user': tester,
                    'rating': 2,
                    'comments': f'发现 {len(defects)} 个缺陷需要修复'
                }]

            # 记录测试结果
            success = self.test_manager.record_test_result(
                execution.execution_id, status, results, defects, feedback
            )

            if success:
                self.stats['total_executions'] += 1
                if status == TestStatus.PASSED:
                    self.stats['successful_executions'] += 1
                else:
                    self.stats['failed_executions'] += 1

                self.stats['defects_found'] += len(defects)
                self.stats['avg_execution_time'] = (
                    (self.stats['avg_execution_time'] * (self.stats['total_executions'] - 1))
                    + execution_time
                ) / self.stats['total_executions']

                print(f"✅ 测试执行完成: {test_id} - {status.value} ({execution_time:.1f}s)")

                if defects:
                    print(f"   发现缺陷: {len(defects)} 个")
                    for defect in defects:
                        print(f"     - {defect['title']} ({defect['severity']})")

            else:
                print(f"❌ 记录测试结果失败: {test_id}")

        except Exception as e:
            logger.error(f"执行测试异常: {test_id} - {e}")
            print(f"❌ 测试执行异常: {test_id} - {e}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        report = self.test_manager.generate_test_report()

        return {
            'execution_stats': self.stats.copy(),
            'test_report': report,
            'is_executing': self.is_executing,
            'queue_size': self.execution_queue.qsize(),
            'active_workers': self.executors._threads.__len__() if hasattr(self.executors, '_threads') else 0
        }


def main():
    """主函数 - 用户验收测试执行系统演示"""
    print("👤 RQA2025用户验收测试执行系统")
    print("=" * 60)

    # 创建验收测试执行器
    executor = UserAcceptanceTestExecutor()

    print("✅ 用户验收测试执行器创建完成")
    print("   包含以下功能:")
    print("   - 自动化测试分配")
    print("   - 并发生测执行")
    print("   - 结果收集分析")
    print("   - 缺陷跟踪管理")

    # 加载测试库
    test_manager = executor.test_manager

    # 显示测试统计
    total_tests = sum(len(tests) for tests in test_manager.test_library.values())
    print(f"   测试库总计: {total_tests} 个测试用例")

    for category, tests in test_manager.test_library.items():
        print(f"     {category}: {len(tests)} 个")

    try:
        # 启动验收测试
        print("\n🚀 启动用户验收测试...")

        # 选择测试类型和用户角色
        test_types = [
            (UserRole.TRADER, "交易员测试"),
            (UserRole.RISK_MANAGER, "风险经理测试"),
            (UserRole.SYSTEM_ADMINISTRATOR, "系统管理员测试")
        ]

        for user_role, description in test_types:
            print(f"\n🎯 执行{description}...")
            test_plan = test_manager.get_test_plan(user_role)

            if test_plan:
                print(f"   找到 {len(test_plan)} 个相关测试用例")

                # 启动该角色的测试
                executor.start_acceptance_testing(user_role, "staging")

                # 等待一小段时间
                time.sleep(5)

                # 显示进度
                summary = executor.get_execution_summary()
                report = summary['test_report']['summary']

                print(f"   进度: {report['completed_tests']}/{report['total_tests']} 完成")
                print(f"   通过率: {report['pass_rate']:.1%}")

        # 等待测试完成
        print("\n⏳ 等待测试完成...")
        time.sleep(30)  # 等待30秒

        print("\n🎉 用户验收测试演示完成！")
        print("   测试已成功执行并收集用户反馈")
        print("   缺陷已记录并分配修复")

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在停止测试...")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止测试
        executor.stop_acceptance_testing()
        print("✅ 用户验收测试已停止")

        # 显示最终统计
        final_summary = executor.get_execution_summary()
        execution_stats = final_summary['execution_stats']
        report = final_summary['test_report']

        print("\n📋 最终测试统计:")
        print(f"   总执行数: {execution_stats['total_executions']}")
        print(f"   成功执行: {execution_stats['successful_executions']}")
        print(f"   失败执行: {execution_stats['failed_executions']}")
        print(f"   发现缺陷: {execution_stats['defects_found']}")
        print(f"   平均执行时间: {execution_stats['avg_execution_time']:.1f}秒")

        summary = report['summary']
        print("\n📊 测试通过情况:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   已完成: {summary['completed_tests']}")
        print(f"   通过测试: {summary['passed_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   通过率: {summary['pass_rate']:.1%}")
        print(f"   完成率: {summary['completion_rate']:.1%}")

        if report.get('defect_summary'):

            defects = report['defect_summary']
            print("\n🐛 缺陷统计:")
            print(f"   总缺陷数: {defects['total_defects']}")
            print(f"   严重缺陷: {defects['critical_defects']}")
            print(f"   平均每测: {defects['defects_per_test']:.2f} 个")

        print("\n✅ 用户验收测试系统验证完成！")
        print("   系统已成功演示了完整的UAT流程")
        print("   包括测试分配、执行、结果收集和缺陷管理")

    return executor


if __name__ == "__main__":
    executor = main()

# Logger setup
logger = logging.getLogger(__name__)
