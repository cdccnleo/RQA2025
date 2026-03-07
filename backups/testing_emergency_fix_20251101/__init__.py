"""
RQA2025 测试层模块

提供完整的测试框架和质量保障体系，包括测试用例管理、执行引擎和结果分析
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


class TestPriority(Enum):

    """测试优先级枚举"""
    CRITICAL = 1    # 关键测试 - 系统核心功能
    HIGH = 2        # 高优先级 - 重要功能
    MEDIUM = 3      # 中等优先级 - 一般功能
    LOW = 4         # 低优先级 - 辅助功能


class TestCategory(Enum):

    """测试类别枚举"""
    SMOKE = "smoke"                    # 冒烟测试
    FUNCTIONAL = "functional"          # 功能测试
    INTEGRATION = "integration"       # 集成测试
    REGRESSION = "regression"          # 回归测试
    PERFORMANCE = "performance"       # 性能测试
    SECURITY = "security"              # 安全测试
    COMPLIANCE = "compliance"          # 合规测试
    ACCEPTANCE = "acceptance"          # 验收测试


class TestStatus(Enum):

    """测试状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PASSED = "passed"          # 通过
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    BLOCKED = "blocked"        # 阻塞
    ERROR = "error"            # 错误


@dataclass
class TestCase:

    """测试用例类"""
    test_id: str
    name: str
    description: str
    category: TestCategory
    priority: TestPriority
    estimated_duration: float  # 预估执行时间（秒）
    dependencies: List[str]    # 依赖的测试用例ID
    tags: List[str]            # 标签
    test_function: Optional[Callable] = None

    # 运行时状态
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 统计信息
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_execution: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'success_rate': self.success_count / max(self.execution_count, 1) * 100
        }


@dataclass
class TestSuite:

    """测试套件类"""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase]
    priority: TestPriority = TestPriority.MEDIUM
    timeout: int = 3600  # 超时时间（秒）

    # 执行统计
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    execution_time: float = 0.0

    def __post_init__(self):

        self.total_tests = len(self.test_cases)


class TestExecutionResult:

    """测试执行结果类"""

    def __init__(self, test_case: TestCase):

        self.test_case = test_case
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.status = TestStatus.RUNNING
        self.result_details: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
        self.performance_metrics: Dict[str, float] = {}

    def complete(self, status: TestStatus, result_details: Optional[Dict[str, Any]] = None,


                 error_message: Optional[str] = None):
        """完成测试执行"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = status
        if result_details:
            self.result_details = result_details
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_case.test_id,
            'test_name': self.test_case.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }


class TestCaseManager:

    """
    测试用例管理器 - 支持分类管理和优先级调度

    Test case manager supporting categorization and priority scheduling
    """

    def __init__(self):

        self.test_cases: Dict[str, TestCase] = {}
        self.test_suites: Dict[str, TestSuite] = {}

        # 分类统计
        self.category_stats = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'failed': 0, 'pending': 0, 'running': 0
        })

        # 优先级统计
        self.priority_stats = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'failed': 0, 'pending': 0, 'running': 0
        })

        # 执行历史
        self.execution_history: List[TestExecutionResult] = []
        self.max_history_size = 10000

        # 配置
        self.parallel_execution = True
        self.max_concurrent_tests = 5
        self.default_timeout = 300

        self.logger = logging.getLogger(self.__class__.__name__)

    def register_test_case(self, test_case: TestCase):
        """
        注册测试用例

        Register test case

        Args:
            test_case: 测试用例对象
                      Test case object
        """
        if test_case.test_id in self.test_cases:
            self.logger.warning(f"Test case {test_case.test_id} already exists, updating...")

        self.test_cases[test_case.test_id] = test_case

        # 更新分类统计
        self.category_stats[test_case.category.value]['total'] += 1
        self.category_stats[test_case.category.value]['pending'] += 1

        # 更新优先级统计
        self.priority_stats[test_case.priority.value]['total'] += 1
        self.priority_stats[test_case.priority.value]['pending'] += 1

        self.logger.info(f"Registered test case: {test_case.name} ({test_case.test_id})")

    def unregister_test_case(self, test_id: str) -> bool:
        """
        注销测试用例

        Unregister test case

        Args:
            test_id: 测试用例ID
                    Test case ID

        Returns:
            bool: 是否成功注销
                 Whether successfully unregistered
        """
        if test_id not in self.test_cases:
            return False

        test_case = self.test_cases[test_id]

        # 更新统计信息
        self.category_stats[test_case.category.value]['total'] -= 1
        self.category_stats[test_case.category.value][test_case.status.value] -= 1

        self.priority_stats[test_case.priority.value]['total'] -= 1
        self.priority_stats[test_case.priority.value][test_case.status.value] -= 1

        del self.test_cases[test_id]

        self.logger.info(f"Unregistered test case: {test_id}")
        return True

    def create_test_suite(self, suite_id: str, name: str, description: str,


                          test_ids: List[str], priority: TestPriority = TestPriority.MEDIUM) -> Optional[TestSuite]:
        """
        创建测试套件

        Create test suite

        Args:
            suite_id: 套件ID
                     Suite ID
            name: 套件名称
                 Suite name
            description: 套件描述
                        Suite description
            test_ids: 测试用例ID列表
                     List of test case IDs
            priority: 套件优先级
                     Suite priority

        Returns:
            TestSuite: 创建的测试套件对象，如果创建失败返回None
                      Created test suite object, None if creation failed
        """
        test_cases = []
        for test_id in test_ids:
            if test_id in self.test_cases:
                test_cases.append(self.test_cases[test_id])
            else:
                self.logger.warning(f"Test case {test_id} not found, skipping...")
                return None

        suite = TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            test_cases=test_cases,
            priority=priority
        )

        self.test_suites[suite_id] = suite

        self.logger.info(f"Created test suite: {name} with {len(test_cases)} test cases")
        return suite

    def get_test_cases_by_category(self, category: TestCategory) -> List[TestCase]:
        """
        根据类别获取测试用例

        Get test cases by category

        Args:
            category: 测试类别
                     Test category

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tc.category == category
        ]

    def get_test_cases_by_priority(self, priority: TestPriority) -> List[TestCase]:
        """
        根据优先级获取测试用例

        Get test cases by priority

        Args:
            priority: 测试优先级
                     Test priority

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tc.priority == priority
        ]

    def get_test_cases_by_tag(self, tag: str) -> List[TestCase]:
        """
        根据标签获取测试用例

        Get test cases by tag

        Args:
            tag: 标签
                Tag

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tag in tc.tags
        ]

    def get_test_cases_for_execution(self, max_priority: Optional[TestPriority] = None,


                                     categories: Optional[List[TestCategory]] = None,
                                     tags: Optional[List[str]] = None) -> List[TestCase]:
        """
        获取待执行的测试用例（支持过滤条件）

        Get test cases for execution with filtering options

        Args:
            max_priority: 最大优先级（包含此优先级及以上的测试）
                         Maximum priority (includes this priority and higher)
            categories: 类别过滤列表
                       Category filter list
            tags: 标签过滤列表
                 Tag filter list

        Returns:
            List[TestCase]: 过滤后的测试用例列表
                           Filtered test case list
        """
        candidates = [
            tc for tc in self.test_cases.values()
            if tc.status in [TestStatus.PENDING, TestStatus.FAILED]
        ]

        # 优先级过滤
        if max_priority:
            candidates = [
                tc for tc in candidates
                if tc.priority.value <= max_priority.value
            ]

        # 类别过滤
        if categories:
            candidates = [
                tc for tc in candidates
                if tc.category in categories
            ]

        # 标签过滤
        if tags:
            candidates = [
                tc for tc in candidates
                if any(tag in tc.tags for tag in tags)
            ]

        # 按优先级排序（高优先级在前）
        candidates.sort(key=lambda tc: tc.priority.value)

        return candidates

    def execute_test_case(self, test_case: TestCase) -> TestExecutionResult:
        """
        执行单个测试用例

        Execute single test case

        Args:
            test_case: 测试用例对象
                      Test case object

        Returns:
            TestExecutionResult: 执行结果
                                Execution result
        """
        result = TestExecutionResult(test_case)

        # 更新测试用例状态
        test_case.status = TestStatus.RUNNING
        test_case.start_time = result.start_time
        test_case.execution_count += 1

        # 更新统计信息
        self._update_stats(test_case.category.value, TestStatus.PENDING, TestStatus.RUNNING)
        self._update_stats(test_case.priority.value, TestStatus.PENDING, TestStatus.RUNNING)

        try:
            # 执行测试
            if test_case.test_function:
                test_result = test_case.test_function()
            else:
                # 模拟测试执行
                time.sleep(min(test_case.estimated_duration, 5))  # 限制最大等待时间
                test_result = {"status": "passed", "details": "Mock test execution"}

            # 处理执行结果
            if test_result.get("status") == "passed":
                result.complete(TestStatus.PASSED, test_result)
                test_case.status = TestStatus.PASSED
                test_case.success_count += 1
                self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.PASSED)
                self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.PASSED)
            else:
                result.complete(TestStatus.FAILED, test_result, test_result.get("error"))
                test_case.status = TestStatus.FAILED
                test_case.failure_count += 1
                self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.FAILED)
                self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.FAILED)

        except Exception as e:
            result.complete(TestStatus.ERROR, error_message=str(e))
            test_case.status = TestStatus.ERROR
            test_case.failure_count += 1
            self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.ERROR)
            self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.ERROR)

        # 更新测试用例信息
        test_case.end_time = result.end_time
        test_case.duration = result.duration
        test_case.last_execution = result.end_time
        test_case.result_details = result.result_details
        test_case.error_message = result.error_message

        # 添加到执行历史
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

        return result

    def _update_stats(self, key: str, old_status: TestStatus, new_status: TestStatus):
        """更新统计信息"""
        if key in self.category_stats:
            self.category_stats[key][old_status.value] -= 1
            self.category_stats[key][new_status.value] += 1

        # 同时更新优先级统计（如果key是优先级值）
        try:
            priority_val = int(key)
            if priority_val in self.priority_stats:
                self.priority_stats[priority_val][old_status.value] -= 1
                self.priority_stats[priority_val][new_status.value] += 1
        except (ValueError, KeyError):
            pass

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        获取执行摘要

        Get execution summary

        Returns:
            dict: 执行摘要统计
                 Execution summary statistics
        """
        total_tests = len(self.test_cases)
        executed_tests = sum(1 for tc in self.test_cases.values() if tc.execution_count > 0)
        passed_tests = sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.PASSED)
        failed_tests = sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.FAILED)

        recent_executions = self.execution_history[-10:] if self.execution_history else []

        return {
            'total_test_cases': total_tests,
            'executed_test_cases': executed_tests,
            'passed_test_cases': passed_tests,
            'failed_test_cases': failed_tests,
            'pending_test_cases': total_tests - executed_tests,
            'execution_rate': executed_tests / max(total_tests, 1) * 100,
            'pass_rate': passed_tests / max(executed_tests, 1) * 100,
            'category_stats': dict(self.category_stats),
            'priority_stats': dict(self.priority_stats),
            'recent_executions': [result.to_dict() for result in recent_executions]
        }

    def export_test_cases(self, filepath: str) -> bool:
        """
        导出测试用例到文件

        Export test cases to file

        Args:
            filepath: 导出文件路径
                     Export file path

        Returns:
            bool: 导出是否成功
                 Whether export was successful
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_test_cases': len(self.test_cases),
                'test_cases': [tc.to_dict() for tc in self.test_cases.values()],
                'test_suites': [asdict(suite) for suite in self.test_suites.values()]
            }

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported {len(self.test_cases)} test cases to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export test cases: {e}")
            return False

    def import_test_cases(self, filepath: str) -> int:
        """
        从文件导入测试用例

        Import test cases from file

        Args:
            filepath: 导入文件路径
                     Import file path

        Returns:
            int: 成功导入的测试用例数量
                Number of successfully imported test cases
        """
        try:
            with open(filepath, 'r', encoding='utf - 8') as f:
                import_data = json.load(f)

            imported_count = 0
            for tc_data in import_data.get('test_cases', []):
                # 重建TestCase对象
                test_case = TestCase(
                    test_id=tc_data['test_id'],
                    name=tc_data['name'],
                    description=tc_data['description'],
                    category=TestCategory(tc_data['category']),
                    priority=TestPriority(tc_data['priority']),
                    estimated_duration=tc_data['estimated_duration'],
                    dependencies=tc_data['dependencies'],
                    tags=tc_data['tags']
                )

                # 恢复运行时状态
                if 'status' in tc_data:
                    test_case.status = TestStatus(tc_data['status'])
                if tc_data.get('execution_count'):
                    test_case.execution_count = tc_data['execution_count']
                if tc_data.get('success_count'):
                    test_case.success_count = tc_data['success_count']
                if tc_data.get('failure_count'):
                    test_case.failure_count = tc_data['failure_count']

                self.register_test_case(test_case)
                imported_count += 1

            self.logger.info(f"Imported {imported_count} test cases from {filepath}")
            return imported_count

        except Exception as e:
            self.logger.error(f"Failed to import test cases: {e}")
            return 0

    def reset_test_case_status(self, test_id: Optional[str] = None):
        """
        重置测试用例状态

        Reset test case status

        Args:
            test_id: 特定测试用例ID，如果为None则重置所有
                    Specific test case ID, reset all if None
        """
        if test_id:
            if test_id in self.test_cases:
                old_status = self.test_cases[test_id].status
                self.test_cases[test_id].status = TestStatus.PENDING
                self._update_stats(
                    self.test_cases[test_id].category.value, old_status, TestStatus.PENDING)
                self._update_stats(
                    self.test_cases[test_id].priority.value, old_status, TestStatus.PENDING)
                self.logger.info(f"Reset status for test case: {test_id}")
        else:
            for tc in self.test_cases.values():
                old_status = tc.status
                tc.status = TestStatus.PENDING
                self._update_stats(tc.category.value, old_status, TestStatus.PENDING)
                self._update_stats(tc.priority.value, old_status, TestStatus.PENDING)

            self.logger.info(f"Reset status for all {len(self.test_cases)} test cases")


# 创建全局测试用例管理器实例
test_case_manager = TestCaseManager()

__all__ = [
    'TestPriority',
    'TestCategory',
    'TestStatus',
    'TestCase',
    'TestSuite',
    'TestExecutionResult',
    'TestCaseManager',
    'test_case_manager'
]

"""
RQA2025 测试层模块

提供完整的测试框架和质量保障体系，包括测试用例管理、执行引擎和结果分析
"""


logger = logging.getLogger(__name__)


class TestPriority(Enum):

    """测试优先级枚举"""
    CRITICAL = 1    # 关键测试 - 系统核心功能
    HIGH = 2        # 高优先级 - 重要功能
    MEDIUM = 3      # 中等优先级 - 一般功能
    LOW = 4         # 低优先级 - 辅助功能


class TestCategory(Enum):

    """测试类别枚举"""
    SMOKE = "smoke"                    # 冒烟测试
    FUNCTIONAL = "functional"          # 功能测试
    INTEGRATION = "integration"       # 集成测试
    REGRESSION = "regression"          # 回归测试
    PERFORMANCE = "performance"       # 性能测试
    SECURITY = "security"              # 安全测试
    COMPLIANCE = "compliance"          # 合规测试
    ACCEPTANCE = "acceptance"          # 验收测试


class TestStatus(Enum):

    """测试状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PASSED = "passed"          # 通过
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    BLOCKED = "blocked"        # 阻塞
    ERROR = "error"            # 错误


@dataclass
class TestCase:

    """测试用例类"""
    test_id: str
    name: str
    description: str
    category: TestCategory
    priority: TestPriority
    estimated_duration: float  # 预估执行时间（秒）
    dependencies: List[str]    # 依赖的测试用例ID
    tags: List[str]            # 标签
    test_function: Optional[Callable] = None

    # 运行时状态
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 统计信息
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_execution: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'success_rate': self.success_count / max(self.execution_count, 1) * 100
        }


@dataclass
class TestSuite:

    """测试套件类"""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase]
    priority: TestPriority = TestPriority.MEDIUM
    timeout: int = 3600  # 超时时间（秒）

    # 执行统计
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    execution_time: float = 0.0

    def __post_init__(self):

        self.total_tests = len(self.test_cases)


class TestExecutionResult:

    """测试执行结果类"""

    def __init__(self, test_case: TestCase):

        self.test_case = test_case
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.status = TestStatus.RUNNING
        self.result_details: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
        self.performance_metrics: Dict[str, float] = {}

    def complete(self, status: TestStatus, result_details: Optional[Dict[str, Any]] = None,


                 error_message: Optional[str] = None):
        """完成测试执行"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = status
        if result_details:
            self.result_details = result_details
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_case.test_id,
            'test_name': self.test_case.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }


class TestCaseManager:

    """
    测试用例管理器 - 支持分类管理和优先级调度

    Test case manager supporting categorization and priority scheduling
    """

    def __init__(self):

        self.test_cases: Dict[str, TestCase] = {}
        self.test_suites: Dict[str, TestSuite] = {}

        # 分类统计
        self.category_stats = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'failed': 0, 'pending': 0, 'running': 0
        })

        # 优先级统计
        self.priority_stats = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'failed': 0, 'pending': 0, 'running': 0
        })

        # 执行历史
        self.execution_history: List[TestExecutionResult] = []
        self.max_history_size = 10000

        # 配置
        self.parallel_execution = True
        self.max_concurrent_tests = 5
        self.default_timeout = 300

        self.logger = logging.getLogger(self.__class__.__name__)

    def register_test_case(self, test_case: TestCase):
        """
        注册测试用例

        Register test case

        Args:
            test_case: 测试用例对象
                      Test case object
        """
        if test_case.test_id in self.test_cases:
            self.logger.warning(f"Test case {test_case.test_id} already exists, updating...")

        self.test_cases[test_case.test_id] = test_case

        # 更新分类统计
        self.category_stats[test_case.category.value]['total'] += 1
        self.category_stats[test_case.category.value]['pending'] += 1

        # 更新优先级统计
        self.priority_stats[test_case.priority.value]['total'] += 1
        self.priority_stats[test_case.priority.value]['pending'] += 1

        self.logger.info(f"Registered test case: {test_case.name} ({test_case.test_id})")

    def unregister_test_case(self, test_id: str) -> bool:
        """
        注销测试用例

        Unregister test case

        Args:
            test_id: 测试用例ID
                    Test case ID

        Returns:
            bool: 是否成功注销
                 Whether successfully unregistered
        """
        if test_id not in self.test_cases:
            return False

        test_case = self.test_cases[test_id]

        # 更新统计信息
        self.category_stats[test_case.category.value]['total'] -= 1
        self.category_stats[test_case.category.value][test_case.status.value] -= 1

        self.priority_stats[test_case.priority.value]['total'] -= 1
        self.priority_stats[test_case.priority.value][test_case.status.value] -= 1

        del self.test_cases[test_id]

        self.logger.info(f"Unregistered test case: {test_id}")
        return True

    def create_test_suite(self, suite_id: str, name: str, description: str,


                          test_ids: List[str], priority: TestPriority = TestPriority.MEDIUM) -> Optional[TestSuite]:
        """
        创建测试套件

        Create test suite

        Args:
            suite_id: 套件ID
                     Suite ID
            name: 套件名称
                 Suite name
            description: 套件描述
                        Suite description
            test_ids: 测试用例ID列表
                     List of test case IDs
            priority: 套件优先级
                     Suite priority

        Returns:
            TestSuite: 创建的测试套件对象，如果创建失败返回None
                      Created test suite object, None if creation failed
        """
        test_cases = []
        for test_id in test_ids:
            if test_id in self.test_cases:
                test_cases.append(self.test_cases[test_id])
            else:
                self.logger.warning(f"Test case {test_id} not found, skipping...")
                return None

        suite = TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            test_cases=test_cases,
            priority=priority
        )

        self.test_suites[suite_id] = suite

        self.logger.info(f"Created test suite: {name} with {len(test_cases)} test cases")
        return suite

    def get_test_cases_by_category(self, category: TestCategory) -> List[TestCase]:
        """
        根据类别获取测试用例

        Get test cases by category

        Args:
            category: 测试类别
                     Test category

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tc.category == category
        ]

    def get_test_cases_by_priority(self, priority: TestPriority) -> List[TestCase]:
        """
        根据优先级获取测试用例

        Get test cases by priority

        Args:
            priority: 测试优先级
                     Test priority

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tc.priority == priority
        ]

    def get_test_cases_by_tag(self, tag: str) -> List[TestCase]:
        """
        根据标签获取测试用例

        Get test cases by tag

        Args:
            tag: 标签
                Tag

        Returns:
            List[TestCase]: 测试用例列表
                           List of test cases
        """
        return [
            tc for tc in self.test_cases.values()
            if tag in tc.tags
        ]

    def get_test_cases_for_execution(self, max_priority: Optional[TestPriority] = None,


                                     categories: Optional[List[TestCategory]] = None,
                                     tags: Optional[List[str]] = None) -> List[TestCase]:
        """
        获取待执行的测试用例（支持过滤条件）

        Get test cases for execution with filtering options

        Args:
            max_priority: 最大优先级（包含此优先级及以上的测试）
                         Maximum priority (includes this priority and higher)
            categories: 类别过滤列表
                       Category filter list
            tags: 标签过滤列表
                 Tag filter list

        Returns:
            List[TestCase]: 过滤后的测试用例列表
                           Filtered test case list
        """
        candidates = [
            tc for tc in self.test_cases.values()
            if tc.status in [TestStatus.PENDING, TestStatus.FAILED]
        ]

        # 优先级过滤
        if max_priority:
            candidates = [
                tc for tc in candidates
                if tc.priority.value <= max_priority.value
            ]

        # 类别过滤
        if categories:
            candidates = [
                tc for tc in candidates
                if tc.category in categories
            ]

        # 标签过滤
        if tags:
            candidates = [
                tc for tc in candidates
                if any(tag in tc.tags for tag in tags)
            ]

        # 按优先级排序（高优先级在前）
        candidates.sort(key=lambda tc: tc.priority.value)

        return candidates

    def execute_test_case(self, test_case: TestCase) -> TestExecutionResult:
        """
        执行单个测试用例

        Execute single test case

        Args:
            test_case: 测试用例对象
                      Test case object

        Returns:
            TestExecutionResult: 执行结果
                                Execution result
        """
        result = TestExecutionResult(test_case)

        # 更新测试用例状态
        test_case.status = TestStatus.RUNNING
        test_case.start_time = result.start_time
        test_case.execution_count += 1

        # 更新统计信息
        self._update_stats(test_case.category.value, TestStatus.PENDING, TestStatus.RUNNING)
        self._update_stats(test_case.priority.value, TestStatus.PENDING, TestStatus.RUNNING)

        try:
            # 执行测试
            if test_case.test_function:
                test_result = test_case.test_function()
            else:
                # 模拟测试执行
                time.sleep(min(test_case.estimated_duration, 5))  # 限制最大等待时间
                test_result = {"status": "passed", "details": "Mock test execution"}

            # 处理执行结果
            if test_result.get("status") == "passed":
                result.complete(TestStatus.PASSED, test_result)
                test_case.status = TestStatus.PASSED
                test_case.success_count += 1
                self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.PASSED)
                self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.PASSED)
            else:
                result.complete(TestStatus.FAILED, test_result, test_result.get("error"))
                test_case.status = TestStatus.FAILED
                test_case.failure_count += 1
                self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.FAILED)
                self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.FAILED)

        except Exception as e:
            result.complete(TestStatus.ERROR, error_message=str(e))
            test_case.status = TestStatus.ERROR
            test_case.failure_count += 1
            self._update_stats(test_case.category.value, TestStatus.RUNNING, TestStatus.ERROR)
            self._update_stats(test_case.priority.value, TestStatus.RUNNING, TestStatus.ERROR)

        # 更新测试用例信息
        test_case.end_time = result.end_time
        test_case.duration = result.duration
        test_case.last_execution = result.end_time
        test_case.result_details = result.result_details
        test_case.error_message = result.error_message

        # 添加到执行历史
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

        return result

    def _update_stats(self, key: str, old_status: TestStatus, new_status: TestStatus):
        """更新统计信息"""
        if key in self.category_stats:
            self.category_stats[key][old_status.value] -= 1
            self.category_stats[key][new_status.value] += 1

        # 同时更新优先级统计（如果key是优先级值）
        try:
            priority_val = int(key)
            if priority_val in self.priority_stats:
                self.priority_stats[priority_val][old_status.value] -= 1
                self.priority_stats[priority_val][new_status.value] += 1
        except (ValueError, KeyError):
            pass

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        获取执行摘要

        Get execution summary

        Returns:
            dict: 执行摘要统计
                 Execution summary statistics
        """
        total_tests = len(self.test_cases)
        executed_tests = sum(1 for tc in self.test_cases.values() if tc.execution_count > 0)
        passed_tests = sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.PASSED)
        failed_tests = sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.FAILED)

        recent_executions = self.execution_history[-10:] if self.execution_history else []

        return {
            'total_test_cases': total_tests,
            'executed_test_cases': executed_tests,
            'passed_test_cases': passed_tests,
            'failed_test_cases': failed_tests,
            'pending_test_cases': total_tests - executed_tests,
            'execution_rate': executed_tests / max(total_tests, 1) * 100,
            'pass_rate': passed_tests / max(executed_tests, 1) * 100,
            'category_stats': dict(self.category_stats),
            'priority_stats': dict(self.priority_stats),
            'recent_executions': [result.to_dict() for result in recent_executions]
        }

    def export_test_cases(self, filepath: str) -> bool:
        """
        导出测试用例到文件

        Export test cases to file

        Args:
            filepath: 导出文件路径
                     Export file path

        Returns:
            bool: 导出是否成功
                 Whether export was successful
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_test_cases': len(self.test_cases),
                'test_cases': [tc.to_dict() for tc in self.test_cases.values()],
                'test_suites': [asdict(suite) for suite in self.test_suites.values()]
            }

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported {len(self.test_cases)} test cases to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export test cases: {e}")
            return False

    def import_test_cases(self, filepath: str) -> int:
        """
        从文件导入测试用例

        Import test cases from file

        Args:
            filepath: 导入文件路径
                     Import file path

        Returns:
            int: 成功导入的测试用例数量
                Number of successfully imported test cases
        """
        try:
            with open(filepath, 'r', encoding='utf - 8') as f:
                import_data = json.load(f)

            imported_count = 0
            for tc_data in import_data.get('test_cases', []):
                # 重建TestCase对象
                test_case = TestCase(
                    test_id=tc_data['test_id'],
                    name=tc_data['name'],
                    description=tc_data['description'],
                    category=TestCategory(tc_data['category']),
                    priority=TestPriority(tc_data['priority']),
                    estimated_duration=tc_data['estimated_duration'],
                    dependencies=tc_data['dependencies'],
                    tags=tc_data['tags']
                )

                # 恢复运行时状态
                if 'status' in tc_data:
                    test_case.status = TestStatus(tc_data['status'])
                if tc_data.get('execution_count'):
                    test_case.execution_count = tc_data['execution_count']
                if tc_data.get('success_count'):
                    test_case.success_count = tc_data['success_count']
                if tc_data.get('failure_count'):
                    test_case.failure_count = tc_data['failure_count']

                self.register_test_case(test_case)
                imported_count += 1

            self.logger.info(f"Imported {imported_count} test cases from {filepath}")
            return imported_count

        except Exception as e:
            self.logger.error(f"Failed to import test cases: {e}")
            return 0

    def reset_test_case_status(self, test_id: Optional[str] = None):
        """
        重置测试用例状态

        Reset test case status

        Args:
            test_id: 特定测试用例ID，如果为None则重置所有
                    Specific test case ID, reset all if None
        """
        if test_id:
            if test_id in self.test_cases:
                old_status = self.test_cases[test_id].status
                self.test_cases[test_id].status = TestStatus.PENDING
                self._update_stats(
                    self.test_cases[test_id].category.value, old_status, TestStatus.PENDING)
                self._update_stats(
                    self.test_cases[test_id].priority.value, old_status, TestStatus.PENDING)
                self.logger.info(f"Reset status for test case: {test_id}")
        else:
            for tc in self.test_cases.values():
                old_status = tc.status
                tc.status = TestStatus.PENDING
                self._update_stats(tc.category.value, old_status, TestStatus.PENDING)
                self._update_stats(tc.priority.value, old_status, TestStatus.PENDING)

            self.logger.info(f"Reset status for all {len(self.test_cases)} test cases")


# 创建全局测试用例管理器实例
test_case_manager = TestCaseManager()

__all__ = [
    'TestPriority',
    'TestCategory',
    'TestStatus',
    'TestCase',
    'TestSuite',
    'TestExecutionResult',
    'TestCaseManager',
    'test_case_manager'
]
