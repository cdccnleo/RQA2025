"""
测试数据模型

从原api_test_case_generator.py迁移的数据类
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class TestCase:
    """测试用例"""
    id: str
    title: str
    description: str
    priority: str = "medium"  # high, medium, low
    category: str = "functional"  # functional, integration, performance, security
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    actual_results: Optional[List[str]] = None
    status: str = "pending"  # pending, passed, failed, blocked
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


@dataclass
class TestStatistics:
    """测试统计信息"""
    total_suites: int = 0
    total_scenarios: int = 0
    total_test_cases: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pending_tests: int = 0
    coverage_percentage: float = 0.0
    execution_time: float = 0.0

    @property
    def total_tests(self) -> int:
        """total_tests的别名，用于向后兼容"""
        return self.total_test_cases

    @property
    def skipped_tests(self) -> int:
        """skipped_tests的别名，用于向后兼容"""
        return self.pending_tests

    @property
    def pass_rate(self) -> float:
        """计算通过率"""
        if self.total_test_cases == 0:
            return 0.0
        return (self.passed_tests / self.total_test_cases) * 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧接口）"""
        return {
            'basic': {
                'total_suites': self.total_suites,
                'total_scenarios': self.total_scenarios,
                'total_test_cases': self.total_test_cases,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'pending_tests': self.pending_tests,
                'pass_rate': self.pass_rate,
                'coverage_percentage': self.coverage_percentage,
                'execution_time': self.execution_time
            },
            'by_priority': {},  # 暂时为空
            'by_category': {},  # 暂时为空
            'coverage': {},     # 暂时为空
            'quality': {}       # 暂时为空
        }

    # ===== 新增统计方法，兼容重构版tests =====
    def get_detailed_statistics(self, test_suites: Dict[str, TestSuite]) -> Dict[str, Any]:
        """获取包含详细维度的统计信息"""
        summary = {
            "total_suites": len(test_suites),
            "total_scenarios": sum(len(suite.scenarios) for suite in test_suites.values()),
            "total_cases": sum(len(scenario.test_cases) for suite in test_suites.values() for scenario in suite.scenarios),
        }
        detailed = self._initialize_detailed_stats(summary)
        self._collect_suite_statistics(test_suites, detailed)
        self._collect_cross_suite_statistics(test_suites, detailed)
        return detailed

    def _initialize_detailed_stats(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """初始化详细统计结构"""
        return {
            "summary": summary,
            "by_suite": {},
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "by_category": {},
            "by_status": {},
        }

    def _collect_suite_statistics(self, test_suites: Dict[str, TestSuite], detailed: Dict[str, Any]) -> None:
        """按测试套件收集统计信息"""
        for suite_id, suite in test_suites.items():
            suite_stats = self._calculate_suite_stats(suite)
            detailed["by_suite"][suite_id] = suite_stats

            # 更新状态汇总
            status_bucket = detailed.setdefault("by_status", {})
            for key in ("passed", "failed", "pending", "blocked"):
                status_bucket.setdefault(key, 0)
            status_bucket["passed"] += suite_stats["passed"]
            status_bucket["failed"] += suite_stats["failed"]
            status_bucket["pending"] += suite_stats["pending"]
            status_bucket["blocked"] += suite_stats.get("blocked", 0)

    def _collect_cross_suite_statistics(self, test_suites: Dict[str, TestSuite], detailed: Dict[str, Any]) -> None:
        """跨套件统计优先级与类别"""
        for suite in test_suites.values():
            for scenario in suite.scenarios:
                for test_case in scenario.test_cases:
                    self._update_cross_suite_stats(test_case, detailed)

    def _calculate_suite_stats(self, suite: TestSuite) -> Dict[str, int]:
        """计算单个测试套件的统计信息"""
        stats = {
            "scenarios": len(suite.scenarios),
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "pending": 0,
            "blocked": 0,
        }

        for scenario in suite.scenarios:
            stats["total_cases"] += len(scenario.test_cases)
            for test_case in scenario.test_cases:
                status = (test_case.status or "pending").lower()
                if status == "passed":
                    stats["passed"] += 1
                elif status == "failed":
                    stats["failed"] += 1
                elif status == "blocked":
                    stats["blocked"] += 1
                else:
                    stats["pending"] += 1

        return stats

    def _update_cross_suite_stats(self, test_case: TestCase, detailed: Dict[str, Any]) -> None:
        """更新跨套件统计信息"""
        prio_bucket = detailed.setdefault("by_priority", {"high": 0, "medium": 0, "low": 0})
        priority = (test_case.priority or "medium").lower()
        prio_bucket.setdefault(priority, 0)
        prio_bucket[priority] += 1

        category = (test_case.category or "functional").lower()
        category_bucket = detailed.setdefault("by_category", {})
        category_bucket.setdefault(category, 0)
        category_bucket[category] += 1

    def _process_search_results(self, results: List[Any], limit: int) -> List[Any]:
        """处理搜索结果，按相关度排序并限制数量"""
        sorted_results = sorted(results, key=lambda r: getattr(r, "relevance_score", 0), reverse=True)
        return sorted_results[:limit]


# 避免被pytest当作测试类收集
TestCase.__test__ = False  # type: ignore[attr-defined]
TestScenario.__test__ = False  # type: ignore[attr-defined]
TestSuite.__test__ = False  # type: ignore[attr-defined]
TestStatistics.__test__ = False  # type: ignore[attr-defined]
