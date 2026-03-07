import pytest
from typing import Dict
from src.infrastructure.api.test_generation.models import TestCase, TestScenario, TestSuite
from src.infrastructure.api.test_generation.statistics import TestStatisticsCollector


def _make_suite(
    suite_id: str,
    case_statuses: Dict[str, str],
    priorities: Dict[str, str] = None,
    categories: Dict[str, str] = None,
    execution_times: Dict[str, float] = None,
) -> TestSuite:
    priorities = priorities or {}
    categories = categories or {}
    execution_times = execution_times or {}

    cases = []
    for case_id, status in case_statuses.items():
        cases.append(
            TestCase(
                id=case_id,
                title=f"case-{case_id}",
                description="",
                status=status,
                priority=priorities.get(case_id, "medium"),
                category=categories.get(case_id, "functional"),
                execution_time=execution_times.get(case_id),
            )
        )

    scenario = TestScenario(
        id=f"{suite_id}-sc",
        name="scenario",
        description="",
        endpoint="/api/test",
        method="GET",
        test_cases=cases,
    )

    return TestSuite(
        id=suite_id,
        name=f"suite-{suite_id}",
        description="desc",
        scenarios=[scenario],
    )


def test_collect_statistics_with_dict_and_single_suite():
    collector = TestStatisticsCollector()
    suite_a = _make_suite("A", {"c1": "passed", "c2": "failed"})
    suite_b = _make_suite("B", {"c3": "pending"})

    stats_dict = collector.collect_statistics({"A": suite_a, "B": suite_b})
    assert stats_dict.total_suites == 2
    assert stats_dict.passed_tests == 1
    assert stats_dict.failed_tests == 1
    assert stats_dict.pending_tests == 1

    stats_single = collector.collect_statistics(suite_a)
    assert stats_single.total_suites == 1
    assert stats_single.total_test_cases == 2


def test_calculate_coverage_handles_empty_and_mixed_status():
    collector = TestStatisticsCollector()
    empty_suite = {"empty": _make_suite("empty", {})}
    assert collector.calculate_coverage(empty_suite) == 0.0

    suite = {"suite": _make_suite("suite", {"c1": "passed", "c2": "failed", "c3": "pending"})}
    assert collector.calculate_coverage(suite) == pytest.approx(66.666, rel=1e-3)


def test_get_statistics_summary_formats_strings():
    collector = TestStatisticsCollector()
    suite = {
        "suite": _make_suite(
            "suite",
            {"c1": "passed", "c2": "failed"},
            execution_times={"c1": 1.5, "c2": 0.5},
        )
    }

    summary = collector.get_statistics_summary(suite)
    assert summary["overview"]["total_suites"] == 1
    assert summary["execution_status"]["pass_rate"] == "50.00%"
    assert summary["performance"]["total_execution_time"] == "2.00s"
    assert summary["performance"]["average_case_time"] == "1.000s"


def test_get_detailed_statistics_collects_cross_suite_data():
    collector = TestStatisticsCollector()
    suite = {
        "suite": _make_suite(
            "suite",
            {"c1": "passed", "c2": "failed", "c3": "pending"},
            priorities={"c1": "high", "c2": "low", "c3": "medium"},
            categories={"c1": "functional", "c2": "security", "c3": "performance"},
        )
    }

    detailed = collector.get_detailed_statistics(suite)
    assert detailed["by_suite"]["suite"]["total_cases"] == 3
    assert detailed["by_priority"]["high"] == 1
    assert detailed["by_category"]["security"] == 1


def test_get_detailed_statistics_summary_matches_overview():
    collector = TestStatisticsCollector()
    suite = {
        "suite1": _make_suite("suite1", {"c1": "passed"}),
        "suite2": _make_suite("suite2", {"c2": "pending"}),
    }

    detailed = collector.get_detailed_statistics(suite)
    overview = detailed["summary"]["overview"]
    assert overview["total_suites"] == 2
    assert overview["total_test_cases"] == 2

