"""
单元测试 - 统计模块重构版本

测试statistics.py中的重构功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock
from src.infrastructure.api.test_generation.statistics import TestStatistics
from src.infrastructure.api.test_generation.models import TestSuite, TestScenario, TestCase


class TestTestStatisticsRefactored:
    """测试重构后的TestStatistics类"""

    @pytest.fixture
    def stats(self):
        """创建统计实例"""
        return TestStatistics()

    @pytest.fixture
    def sample_test_suites(self):
        """创建示例测试套件"""
        # 创建测试用例
        test_case1 = TestCase(
            id="tc1",
            title="测试用例1",
            description="测试描述",
            priority="high",
            category="functional",
            status="passed"
        )
        test_case2 = TestCase(
            id="tc2",
            title="测试用例2",
            description="性能测试",
            priority="medium",
            category="performance",
            status="failed"
        )
        test_case3 = TestCase(
            id="tc3",
            title="测试用例3",
            description="功能测试",
            priority="high",
            category="functional",
            status="passed"
        )

        # 创建测试场景
        scenario1 = TestScenario(
            id="scenario1",
            name="场景1",
            description="测试场景1",
            endpoint="/api/test1",
            method="GET",
            test_cases=[test_case1, test_case2]
        )
        scenario2 = TestScenario(
            id="scenario2",
            name="场景2",
            description="测试场景2",
            endpoint="/api/test2",
            method="POST",
            test_cases=[test_case3]
        )

        # 创建测试套件
        suite1 = TestSuite(
            id="suite1",
            name="测试套件1",
            description="第一个测试套件",
            scenarios=[scenario1]
        )
        suite2 = TestSuite(
            id="suite2",
            name="测试套件2",
            description="第二个测试套件",
            scenarios=[scenario2]
        )

        return {"suite1": suite1, "suite2": suite2}

    def test_get_detailed_statistics_structure(self, stats, sample_test_suites):
        """测试详细统计的结构"""
        result = stats.get_detailed_statistics(sample_test_suites)

        # 验证基本结构
        assert "summary" in result
        assert "by_suite" in result
        assert "by_priority" in result
        assert "by_category" in result
        assert "by_status" in result

        # 验证按套件统计
        assert "suite1" in result["by_suite"]
        assert "suite2" in result["by_suite"]

        # 验证优先级统计
        assert "high" in result["by_priority"]
        assert "medium" in result["by_priority"]
        assert result["by_priority"]["high"] == 2  # tc1和tc3都是high
        assert result["by_priority"]["medium"] == 1  # tc2是medium

        # 验证类别统计
        assert "functional" in result["by_category"]
        assert "performance" in result["by_category"]
        assert result["by_category"]["functional"] == 2  # tc1和tc3都是functional
        assert result["by_category"]["performance"] == 1  # tc2是performance

    def test_initialize_detailed_stats(self, stats):
        """测试详细统计数据结构的初始化"""
        summary = {"total_suites": 2, "total_scenarios": 5, "total_cases": 10}

        result = stats._initialize_detailed_stats(summary)

        assert result["summary"] == summary
        assert result["by_suite"] == {}
        assert result["by_priority"] == {"high": 0, "medium": 0, "low": 0}
        assert result["by_category"] == {}
        assert result["by_status"] == {}

    def test_collect_suite_statistics(self, stats, sample_test_suites):
        """测试按套件统计收集"""
        detailed = {"by_suite": {}}

        stats._collect_suite_statistics(sample_test_suites, detailed)

        # 验证suite1的统计
        suite1_stats = detailed["by_suite"]["suite1"]
        assert suite1_stats["scenarios"] == 1
        assert suite1_stats["total_cases"] == 2
        assert suite1_stats["passed"] == 1
        assert suite1_stats["failed"] == 1
        assert suite1_stats["pending"] == 0

        # 验证suite2的统计
        suite2_stats = detailed["by_suite"]["suite2"]
        assert suite2_stats["scenarios"] == 1
        assert suite2_stats["total_cases"] == 1
        assert suite2_stats["passed"] == 1
        assert suite2_stats["failed"] == 0
        assert suite2_stats["pending"] == 0

    def test_collect_cross_suite_statistics(self, stats, sample_test_suites):
        """测试跨套件统计收集"""
        detailed = {
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "by_category": {}
        }

        stats._collect_cross_suite_statistics(sample_test_suites, detailed)

        # 验证优先级统计
        assert detailed["by_priority"]["high"] == 2  # tc1和tc3
        assert detailed["by_priority"]["medium"] == 1  # tc2

        # 验证类别统计
        assert detailed["by_category"]["functional"] == 2  # tc1和tc3
        assert detailed["by_category"]["performance"] == 1  # tc2

    def test_calculate_suite_stats(self, stats, sample_test_suites):
        """测试单个套件的统计计算"""
        suite = sample_test_suites["suite1"]

        result = stats._calculate_suite_stats(suite)

        assert result["scenarios"] == 1
        assert result["total_cases"] == 2
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["pending"] == 0

    def test_update_cross_suite_stats(self, stats):
        """测试跨套件统计更新"""
        detailed = {
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "by_category": {}
        }

        # 模拟测试用例
        test_case = Mock()
        test_case.priority = "high"
        test_case.category = "functional"

        stats._update_cross_suite_stats(test_case, detailed)

        assert detailed["by_priority"]["high"] == 1
        assert detailed["by_category"]["functional"] == 1

    def test_process_search_results(self, stats):
        """测试搜索结果处理"""
        # 创建模拟的搜索结果
        results = [
            Mock(relevance_score=0.9),
            Mock(relevance_score=0.7),
            Mock(relevance_score=0.8)
        ]

        processed = stats._process_search_results(results, 2)

        # 验证排序和限制
        assert len(processed) == 2
        assert processed[0].relevance_score == 0.9
        assert processed[1].relevance_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
