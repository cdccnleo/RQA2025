"""
测试测试统计收集器

覆盖 statistics.py 中的 TestStatisticsCollector 类
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.test_generation.statistics import TestStatisticsCollector


class TestTestStatisticsCollector:
    """TestStatisticsCollector 类测试"""

    def test_initialization(self):
        """测试初始化"""
        collector = TestStatisticsCollector()

        assert collector is not None
        assert hasattr(collector, 'collect_statistics')
        assert hasattr(collector, 'get_statistics')
        assert hasattr(collector, 'validate_test_suite')
        assert hasattr(collector, 'get_coverage_report')
        assert hasattr(collector, 'get_performance_metrics')

    def test_collect_statistics_single_suite(self):
        """测试收集单个测试套件的统计信息"""
        collector = TestStatisticsCollector()

        # 创建真实的测试套件
        from src.infrastructure.api.test_generation.models import TestSuite, TestScenario, TestCase
        test_case1 = TestCase("TC001", "测试用例1", "描述", status="passed", execution_time=10.0)
        test_case2 = TestCase("TC002", "测试用例2", "描述", status="failed", execution_time=5.0)
        scenario = TestScenario("SC001", "测试场景", "描述", "/test", "GET", [test_case1, test_case2])
        suite = TestSuite("TS001", "测试套件", "描述", [scenario])

        result = collector.collect_statistics(suite)

        assert result is not None
        assert hasattr(result, 'total_suites')
        assert hasattr(result, 'total_test_cases')
        assert hasattr(result, 'passed_tests')
        assert hasattr(result, 'failed_tests')
        assert hasattr(result, 'pending_tests')
        assert hasattr(result, 'coverage_percentage')
        assert hasattr(result, 'execution_time')

    def test_collect_statistics_multiple_suites(self):
        """测试收集多个测试套件的统计信息"""
        collector = TestStatisticsCollector()

        # 创建真实的测试套件
        from src.infrastructure.api.test_generation.models import TestSuite, TestScenario, TestCase

        # 第一个套件
        test_case1 = TestCase("TC001", "测试用例1", "描述", status="passed", execution_time=10.0)
        test_case2 = TestCase("TC002", "测试用例2", "描述", status="failed", execution_time=5.0)
        scenario1 = TestScenario("SC001", "测试场景1", "描述", "/test1", "GET", [test_case1, test_case2])
        suite1 = TestSuite("TS001", "测试套件1", "描述", [scenario1])

        # 第二个套件
        test_case3 = TestCase("TC003", "测试用例3", "描述", status="passed", execution_time=8.0)
        test_case4 = TestCase("TC004", "测试用例4", "描述", status="pending", execution_time=2.0)
        scenario2 = TestScenario("SC002", "测试场景2", "描述", "/test2", "POST", [test_case3, test_case4])
        suite2 = TestSuite("TS002", "测试套件2", "描述", [scenario2])

        test_suites = {
            "suite1": suite1,
            "suite2": suite2
        }

        result = collector.collect_statistics(test_suites)

        assert result is not None
        assert result.total_suites == 2
        assert result.total_test_cases == 4
        assert result.passed_tests == 2
        assert result.failed_tests == 1
        assert result.pending_tests == 1

    def test_get_statistics(self):
        """测试获取统计信息"""
        collector = TestStatisticsCollector()

        # 首先收集一些统计信息
        mock_suite = Mock()
        mock_suite.name = "test_suite"
        mock_suite.test_count = 10
        mock_suite.passed_count = 8
        mock_suite.failed_count = 1
        mock_suite.skipped_count = 1
        mock_suite.coverage = 80.0
        mock_suite.execution_time = 30.5

        collector.collect_statistics(mock_suite)

        result = collector.get_statistics()

        assert isinstance(result, dict)
        assert "total_suites" in result
        assert "total_tests" in result
        assert "passed_tests" in result
        assert "failed_tests" in result
        assert "skipped_tests" in result
        assert "success_rate" in result
        assert "average_coverage" in result

    def test_validate_test_suite_valid(self):
        """测试验证有效的测试套件"""
        collector = TestStatisticsCollector()

        # 创建有效的测试套件
        mock_suite = Mock()
        mock_suite.name = "valid_test_suite"
        mock_suite.test_count = 10
        mock_suite.passed_count = 8
        mock_suite.failed_count = 1
        mock_suite.skipped_count = 1
        mock_suite.coverage = 85.0
        mock_suite.execution_time = 45.2
        mock_suite.test_cases = ["test1", "test2", "test3"]

        result = collector.validate_test_suite(mock_suite)

        assert isinstance(result, dict)
        assert result["valid"] == True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0

    def test_validate_test_suite_invalid(self):
        """测试验证无效的测试套件"""
        collector = TestStatisticsCollector()

        # 创建无效的测试套件（测试数量不匹配）
        mock_suite = Mock()
        mock_suite.name = "invalid_test_suite"
        mock_suite.test_count = 10
        mock_suite.passed_count = 8
        mock_suite.failed_count = 3  # 8 + 3 + 1 = 12 > 10
        mock_suite.skipped_count = 1
        mock_suite.coverage = 85.0
        mock_suite.execution_time = 45.2
        mock_suite.test_cases = ["test1", "test2"]

        result = collector.validate_test_suite(mock_suite)

        assert isinstance(result, dict)
        assert result["valid"] == False
        assert len(result["errors"]) > 0

    def test_get_coverage_report(self):
        """测试获取覆盖率报告"""
        collector = TestStatisticsCollector()

        # 设置一些测试数据
        mock_suite1 = Mock()
        mock_suite1.name = "suite1"
        mock_suite1.coverage = 85.0
        mock_suite1.test_count = 10

        mock_suite2 = Mock()
        mock_suite2.name = "suite2"
        mock_suite2.coverage = 90.0
        mock_suite2.test_count = 15

        test_suites = {"suite1": mock_suite1, "suite2": mock_suite2}
        collector.collect_statistics(test_suites)

        result = collector.get_coverage_report()

        assert isinstance(result, dict)
        assert "average_coverage" in result
        assert "min_coverage" in result
        assert "max_coverage" in result
        assert "coverage_distribution" in result

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        collector = TestStatisticsCollector()

        # 设置一些测试数据
        mock_suite1 = Mock()
        mock_suite1.name = "suite1"
        mock_suite1.execution_time = 45.2
        mock_suite1.test_count = 10

        mock_suite2 = Mock()
        mock_suite2.name = "suite2"
        mock_suite2.execution_time = 62.8
        mock_suite2.test_count = 15

        test_suites = {"suite1": mock_suite1, "suite2": mock_suite2}
        collector.collect_statistics(test_suites)

        result = collector.get_performance_metrics()

        assert isinstance(result, dict)
        assert "average_execution_time" in result
        assert "total_execution_time" in result
        assert "tests_per_second" in result
        assert "performance_trends" in result

    def test_collect_statistics_empty_input(self):
        """测试收集空输入的统计信息"""
        collector = TestStatisticsCollector()

        result = collector.collect_statistics({})

        assert result is not None
        assert result.total_suites == 0
        assert result.total_tests == 0

    def test_get_statistics_no_data(self):
        """测试获取没有数据的统计信息"""
        collector = TestStatisticsCollector()

        result = collector.get_statistics()

        assert isinstance(result, dict)
        assert result["total_suites"] == 0
        assert result["total_tests"] == 0

    def test_validate_test_suite_none_input(self):
        """测试验证None输入的测试套件"""
        collector = TestStatisticsCollector()

        result = collector.validate_test_suite(None)

        assert isinstance(result, dict)
        assert result["valid"] == False
        assert len(result["errors"]) > 0

    def test_get_coverage_report_no_data(self):
        """测试获取没有数据的覆盖率报告"""
        collector = TestStatisticsCollector()

        result = collector.get_coverage_report()

        assert isinstance(result, dict)
        assert result["average_coverage"] == 0.0
        assert result["min_coverage"] == 0.0
        assert result["max_coverage"] == 0.0

    def test_get_performance_metrics_no_data(self):
        """测试获取没有数据的性能指标"""
        collector = TestStatisticsCollector()

        result = collector.get_performance_metrics()

        assert isinstance(result, dict)
        assert result["average_execution_time"] == 0.0
        assert result["total_execution_time"] == 0.0
        assert result["tests_per_second"] == 0.0


class TestTestStatisticsCollectorIntegration:
    """TestStatisticsCollector 集成测试"""

    def test_complete_statistics_workflow(self):
        """测试完整的统计工作流"""
        collector = TestStatisticsCollector()

        # 1. 创建多个测试套件
        mock_suite1 = Mock()
        mock_suite1.name = "user_api_tests"
        mock_suite1.test_count = 15
        mock_suite1.passed_count = 12
        mock_suite1.failed_count = 2
        mock_suite1.skipped_count = 1
        mock_suite1.coverage = 85.5
        mock_suite1.execution_time = 45.2

        mock_suite2 = Mock()
        mock_suite2.name = "order_api_tests"
        mock_suite2.test_count = 20
        mock_suite2.passed_count = 18
        mock_suite2.failed_count = 1
        mock_suite2.skipped_count = 1
        mock_suite2.coverage = 90.0
        mock_suite2.execution_time = 62.8

        mock_suite3 = Mock()
        mock_suite3.name = "payment_api_tests"
        mock_suite3.test_count = 12
        mock_suite3.passed_count = 10
        mock_suite3.failed_count = 1
        mock_suite3.skipped_count = 1
        mock_suite3.coverage = 87.5
        mock_suite3.execution_time = 38.9

        test_suites = {
            "user_api": mock_suite1,
            "order_api": mock_suite2,
            "payment_api": mock_suite3
        }

        # 2. 收集统计信息
        stats = collector.collect_statistics(test_suites)

        # 3. 验证汇总统计
        assert stats.total_suites == 3
        assert stats.total_tests == 47  # 15 + 20 + 12
        assert stats.passed_tests == 40  # 12 + 18 + 10
        assert stats.failed_tests == 4   # 2 + 1 + 1
        assert stats.skipped_tests == 3  # 1 + 1 + 1

        # 4. 获取详细统计信息
        detailed_stats = collector.get_statistics(test_suites)

        # 5. 验证成功率计算
        expected_success_rate = (40 / 47) * 100  # 85.11%
        assert abs(detailed_stats["success_rate"] - expected_success_rate) < 0.01

        # 6. 验证平均覆盖率
        expected_avg_coverage = (85.5 + 90.0 + 87.5) / 3  # 87.67%
        assert abs(detailed_stats["average_coverage"] - expected_avg_coverage) < 0.01

        # 7. 获取覆盖率报告
        coverage_report = collector.get_coverage_report(test_suites)

        # 8. 验证覆盖率范围
        assert coverage_report["min_coverage"] == 85.5
        assert coverage_report["max_coverage"] == 90.0
        assert coverage_report["average_coverage"] == expected_avg_coverage

        # 9. 获取性能指标
        performance_metrics = collector.get_performance_metrics(test_suites)

        # 10. 验证性能计算
        expected_total_time = 45.2 + 62.8 + 38.9  # 146.9
        expected_avg_time = expected_total_time / 3  # 48.97
        expected_tests_per_second = 47 / expected_total_time

        assert abs(performance_metrics["total_execution_time"] - expected_total_time) < 0.01
        assert abs(performance_metrics["average_execution_time"] - expected_avg_time) < 0.01
        assert abs(performance_metrics["tests_per_second"] - expected_tests_per_second) < 0.01

    def test_validation_workflow(self):
        """测试验证工作流"""
        collector = TestStatisticsCollector()

        # 1. 创建有效的测试套件
        valid_suite = Mock()
        valid_suite.name = "valid_suite"
        valid_suite.test_count = 10
        valid_suite.passed_count = 7
        valid_suite.failed_count = 2
        valid_suite.skipped_count = 1
        valid_suite.coverage = 85.0
        valid_suite.execution_time = 30.0
        valid_suite.test_cases = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"]

        # 2. 创建无效的测试套件（覆盖率超出范围）
        invalid_suite = Mock()
        invalid_suite.name = "invalid_suite"
        invalid_suite.test_count = 5
        invalid_suite.passed_count = 4
        invalid_suite.failed_count = 1
        invalid_suite.skipped_count = 0
        invalid_suite.coverage = 150.0  # 无效的覆盖率
        invalid_suite.execution_time = 15.0
        invalid_suite.test_cases = ["test1", "test2", "test3", "test4", "test5"]

        # 3. 验证有效套件
        valid_result = collector.validate_test_suite(valid_suite)
        assert valid_result["valid"] == True
        assert len(valid_result["errors"]) == 0

        # 4. 验证无效套件
        invalid_result = collector.validate_test_suite(invalid_suite)
        assert invalid_result["valid"] == False
        assert len(invalid_result["errors"]) > 0
        assert any("覆盖率" in error for error in invalid_result["errors"])

    def test_statistics_accumulation(self):
        """测试统计信息累积"""
        collector = TestStatisticsCollector()

        # 第一次收集
        suite1 = Mock()
        suite1.name = "suite1"
        suite1.test_count = 10
        suite1.passed_count = 8
        suite1.failed_count = 1
        suite1.skipped_count = 1
        suite1.coverage = 80.0
        suite1.execution_time = 25.0

        suite2 = Mock()
        suite2.name = "suite2"
        suite2.test_count = 15
        suite2.passed_count = 12
        suite2.failed_count = 2
        suite2.skipped_count = 1
        suite2.coverage = 85.0
        suite2.execution_time = 35.0

        # 同时收集两个suite的统计
        test_suites = {"suite1": suite1, "suite2": suite2}
        stats2 = collector.get_statistics(test_suites)

        # 验证累积效果
        assert stats2["total_suites"] == 2
        assert stats2["total_tests"] == 25  # 10 + 15
        assert stats2["passed_tests"] == 20  # 8 + 12
        assert stats2["failed_tests"] == 3   # 1 + 2
        assert stats2["skipped_tests"] == 2  # 1 + 1

    def test_coverage_distribution_analysis(self):
        """测试覆盖率分布分析"""
        collector = TestStatisticsCollector()

        # 创建具有不同覆盖率的测试套件
        suites = {}
        coverage_values = [75.0, 80.0, 85.0, 90.0, 95.0]

        for i, coverage in enumerate(coverage_values):
            suite = Mock()
            suite.name = f"suite_{i}"
            suite.test_count = 10
            suite.passed_count = 8
            suite.failed_count = 1
            suite.skipped_count = 1
            suite.coverage = coverage
            suite.execution_time = 20.0
            suites[f"suite_{i}"] = suite

        collector.collect_statistics(suites)

        coverage_report = collector.get_coverage_report(suites)

        # 验证覆盖率分布
        assert coverage_report["min_coverage"] == 75.0
        assert coverage_report["max_coverage"] == 95.0
        expected_avg = sum(coverage_values) / len(coverage_values)
        assert abs(coverage_report["average_coverage"] - expected_avg) < 0.01

        # 验证分布数据存在
        assert "coverage_distribution" in coverage_report
        distribution = coverage_report["coverage_distribution"]
        assert isinstance(distribution, dict)

    def test_performance_trend_analysis(self):
        """测试性能趋势分析"""
        collector = TestStatisticsCollector()

        # 创建具有不同执行时间的测试套件
        suites = {}
        execution_times = [10.0, 15.0, 20.0, 25.0, 30.0]

        for i, exec_time in enumerate(execution_times):
            suite = Mock()
            suite.name = f"suite_{i}"
            suite.test_count = 10
            suite.passed_count = 8
            suite.failed_count = 1
            suite.skipped_count = 1
            suite.coverage = 85.0
            suite.execution_time = exec_time
            suites[f"suite_{i}"] = suite

        collector.collect_statistics(suites)

        performance_metrics = collector.get_performance_metrics(suites)

        # 验证性能指标
        expected_total_time = sum(execution_times)
        expected_avg_time = expected_total_time / len(execution_times)
        expected_tests_per_second = 50 / expected_total_time  # 5 suites * 10 tests

        assert abs(performance_metrics["total_execution_time"] - expected_total_time) < 0.01
        assert abs(performance_metrics["average_execution_time"] - expected_avg_time) < 0.01
        assert abs(performance_metrics["tests_per_second"] - expected_tests_per_second) < 0.01

        # 验证趋势数据
        assert "performance_trends" in performance_metrics
        trends = performance_metrics["performance_trends"]
        assert isinstance(trends, list)

    def test_error_handling_and_edge_cases(self):
        """测试错误处理和边界情况"""
        collector = TestStatisticsCollector()

        # 测试空测试套件
        empty_suite = Mock()
        empty_suite.name = "empty_suite"
        empty_suite.test_count = 0
        empty_suite.passed_count = 0
        empty_suite.failed_count = 0
        empty_suite.skipped_count = 0
        empty_suite.coverage = 0.0
        empty_suite.execution_time = 0.0
        empty_suite.test_cases = []

        result = collector.validate_test_suite(empty_suite)
        # 空套件应该是有效的（没有违反规则）
        assert isinstance(result, dict)

        # 测试极端覆盖率值
        extreme_suite = Mock()
        extreme_suite.name = "extreme_suite"
        extreme_suite.test_count = 10
        extreme_suite.passed_count = 10
        extreme_suite.failed_count = 0
        extreme_suite.skipped_count = 0
        extreme_suite.coverage = 100.0
        extreme_suite.execution_time = 10.0
        extreme_suite.test_cases = [f"test{i}" for i in range(10)]

        result = collector.validate_test_suite(extreme_suite)
        assert result["valid"] == True

        # 测试负值（理论上不应该发生，但要处理）
        negative_suite = Mock()
        negative_suite.name = "negative_suite"
        negative_suite.test_count = 10
        negative_suite.passed_count = 8
        negative_suite.failed_count = -1  # 负值
        negative_suite.skipped_count = 1
        negative_suite.coverage = 85.0
        negative_suite.execution_time = 20.0
        negative_suite.test_cases = [f"test{i}" for i in range(10)]

        result = collector.validate_test_suite(negative_suite)
        assert result["valid"] == False

    def test_statistics_persistence(self):
        """测试统计信息持久性"""
        collector1 = TestStatisticsCollector()
        collector2 = TestStatisticsCollector()

        # 在第一个收集器中收集数据
        suite = Mock()
        suite.name = "test_suite"
        suite.test_count = 10
        suite.passed_count = 8
        suite.failed_count = 1
        suite.skipped_count = 1
        suite.coverage = 85.0
        suite.execution_time = 25.0

        collector1.collect_statistics(suite)

        # 验证两个收集器独立
        stats1 = collector1.get_statistics({"test_suite": suite})
        stats2 = collector2.get_statistics()

        assert stats1["total_suites"] == 1
        assert stats1["total_tests"] == 10

        assert stats2["total_suites"] == 0
        assert stats2["total_tests"] == 0
