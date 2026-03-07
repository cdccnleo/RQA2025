"""
测试统计收集器

职责: 收集和分析测试统计信息
原位置: APITestCaseGenerator.get_test_statistics()
"""

from typing import Dict, Any
from .models import TestSuite, TestStatistics


class TestStatisticsCollector:
    """测试统计收集器 - 负责收集和分析测试统计信息"""
    
    def __init__(self):
        """初始化统计收集器"""
        pass
    
    def collect_statistics(self, test_suites) -> TestStatistics:
        """
        收集测试统计信息

        Args:
            test_suites: 测试套件字典或单个TestSuite对象

        Returns:
            TestStatistics对象
        """
        stats = TestStatistics()

        # 处理不同类型的输入
        if isinstance(test_suites, dict):
            stats.total_suites = len(test_suites)
            suites_to_process = test_suites
        else:
            # 单个TestSuite对象
            stats.total_suites = 1
            suites_to_process = {'single': test_suites}
        
        # 统计场景和测试用例
        for suite in suites_to_process.values():
            stats.total_scenarios += len(suite.scenarios)
            
            for scenario in suite.scenarios:
                stats.total_test_cases += len(scenario.test_cases)
                
                # 统计测试状态
                for test_case in scenario.test_cases:
                    if test_case.status == "passed":
                        stats.passed_tests += 1
                    elif test_case.status == "failed":
                        stats.failed_tests += 1
                    elif test_case.status == "pending":
                        stats.pending_tests += 1
                    
                    # 累计执行时间
                    if test_case.execution_time:
                        stats.execution_time += test_case.execution_time
        
        return stats
    
    def calculate_coverage(self, test_suites: Dict[str, TestSuite]) -> float:
        """
        计算测试覆盖率
        
        Args:
            test_suites: 测试套件字典
        
        Returns:
            覆盖率百分比
        """
        # 简化计算：基于测试用例数量
        stats = self.collect_statistics(test_suites)
        
        if stats.total_test_cases == 0:
            return 0.0
        
        # 假设每个测试用例覆盖一定比例的代码
        # 实际应该与代码覆盖率工具集成
        executed_tests = stats.passed_tests + stats.failed_tests
        return (executed_tests / stats.total_test_cases) * 100
    
    def get_statistics_summary(self, test_suites: Dict[str, TestSuite]) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Args:
            test_suites: 测试套件字典
        
        Returns:
            统计摘要字典
        """
        stats = self.collect_statistics(test_suites)
        
        return {
            "overview": {
                "total_suites": stats.total_suites,
                "total_scenarios": stats.total_scenarios,
                "total_test_cases": stats.total_test_cases
            },
            "execution_status": {
                "passed": stats.passed_tests,
                "failed": stats.failed_tests,
                "pending": stats.pending_tests,
                "pass_rate": f"{stats.pass_rate:.2f}%"
            },
            "performance": {
                "total_execution_time": f"{stats.execution_time:.2f}s",
                "average_case_time": f"{stats.execution_time / max(1, stats.total_test_cases):.3f}s"
            },
            "coverage": {
                "estimated_coverage": f"{self.calculate_coverage(test_suites):.2f}%"
            }
        }
    
    def get_detailed_statistics(self, test_suites: Dict[str, TestSuite]) -> Dict[str, Any]:
        """
        获取详细统计信息

        Args:
            test_suites: 测试套件字典

        Returns:
            详细统计字典
        """
        stats_summary = self.get_statistics_summary(test_suites)

        # 初始化详细统计结构
        detailed = self._initialize_detailed_stats(stats_summary)

        # 执行各种统计
        self._collect_suite_statistics(test_suites, detailed)
        self._collect_cross_suite_statistics(test_suites, detailed)

        return detailed

    def _initialize_detailed_stats(self, stats_summary: Dict[str, Any]) -> Dict[str, Any]:
        """初始化详细统计数据结构"""
        return {
            "summary": stats_summary,
            "by_suite": {},
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "by_category": {},
            "by_status": {}
        }

    def _collect_suite_statistics(self, test_suites: Dict[str, TestSuite],
                                detailed: Dict[str, Any]) -> None:
        """收集按套件的统计信息"""
        for suite_name, suite in test_suites.items():
            suite_stats = self._calculate_suite_stats(suite)
            detailed["by_suite"][suite_name] = suite_stats

    def _collect_cross_suite_statistics(self, test_suites: Dict[str, TestSuite],
                                      detailed: Dict[str, Any]) -> None:
        """收集跨套件的统计信息"""
        for suite in test_suites.values():
            for scenario in suite.scenarios:
                for test_case in scenario.test_cases:
                    self._update_cross_suite_stats(test_case, detailed)

    def _calculate_suite_stats(self, suite: TestSuite) -> Dict[str, Any]:
        """计算单个套件的统计信息"""
        suite_stats = {
            "scenarios": len(suite.scenarios),
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "pending": 0
        }

        for scenario in suite.scenarios:
            suite_stats["total_cases"] += len(scenario.test_cases)

            for test_case in scenario.test_cases:
                # 按状态统计
                if test_case.status == "passed":
                    suite_stats["passed"] += 1
                elif test_case.status == "failed":
                    suite_stats["failed"] += 1
                elif test_case.status == "pending":
                    suite_stats["pending"] += 1

        return suite_stats

    def _update_cross_suite_stats(self, test_case, detailed: Dict[str, Any]) -> None:
        """更新跨套件统计信息"""
        # 按优先级统计
        priority = test_case.priority
        detailed["by_priority"][priority] = detailed["by_priority"].get(priority, 0) + 1

        # 按类别统计
        category = test_case.category
        detailed["by_category"][category] = detailed["by_category"].get(category, 0) + 1


# 避免pytest将统计收集器误判为测试类
TestStatisticsCollector.__test__ = False  # type: ignore[attr-defined]
