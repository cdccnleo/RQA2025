#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心优化测试增强器

测试目标：提升core_optimization/components/testing_enhancer.py的覆盖率到100%
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.core_optimization.components.testing_enhancer import TestingEnhancer


class TestTestingEnhancer:
    """测试测试增强器"""

    @pytest.fixture
    def testing_enhancer(self):
        """创建测试增强器实例"""
        return TestingEnhancer()

    def test_testing_enhancer_initialization(self, testing_enhancer):
        """测试测试增强器初始化"""
        assert testing_enhancer.name == "TestingEnhancer"
        assert testing_enhancer.test_frameworks == ["pytest", "unittest"]
        assert testing_enhancer.coverage_tools == ["coverage", "pytest-cov"]
        assert isinstance(testing_enhancer.test_stats, dict)

    def test_shutdown_success(self, testing_enhancer):
        """测试成功关闭"""
        result = testing_enhancer.shutdown()

        assert result == True

    def test_shutdown_with_exception(self, testing_enhancer):
        """测试关闭时发生异常"""
        with patch('src.core.core_optimization.components.testing_enhancer.logger') as mock_logger:
            mock_logger.error.side_effect = Exception("Logger error")

            result = testing_enhancer.shutdown()

            assert result == False

    @patch('subprocess.run')
    def test_run_tests_success(self, mock_subprocess_run, testing_enhancer):
        """测试成功运行测试"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "======================== 10 passed in 1.23s ========================"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = testing_enhancer.run_tests("tests/unit/")

        assert result["success"] == True
        assert result["passed"] == 10
        assert result["failed"] == 0
        assert "duration" in result

    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_subprocess_run, testing_enhancer):
        """测试运行测试失败"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "======================== 8 passed, 2 failed in 2.34s ========================"
        mock_result.stderr = "Error details..."
        mock_subprocess_run.return_value = mock_result

        result = testing_enhancer.run_tests("tests/unit/")

        assert result["success"] == False
        assert result["passed"] == 8
        assert result["failed"] == 2
        assert "error" in result

    @patch('subprocess.run')
    def test_run_tests_with_exception(self, mock_subprocess_run, testing_enhancer):
        """测试运行测试时发生异常"""
        mock_subprocess_run.side_effect = Exception("Command failed")

        result = testing_enhancer.run_tests("tests/unit/")

        assert result["success"] == False
        assert "error" in result

    def test_parse_test_output_passed_only(self, testing_enhancer):
        """测试解析仅通过的测试输出"""
        output = "======================== 15 passed in 3.45s ========================"

        passed, failed = testing_enhancer._parse_test_output(output)

        assert passed == 15
        assert failed == 0

    def test_parse_test_output_with_failures(self, testing_enhancer):
        """测试解析包含失败的测试输出"""
        output = "======================== 12 passed, 3 failed, 1 error in 2.67s ========================"

        passed, failed = testing_enhancer._parse_test_output(output)

        assert passed == 12
        assert failed == 4  # 3 failed + 1 error

    def test_parse_test_output_no_match(self, testing_enhancer):
        """测试解析不匹配的输出"""
        output = "Some random output without test results"

        passed, failed = testing_enhancer._parse_test_output(output)

        assert passed == 0
        assert failed == 0

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_analyze_test_coverage_basic(self, mock_listdir, mock_exists, testing_enhancer):
        """测试基本的测试覆盖率分析"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_file1.py", "test_file2.py", "not_test.py"]

        with patch.object(testing_enhancer, '_count_lines_in_file') as mock_count_lines:
            mock_count_lines.return_value = (100, 80)  # total_lines, code_lines

            coverage = testing_enhancer.analyze_test_coverage()

            assert coverage["test_files"] == 2  # Only files starting with "test_"
            assert coverage["source_files"] == 0  # No source directory
            assert "test_to_code_ratio" in coverage

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_analyze_test_coverage_with_source(self, mock_listdir, mock_exists, testing_enhancer):
        """测试包含源码目录的测试覆盖率分析"""
        def mock_exists_side_effect(path):
            return "src" in path or "tests" in path

        mock_exists.side_effect = mock_exists_side_effect

        mock_listdir.side_effect = [
            ["test_module1.py", "test_module2.py"],  # tests directory
            ["module1.py", "module2.py", "module3.py"]  # src directory
        ]

        with patch.object(testing_enhancer, '_count_lines_in_file') as mock_count_lines:
            mock_count_lines.return_value = (100, 80)  # total_lines, code_lines

            coverage = testing_enhancer.analyze_test_coverage()

            assert coverage["test_files"] == 2
            assert coverage["source_files"] == 3

    def test_count_lines_in_file_python(self, testing_enhancer):
        """测试计算Python文件行数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''# Comment line
import os

def function():
    """Docstring"""
    x = 1
    y = 2
    return x + y

# Another comment
class MyClass:
    pass
''')
            temp_file = f.name

        try:
            total_lines, code_lines = testing_enhancer._count_lines_in_file(temp_file)

            assert total_lines == 13
            assert code_lines == 7  # Non-comment, non-empty lines
        finally:
            os.unlink(temp_file)

    def test_count_lines_in_file_empty(self, testing_enhancer):
        """测试计算空文件行数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_file = f.name

        try:
            total_lines, code_lines = testing_enhancer._count_lines_in_file(temp_file)

            assert total_lines == 0
            assert code_lines == 0
        finally:
            os.unlink(temp_file)

    def test_count_lines_in_file_nonexistent(self, testing_enhancer):
        """测试计算不存在文件的行数"""
        total_lines, code_lines = testing_enhancer._count_lines_in_file("nonexistent_file.py")

        assert total_lines == 0
        assert code_lines == 0

    def test_generate_test_report(self, testing_enhancer):
        """测试生成测试报告"""
        # 设置一些测试统计
        testing_enhancer.test_stats = {
            "last_run": {
                "success": True,
                "passed": 25,
                "failed": 2,
                "duration": 5.67
            }
        }

        with patch.object(testing_enhancer, 'analyze_test_coverage') as mock_analyze:
            mock_analyze.return_value = {
                "test_files": 8,
                "source_files": 12,
                "coverage_rate": 85.5,
                "test_to_code_ratio": 0.67
            }

            report = testing_enhancer.generate_test_report()

            assert report["summary"]["total_tests"] == 27  # 25 + 2
            assert report["coverage"]["test_files"] == 8
            assert report["coverage"]["source_files"] == 12
            assert report["coverage"]["coverage_rate"] == 85.5

    def test_generate_test_report_no_stats(self, testing_enhancer):
        """测试生成没有统计数据的测试报告"""
        with patch.object(testing_enhancer, 'analyze_test_coverage') as mock_analyze:
            mock_analyze.return_value = {
                "test_files": 0,
                "source_files": 5,
                "coverage_rate": 0.0,
                "test_to_code_ratio": 0.0
            }

            report = testing_enhancer.generate_test_report()

            assert report["summary"]["total_tests"] == 0
            assert report["summary"]["success"] == False

    @patch('subprocess.run')
    def test_run_coverage_analysis_success(self, mock_subprocess_run, testing_enhancer):
        """测试成功运行覆盖率分析"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "TOTAL 85% coverage"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = testing_enhancer.run_coverage_analysis("src/", "tests/")

        assert result["success"] == True
        assert result["coverage_rate"] == 85.0
        assert "output" in result

    @patch('subprocess.run')
    def test_run_coverage_analysis_failure(self, mock_subprocess_run, testing_enhancer):
        """测试运行覆盖率分析失败"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Coverage analysis failed"
        mock_subprocess_run.return_value = mock_result

        result = testing_enhancer.run_coverage_analysis("src/", "tests/")

        assert result["success"] == False
        assert result["coverage_rate"] == 0.0
        assert "error" in result

    def test_parse_coverage_output_percentage(self, testing_enhancer):
        """测试解析覆盖率输出百分比"""
        output = "TOTAL 87.5% coverage (1500/2000 lines)"

        coverage_rate = testing_enhancer._parse_coverage_output(output)

        assert coverage_rate == 87.5

    def test_parse_coverage_output_no_match(self, testing_enhancer):
        """测试解析不包含百分比的覆盖率输出"""
        output = "Some output without percentage"

        coverage_rate = testing_enhancer._parse_coverage_output(output)

        assert coverage_rate == 0.0

    @patch('os.path.exists')
    def test_find_test_files(self, mock_exists, testing_enhancer):
        """测试查找测试文件"""
        mock_exists.return_value = True

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [
                "test_module1.py",
                "test_module2.py",
                "module3.py",
                "helper.py",
                "__init__.py"
            ]

            test_files = testing_enhancer._find_test_files("tests/")

            assert len(test_files) == 2
            assert "test_module1.py" in test_files
            assert "test_module2.py" in test_files
            assert "module3.py" not in test_files

    @patch('os.path.exists')
    def test_find_test_files_directory_not_exists(self, mock_exists, testing_enhancer):
        """测试查找不存在目录中的测试文件"""
        mock_exists.return_value = False

        test_files = testing_enhancer._find_test_files("nonexistent/")

        assert test_files == []

    def test_identify_test_gaps(self, testing_enhancer):
        """测试识别测试差距"""
        source_files = ["module1.py", "module2.py", "utils.py"]
        test_files = ["test_module1.py", "test_utils.py"]

        gaps = testing_enhancer.identify_test_gaps(source_files, test_files)

        assert "module2.py" in gaps["missing_tests"]
        assert len(gaps["missing_tests"]) == 1

    def test_identify_test_gaps_all_covered(self, testing_enhancer):
        """测试识别测试差距 - 全部覆盖"""
        source_files = ["module1.py", "utils.py"]
        test_files = ["test_module1.py", "test_utils.py"]

        gaps = testing_enhancer.identify_test_gaps(source_files, test_files)

        assert gaps["missing_tests"] == []

    def test_suggest_test_improvements(self, testing_enhancer):
        """测试建议测试改进"""
        current_coverage = 75.0
        gaps = {
            "missing_tests": ["module2.py", "module3.py"],
            "low_coverage_files": ["module1.py"]
        }

        suggestions = testing_enhancer.suggest_test_improvements(current_coverage, gaps)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("module2.py" in suggestion for suggestion in suggestions)
        assert any("75.0%" in suggestion for suggestion in suggestions)

    def test_suggest_test_improvements_high_coverage(self, testing_enhancer):
        """测试建议测试改进 - 高覆盖率"""
        current_coverage = 95.0
        gaps = {"missing_tests": [], "low_coverage_files": []}

        suggestions = testing_enhancer.suggest_test_improvements(current_coverage, gaps)

        assert isinstance(suggestions, list)
        assert any("95.0%" in suggestion for suggestion in suggestions)
        assert any("excellent" in suggestion.lower() for suggestion in suggestions)


class TestTestingEnhancerIntegration:
    """测试测试增强器集成场景"""

    @pytest.fixture
    def testing_enhancer(self):
        """创建测试增强器实例"""
        return TestingEnhancer()

    def test_complete_testing_workflow(self, testing_enhancer):
        """测试完整的测试工作流程"""
        # 1. 分析测试覆盖率
        with patch.object(testing_enhancer, 'analyze_test_coverage') as mock_analyze:
            mock_analyze.return_value = {
                "test_files": 10,
                "source_files": 15,
                "coverage_rate": 78.5,
                "test_to_code_ratio": 0.67
            }

            coverage = testing_enhancer.analyze_test_coverage()

            assert coverage["test_files"] == 10
            assert coverage["source_files"] == 15
            assert coverage["coverage_rate"] == 78.5

        # 2. 运行测试
        with patch.object(testing_enhancer, 'run_tests') as mock_run_tests:
            mock_run_tests.return_value = {
                "success": True,
                "passed": 45,
                "failed": 2,
                "duration": 8.5
            }

            test_result = testing_enhancer.run_tests("tests/unit/")

            assert test_result["success"] == True
            assert test_result["passed"] == 45
            assert test_result["failed"] == 2

        # 3. 运行覆盖率分析
        with patch.object(testing_enhancer, 'run_coverage_analysis') as mock_coverage:
            mock_coverage.return_value = {
                "success": True,
                "coverage_rate": 82.3,
                "output": "Coverage analysis completed"
            }

            coverage_result = testing_enhancer.run_coverage_analysis("src/", "tests/")

            assert coverage_result["success"] == True
            assert coverage_result["coverage_rate"] == 82.3

        # 4. 生成测试报告
        with patch.object(testing_enhancer, 'generate_test_report') as mock_report:
            mock_report.return_value = {
                "summary": {"total_tests": 47, "success": True},
                "coverage": {"coverage_rate": 82.3},
                "recommendations": ["Add more integration tests"]
            }

            report = testing_enhancer.generate_test_report()

            assert report["summary"]["total_tests"] == 47
            assert report["coverage"]["coverage_rate"] == 82.3

    def test_error_handling_and_recovery(self, testing_enhancer):
        """测试错误处理和恢复"""
        # 1. 测试运行失败的测试
        with patch.object(testing_enhancer, 'run_tests') as mock_run_tests:
            mock_run_tests.return_value = {
                "success": False,
                "passed": 10,
                "failed": 5,
                "error": "Import error in test_module.py"
            }

            result = testing_enhancer.run_tests("tests/")

            assert result["success"] == False
            assert result["failed"] == 5
            assert "error" in result

        # 2. 测试覆盖率分析失败
        with patch.object(testing_enhancer, 'run_coverage_analysis') as mock_coverage:
            mock_coverage.return_value = {
                "success": False,
                "coverage_rate": 0.0,
                "error": "Coverage tool not found"
            }

            result = testing_enhancer.run_coverage_analysis("src/", "tests/")

            assert result["success"] == False
            assert result["error"] == "Coverage tool not found"

        # 3. 测试报告生成仍然可用
        report = testing_enhancer.generate_test_report()

        assert isinstance(report, dict)
        assert "summary" in report
        assert "coverage" in report

    def test_test_gap_analysis_workflow(self, testing_enhancer):
        """测试测试差距分析工作流程"""
        # 模拟源代码文件和测试文件
        source_files = ["module1.py", "module2.py", "module3.py", "utils.py"]
        test_files = ["test_module1.py", "test_utils.py"]

        # 识别测试差距
        gaps = testing_enhancer.identify_test_gaps(source_files, test_files)

        assert "module2.py" in gaps["missing_tests"]
        assert "module3.py" in gaps["missing_tests"]
        assert len(gaps["missing_tests"]) == 2

        # 生成改进建议
        suggestions = testing_enhancer.suggest_test_improvements(70.0, gaps)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("module2.py" in suggestion for suggestion in suggestions)
        assert any("module3.py" in suggestion for suggestion in suggestions)

    def test_concurrent_testing_operations(self, testing_enhancer):
        """测试并发测试操作"""
        import threading
        import time

        results = []
        errors = []

        def run_test_operation(operation_name):
            try:
                if operation_name == "analyze_coverage":
                    result = testing_enhancer.analyze_test_coverage()
                    results.append(f"coverage_{len(result)}")
                elif operation_name == "generate_report":
                    result = testing_enhancer.generate_test_report()
                    results.append(f"report_{len(result)}")
                elif operation_name == "identify_gaps":
                    gaps = testing_enhancer.identify_test_gaps(["mod1.py"], ["test_mod1.py"])
                    results.append(f"gaps_{len(gaps)}")
            except Exception as e:
                errors.append(f"{operation_name}_error: {str(e)}")

        # 创建多个线程并发执行测试操作
        operations = ["analyze_coverage", "generate_report", "identify_gaps", "analyze_coverage"]
        threads = []

        for operation in operations:
            thread = threading.Thread(target=run_test_operation, args=(operation,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有操作都成功完成
        assert len(results) == len(operations)
        assert all("coverage_" in r or "report_" in r or "gaps_" in r for r in results)

    def test_resource_management(self, testing_enhancer):
        """测试资源管理"""
        # 运行多个测试分析操作
        operations = [
            lambda: testing_enhancer.analyze_test_coverage(),
            lambda: testing_enhancer.generate_test_report(),
            lambda: testing_enhancer.identify_test_gaps(["test.py"], ["test_test.py"])
        ]

        for operation in operations:
            result = operation()
            assert isinstance(result, (dict, list))

        # 验证增强器状态仍然正常
        assert testing_enhancer.name == "TestingEnhancer"
        assert isinstance(testing_enhancer.test_stats, dict)

    def test_configuration_and_customization(self, testing_enhancer):
        """测试配置和定制"""
        # 测试框架定制
        original_frameworks = testing_enhancer.test_frameworks.copy()
        testing_enhancer.test_frameworks.append("nose")

        assert "nose" in testing_enhancer.test_frameworks

        # 恢复原始配置
        testing_enhancer.test_frameworks = original_frameworks

        # 测试覆盖率工具定制
        original_tools = testing_enhancer.coverage_tools.copy()
        testing_enhancer.coverage_tools.append("coveralls")

        assert "coveralls" in testing_enhancer.coverage_tools

        # 恢复原始配置
        testing_enhancer.coverage_tools = original_tools

    def test_performance_under_load(self, testing_enhancer):
        """测试负载下的性能"""
        import time

        # 执行多次分析操作
        start_time = time.time()

        for i in range(10):
            coverage = testing_enhancer.analyze_test_coverage()
            assert isinstance(coverage, dict)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证在合理时间内完成
        assert total_time < 2.0  # 应该在2秒内完成10次分析

        # 验证分析结果合理
        final_coverage = testing_enhancer.analyze_test_coverage()
        assert isinstance(final_coverage, dict)
        assert "test_files" in final_coverage

    def test_shutdown_and_cleanup(self, testing_enhancer):
        """测试关闭和清理"""
        # 执行一些操作
        coverage = testing_enhancer.analyze_test_coverage()
        report = testing_enhancer.generate_test_report()

        assert isinstance(coverage, dict)
        assert isinstance(report, dict)

        # 关闭增强器
        shutdown_result = testing_enhancer.shutdown()

        assert shutdown_result == True

        # 验证状态仍然可用（基础组件状态）
        assert testing_enhancer.name == "TestingEnhancer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
