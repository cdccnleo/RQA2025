"""
测试测试套件导出器

覆盖 exporter.py 中的 TestSuiteExporter 类
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.test_generation.exporter import TestSuiteExporter
from src.infrastructure.api.test_generation.models import TestSuite, TestScenario, TestCase


class TestTestSuiteExporter:
    """TestSuiteExporter 类测试"""

    def test_initialization(self):
        """测试初始化"""
        exporter = TestSuiteExporter()

        assert exporter is not None
        assert hasattr(exporter, 'export')
        assert hasattr(exporter, 'export_to_json')
        assert hasattr(exporter, 'export_to_yaml')
        assert hasattr(exporter, 'export_to_html')

    def test_export_json(self):
        """测试导出为JSON格式"""
        exporter = TestSuiteExporter()

        # 创建测试数据
        test_case = TestCase("TC001", "测试用例", "描述")
        scenario = TestScenario("SC001", "测试场景", "描述", "/test", "GET", [test_case])
        suite = TestSuite("TS001", "测试套件", "描述", [scenario])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            exporter.export_to_json(suite, str(output_dir))

            # 验证文件已创建
            output_path = output_dir / "test_suites.json"
            assert output_path.exists()

            # 验证文件内容
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            suite_data = data["default"]
            assert suite_data["id"] == "TS001"
            assert suite_data["name"] == "测试套件"
            assert len(suite_data["scenarios"]) == 1

    def test_get_supported_formats(self):
        """测试获取支持的格式"""
        exporter = TestSuiteExporter()

        formats = exporter.get_supported_formats()

        assert isinstance(formats, list)
        assert "json" in formats

    def test_export_main_interface(self):
        """测试主要的导出接口"""
        exporter = TestSuiteExporter()

        # 创建测试数据
        test_case = TestCase("TC001", "测试用例", "描述")
        scenario = TestScenario("SC001", "测试场景", "描述", "/test", "GET", [test_case])
        suite = TestSuite("TS001", "测试套件", "描述", [scenario])
        test_suites = {"test": suite}

        with tempfile.TemporaryDirectory() as temp_dir:
            exporter.export(test_suites, "json", temp_dir)

            # 验证文件已创建
            output_path = Path(temp_dir) / "test_suites.json"
            assert output_path.exists()

    def test_export_unsupported_format(self):
        """测试不支持的导出格式"""
        exporter = TestSuiteExporter()

        test_suites = {"test": TestSuite("TS001", "测试套件", "描述")}

        with tempfile.TemporaryDirectory() as temp_dir:
            # 应该抛出异常
            with pytest.raises(ValueError, match="不支持的导出格式"):
                exporter.export(test_suites, "unsupported", temp_dir)

    def test_export_empty_suites(self):
        """测试导出空测试套件"""
        exporter = TestSuiteExporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            # 空字典应该抛出异常
            with pytest.raises(ValueError, match="必须提供 test_suites 或 test_suite 参数"):
                exporter.export({}, "json", temp_dir)

            # 应该没有创建文件
            files = list(Path(temp_dir).glob("*.json"))
            assert len(files) == 0

    def test_get_supported_formats(self):
        """测试获取支持的格式"""
        exporter = TestSuiteExporter()

        formats = exporter.get_supported_formats()

        assert isinstance(formats, list)
        assert "json" in formats
        assert "yaml" in formats
        assert "html" in formats

    def test_create_output_directory(self):
        """测试创建输出目录"""
        exporter = TestSuiteExporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "deep" / "path"
            output_path = nested_dir / "test.json"

            # 创建测试数据
            suite = TestSuite("TS001", "测试套件", "描述")

            exporter.export_to_json(suite, str(output_path))

            # 验证目录和文件都已创建
            assert nested_dir.exists()
            assert output_path.exists()

    def test_export_with_metadata(self):
        """测试导出包含元数据"""
        exporter = TestSuiteExporter()

        # 创建包含完整信息的测试套件
        test_case = TestCase(
            "TC001",
            "完整测试用例",
            "详细描述",
            priority="high",
            category="functional",
            preconditions=["条件1"],
            test_steps=[{"action": "步骤1", "expected": "结果1"}],
            expected_results=["期望结果"],
            status="passed",
            execution_time=1.5,
            tags=["tag1"]
        )

        scenario = TestScenario(
            "SC001",
            "完整测试场景",
            "场景描述",
            "/api/test",
            "POST",
            [test_case],
            ["设置步骤"],
            ["清理步骤"],
            {"key": "value"}
        )

        suite = TestSuite("TS001", "完整测试套件", "套件描述", [scenario])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            exporter.export_to_json(suite, str(output_dir))

            # 验证文件内容包含所有元数据
            output_path = output_dir / "test_suites.json"
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            suite_data = data["default"]
            assert suite_data["scenarios"][0]["endpoint"] == "/api/test"
            assert suite_data["scenarios"][0]["method"] == "POST"
            assert len(suite_data["scenarios"][0]["test_cases"][0]["test_steps"]) == 1
            assert suite_data["scenarios"][0]["test_cases"][0]["priority"] == "high"
