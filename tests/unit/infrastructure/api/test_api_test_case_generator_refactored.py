#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API测试用例生成器重构版本

测试目标：提升api_test_case_generator_refactored.py的覆盖率到100%
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.infrastructure.api.api_test_case_generator_refactored import APITestCaseGenerator
from src.infrastructure.api.test_generation.builders.base_builder import TestSuite


class TestAPITestCaseGenerator:
    """测试APITestCaseGenerator类"""

    @pytest.fixture
    def generator(self, tmp_path):
        """创建APITestCaseGenerator实例"""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        return APITestCaseGenerator(template_dir=template_dir)

    @pytest.fixture
    def default_generator(self):
        """创建默认APITestCaseGenerator实例"""
        return APITestCaseGenerator()

    def test_initialization_with_template_dir(self, tmp_path):
        """测试带模板目录的初始化"""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        generator = APITestCaseGenerator(template_dir=template_dir)

        assert hasattr(generator, '_template_manager')
        assert hasattr(generator, '_exporter')
        assert hasattr(generator, '_statistics')
        assert hasattr(generator, '_builders')
        assert isinstance(generator.test_suites, dict)
        assert isinstance(generator.templates, dict)

    def test_initialization_default(self):
        """测试默认初始化"""
        generator = APITestCaseGenerator()

        assert hasattr(generator, '_template_manager')
        assert hasattr(generator, '_exporter')
        assert hasattr(generator, '_statistics')
        assert hasattr(generator, '_builders')
        assert isinstance(generator.test_suites, dict)
        assert isinstance(generator.templates, dict)

    def test_supported_service_types(self, generator):
        """测试支持的服务类型"""
        # 检查构建器字典包含预期的服务类型
        expected_services = ['data_service', 'feature_service', 'trading_service', 'monitoring_service']
        assert all(service in generator._builders for service in expected_services)

    def test_create_data_service_test_suite(self, generator):
        """测试创建数据服务测试套件"""
        suite = generator.create_data_service_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id in generator.test_suites
        assert generator.test_suites[suite.id] == suite

    def test_create_feature_service_test_suite(self, generator):
        """测试创建特征服务测试套件"""
        suite = generator.create_feature_service_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id in generator.test_suites

    def test_create_trading_service_test_suite(self, generator):
        """测试创建交易服务测试套件"""
        suite = generator.create_trading_service_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id in generator.test_suites

    def test_create_monitoring_service_test_suite(self, generator):
        """测试创建监控服务测试套件"""
        suite = generator.create_monitoring_service_test_suite()

        assert isinstance(suite, TestSuite)
        assert suite.id in generator.test_suites

    def test_generate_complete_test_suite(self, generator):
        """测试生成完整的测试套件"""
        test_suites = generator.generate_complete_test_suite()

        assert isinstance(test_suites, dict)
        assert len(test_suites) == 4  # 四个服务类型
        assert all(isinstance(suite, TestSuite) for suite in test_suites.values())

        # 验证test_suites属性也被更新
        assert generator.test_suites == test_suites

    def test_export_test_cases_default(self, generator, tmp_path):
        """测试默认导出测试用例"""
        # 先创建一些测试套件
        generator.generate_complete_test_suite()

        output_dir = tmp_path / "output"
        result = generator.export_test_cases(output_dir=str(output_dir))

        assert isinstance(result, str)
        # 应该返回输出路径或成功消息

    def test_export_test_cases_custom_format(self, generator, tmp_path):
        """测试自定义格式导出测试用例"""
        generator.generate_complete_test_suite()

        output_dir = tmp_path / "output"
        result = generator.export_test_cases(format_type="json", output_dir=str(output_dir))

        assert isinstance(result, str)

    def test_get_test_statistics(self, generator):
        """测试获取测试统计信息"""
        # 先创建测试套件
        generator.generate_complete_test_suite()

        stats = generator.get_test_statistics()

        assert isinstance(stats, dict)
        assert "total_suites" in stats
        assert "total_scenarios" in stats
        assert stats["total_suites"] == 4

    def test_list_available_templates(self, generator):
        """测试列出可用模板"""
        # 这个方法可能不存在，检查templates属性
        templates = generator.templates

        assert isinstance(templates, dict)
        # 模板字典应该包含各种测试模板

    def test_thread_safety_concurrent_generation(self, generator):
        """测试并发生成测试套件的线程安全性"""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(5):
                    if worker_id % 4 == 0:
                        suite = generator.create_data_service_test_suite()
                    elif worker_id % 4 == 1:
                        suite = generator.create_feature_service_test_suite()
                    elif worker_id % 4 == 2:
                        suite = generator.create_trading_service_test_suite()
                    else:
                        suite = generator.create_monitoring_service_test_suite()

                    results.append(f"worker_{worker_id}_suite_{suite.id}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) >= 16, f"Expected at least 16 results, got {len(results)}"


class TestAPITestCaseGeneratorIntegration:
    """测试APITestCaseGenerator集成场景"""

    @pytest.fixture
    def generator(self):
        """创建生成器fixture"""
        return APITestCaseGenerator()

    def test_complete_workflow(self, generator):
        """测试完整工作流程"""
        # 生成所有测试套件
        test_suites = generator.generate_complete_test_suite()

        assert len(test_suites) == 4
        assert all(isinstance(suite, TestSuite) for suite in test_suites.values())

        # 获取统计信息
        stats = generator.get_test_statistics()
        assert isinstance(stats, dict)
        assert stats["total_suites"] == 4

        # 验证所有套件都在test_suites中
        assert len(generator.test_suites) == 4

    def test_service_types_coverage(self, generator):
        """测试服务类型覆盖"""
        # 生成所有服务类型的测试套件
        test_suites = generator.generate_complete_test_suite()

        expected_services = ['data_service', 'feature_service', 'trading_service', 'monitoring_service']

        for service_type in expected_services:
            # 验证构建器存在
            assert service_type in generator._builders
            # 验证测试套件已生成
            assert any(service_type in suite_id for suite_id in test_suites.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])