"""
测试目标：提升resource/monitoring/health/health_report_builder.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.health_report_builder模块
"""

from unittest.mock import Mock, patch
import pytest
from datetime import datetime

from src.infrastructure.resource.monitoring.health.health_report_builder import HealthReportBuilder
from src.infrastructure.resource.models.alert_dataclasses import PerformanceMetrics


class TestHealthReportBuilder:
    """测试HealthReportBuilder类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def builder(self, mock_logger):
        """创建构建器实例"""
        return HealthReportBuilder(mock_logger)

    def test_initialization(self, builder, mock_logger):
        """测试初始化"""
        assert builder.logger == mock_logger
        assert hasattr(builder, 'metrics_formatter')

    def test_initialization_without_logger(self):
        """测试不提供logger时的初始化"""
        builder = HealthReportBuilder()

        assert builder.logger is not None
        assert hasattr(builder.logger, 'log_info')

    @patch('src.infrastructure.resource.monitoring.health.health_report_builder.datetime')
    def test_build_health_report_success(self, mock_datetime, builder, mock_logger):
        """测试成功构建健康报告"""
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

        health_assessment = {
            "overall_score": 85.5,
            "overall_status": "healthy",
            "component_scores": {"performance": 90, "alerts": 80, "tests": 85},
            "issues": ["Minor issue"],
            "recommendations": ["Monitor closely"]
        }

        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            network_usage=30.0,
            timestamp=1000.0
        )

        alert_stats = {"active_alerts": 2, "total_alerts": 10}
        test_stats = {"success_rate": 0.95, "total_tests": 100}

        result = builder.build_health_report(health_assessment, metrics, alert_stats, test_stats)

        assert result["report_type"] == "system_health_report"
        assert result["generated_at"] == "2024-01-01T12:00:00"
        assert result["overall_health"]["score"] == 85.5
        assert result["overall_health"]["status"] == "healthy"
        assert "current_metrics" in result
        assert "statistics" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "metadata" in result

        mock_logger.log_info.assert_called_once_with("成功构建健康报告")

    def test_build_health_report_failure(self, builder, mock_logger):
        """测试构建健康报告失败"""
        # 传递无效的health_assessment来触发异常
        health_assessment = None

        result = builder.build_health_report(health_assessment)

        assert result["report_type"] == "error_report"
        assert "error" in result
        mock_logger.log_error.assert_called_once()

    def test_initialize_health_report(self, builder):
        """测试初始化健康报告"""
        with patch('src.infrastructure.resource.monitoring.health.health_report_builder.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = builder._initialize_health_report()

            assert result["report_type"] == "system_health_report"
            assert result["generated_at"] == "2024-01-01T12:00:00"
            assert result["report_period"] == "current"

    def test_populate_overall_health(self, builder):
        """测试填充整体健康状态"""
        report = {}
        health_assessment = {
            "overall_score": 85.5,
            "overall_status": "healthy"
        }

        builder._populate_overall_health(report, health_assessment)

        assert report["overall_health"]["score"] == 85.5
        assert report["overall_health"]["status"] == "healthy"

    def test_populate_detailed_scores(self, builder):
        """测试填充详细评分"""
        report = {}
        health_assessment = {
            "component_scores": {"performance": 90, "alerts": 80, "tests": 85}
        }

        builder._populate_detailed_scores(report, health_assessment)

        assert report["detailed_scores"]["performance"] == 90
        assert report["detailed_scores"]["alerts"] == 80
        assert report["detailed_scores"]["tests"] == 85

    def test_populate_current_metrics_with_data(self, builder, mock_logger):
        """测试填充当前指标（有数据）"""
        report = {}
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            network_usage=30.0,
            timestamp=1000.0
        )

        builder._populate_current_metrics(report, metrics)

        assert report["current_metrics"]["cpu_usage"] == 50.0
        assert report["current_metrics"]["memory_usage"] == 60.0
        assert report["current_metrics"]["disk_usage"] == 40.0
        assert report["current_metrics"]["network_usage"] == 30.0
        assert report["current_metrics"]["timestamp"] == 1000.0

    def test_populate_current_metrics_without_data(self, builder):
        """测试填充当前指标（无数据）"""
        report = {}

        builder._populate_current_metrics(report, None)

        assert report["current_metrics"]["status"] == "no_data_available"

    def test_populate_statistics(self, builder):
        """测试填充统计信息"""
        report = {}
        alert_stats = {"active_alerts": 2, "total_alerts": 10}
        test_stats = {"success_rate": 0.95, "total_tests": 100}

        builder._populate_statistics(report, alert_stats, test_stats)

        assert report["statistics"]["alerts"]["active_alerts"] == 2
        assert report["statistics"]["alerts"]["total_alerts"] == 10
        assert report["statistics"]["tests"]["success_rate"] == 0.95
        assert report["statistics"]["tests"]["total_tests"] == 100

    def test_populate_statistics_partial_data(self, builder):
        """测试填充部分统计信息"""
        report = {}
        alert_stats = {"active_alerts": 2}
        test_stats = None

        builder._populate_statistics(report, alert_stats, test_stats)

        assert report["statistics"]["alerts"]["active_alerts"] == 2
        assert report["statistics"]["tests"]["status"] == "no_data_available"

    def test_populate_issues_and_recommendations(self, builder):
        """测试填充问题和建议"""
        report = {}
        health_assessment = {
            "issues": ["CPU high", "Memory low"],
            "recommendations": ["Optimize CPU", "Free memory"]
        }

        builder._populate_issues_and_recommendations(report, health_assessment)

        assert report["issues"] == ["CPU high", "Memory low"]
        assert report["recommendations"] == ["Optimize CPU", "Free memory"]

    def test_populate_metadata(self, builder):
        """测试填充元数据"""
        report = {}
        health_assessment = {
            "metadata": {"version": "1.0", "environment": "test"}
        }

        builder._populate_metadata(report, health_assessment)

        assert report["metadata"]["version"] == "1.0"
        assert report["metadata"]["environment"] == "test"
        assert "generated_by" in report["metadata"]

    def test_generate_error_report(self, builder):
        """测试生成错误报告"""
        error_message = "Test error"

        result = builder._generate_error_report(error_message)

        assert result["report_type"] == "error_report"
        assert result["error"] == "Test error"
        assert "generated_at" in result
