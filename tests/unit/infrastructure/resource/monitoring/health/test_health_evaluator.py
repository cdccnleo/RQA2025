"""
测试目标：提升resource/monitoring/health/health_evaluator.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.health_evaluator模块
"""

from unittest.mock import Mock
import pytest
from datetime import datetime

from src.infrastructure.resource.monitoring.health.health_evaluator import HealthEvaluator
from src.infrastructure.resource.models.alert_dataclasses import AlertPerformanceMetrics as PerformanceMetrics
from src.infrastructure.resource.monitoring.health.health_status import HealthStatus


class TestHealthEvaluator:
    """测试HealthEvaluator类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def evaluator(self, mock_logger):
        """创建评估器实例"""
        return HealthEvaluator(logger=mock_logger)

    @pytest.fixture
    def custom_thresholds(self):
        """自定义阈值"""
        return {
            "cpu_critical": 95.0,
            "cpu_warning": 85.0,
            "memory_critical": 95.0,
            "memory_warning": 90.0,
            "disk_critical": 98.0,
            "disk_warning": 95.0,
            "alerts_critical": 15,
            "alerts_warning": 8,
            "test_success_rate_critical": 0.3,
            "test_success_rate_warning": 0.7
        }

    def test_initialization_default_thresholds(self, evaluator, mock_logger):
        """测试使用默认阈值的初始化"""
        assert evaluator.logger == mock_logger
        assert evaluator.thresholds["cpu_critical"] == 90.0
        assert evaluator.thresholds["memory_warning"] == 85.0
        assert evaluator.thresholds["alerts_critical"] == 10

    def test_initialization_custom_thresholds(self, custom_thresholds, mock_logger):
        """测试使用自定义阈值的初始化"""
        evaluator = HealthEvaluator(thresholds=custom_thresholds, logger=mock_logger)

        assert evaluator.logger == mock_logger
        assert evaluator.thresholds["cpu_critical"] == 95.0
        assert evaluator.thresholds["memory_warning"] == 90.0
        assert evaluator.thresholds["alerts_critical"] == 15

    def test_initialization_without_logger(self, custom_thresholds):
        """测试不提供logger时的初始化"""
        evaluator = HealthEvaluator(thresholds=custom_thresholds)

        assert evaluator.logger is not None
        assert hasattr(evaluator.logger, 'warning')

    def test_evaluate_overall_health_healthy_system(self, evaluator):
        """测试评估健康的系统"""
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        alert_stats = {"active_alerts": 2, "total_alerts": 5}
        test_stats = {"success_rate": 0.95, "total_tests": 100}

        result = evaluator.evaluate_overall_health(metrics, alert_stats, test_stats)

        assert result["overall_score"] > 80
        assert result["overall_status"] == HealthStatus.HEALTHY
        assert "component_scores" in result
        assert "recommendations" in result

    def test_evaluate_overall_health_warning_system(self, evaluator):
        """测试评估警告状态的系统"""
        metrics = PerformanceMetrics(
            cpu_usage=85.0,  # 警告级别
            memory_usage=60.0,
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        alert_stats = {"active_alerts": 6, "total_alerts": 8}  # 警告级别
        test_stats = {"success_rate": 0.75, "total_tests": 100}  # 警告级别

        result = evaluator.evaluate_overall_health(metrics, alert_stats, test_stats)

        assert 60 <= result["overall_score"] <= 80
        assert result["overall_status"] == HealthStatus.WARNING

    def test_evaluate_overall_health_critical_system(self, evaluator):
        """测试评估临界状态的系统"""
        metrics = PerformanceMetrics(
            cpu_usage=95.0,  # 临界级别
            memory_usage=60.0,
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        alert_stats = {"active_alerts": 12, "total_alerts": 15}  # 临界级别
        test_stats = {"success_rate": 0.3, "total_tests": 100}  # 临界级别

        result = evaluator.evaluate_overall_health(metrics, alert_stats, test_stats)

        assert result["overall_score"] < 60
        assert result["overall_status"] == HealthStatus.CRITICAL

    def test_evaluate_overall_health_no_metrics(self, evaluator):
        """测试没有指标时的评估"""
        result = evaluator.evaluate_overall_health(None)

        assert result["overall_score"] > 0  # Returns calculated score even with no metrics
        assert result["overall_status"] == "warning"  # Returns warning status
        assert "无法获取性能指标" in str(result.get("issues", []))

    def test_evaluate_overall_health_partial_data(self, evaluator):
        """测试部分数据时的评估"""
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        # 只有metrics，没有alert_stats和test_stats
        result = evaluator.evaluate_overall_health(metrics)

        assert result["overall_score"] > 0
        assert result["overall_status"] != HealthStatus.UNKNOWN

    def test_evaluate_performance_health_normal(self, evaluator):
        """测试正常性能健康评估"""
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        result = evaluator.evaluate_performance_health(metrics)

        assert result["score"] > 80
        assert result["status"] == HealthStatus.HEALTHY
        assert len(result["issues"]) == 0

    def test_evaluate_performance_health_with_warnings(self, evaluator):
        """测试有警告的性能健康评估"""
        metrics = PerformanceMetrics(
            cpu_usage=85.0,  # 警告
            memory_usage=88.0,  # 警告
            disk_usage=40.0,
            network_latency=30.0,
            timestamp=datetime.now()
        )

        result = evaluator.evaluate_performance_health(metrics)

        assert 60 <= result["score"] <= 80
        assert result["status"] == HealthStatus.WARNING
        assert len(result["issues"]) > 0

    def test_evaluate_performance_health_critical(self, evaluator):
        """测试临界性能健康评估"""
        metrics = PerformanceMetrics(
            cpu_usage=95.0,  # 临界
            memory_usage=95.0,  # 临界
            disk_usage=97.0,  # 临界
            network_latency=30.0,
            timestamp=datetime.now()
        )

        result = evaluator.evaluate_performance_health(metrics)

        assert result["score"] < 60
        assert result["status"] == HealthStatus.CRITICAL
        assert len(result["issues"]) > 0

    def test_evaluate_alert_health_normal(self, evaluator):
        """测试正常告警健康评估"""
        alert_stats = {"active_alerts": 2, "total_alerts": 5}

        result = evaluator.evaluate_alert_health(alert_stats)

        assert result["score"] > 80
        assert result["status"] == HealthStatus.HEALTHY

    def test_evaluate_alert_health_warning(self, evaluator):
        """测试警告告警健康评估"""
        alert_stats = {"active_alerts": 6, "total_alerts": 8}

        result = evaluator.evaluate_alert_health(alert_stats)

        assert 60 <= result["score"] <= 80
        assert result["status"] == HealthStatus.WARNING

    def test_evaluate_alert_health_critical(self, evaluator):
        """测试临界告警健康评估"""
        alert_stats = {"active_alerts": 12, "total_alerts": 15}

        result = evaluator.evaluate_alert_health(alert_stats)

        assert result["score"] < 60
        assert result["status"] == HealthStatus.CRITICAL

    def test_evaluate_alert_health_no_data(self, evaluator):
        """测试无告警数据时的评估"""
        result = evaluator.evaluate_alert_health(None)

        assert result["score"] == 100  # 默认健康
        assert result["status"] == HealthStatus.HEALTHY

    def test_evaluate_test_health_normal(self, evaluator):
        """测试正常测试健康评估"""
        test_stats = {"success_rate": 0.95, "total_tests": 100}

        result = evaluator.evaluate_test_health(test_stats)

        assert result["score"] > 80
        assert result["status"] == HealthStatus.HEALTHY

    def test_evaluate_test_health_warning(self, evaluator):
        """测试警告测试健康评估"""
        test_stats = {"success_rate": 0.75, "total_tests": 100}

        result = evaluator.evaluate_test_health(test_stats)

        assert 60 <= result["score"] <= 80
        assert result["status"] == HealthStatus.WARNING

    def test_evaluate_test_health_critical(self, evaluator):
        """测试临界测试健康评估"""
        test_stats = {"success_rate": 0.3, "total_tests": 100}

        result = evaluator.evaluate_test_health(test_stats)

        assert result["score"] < 60
        assert result["status"] == HealthStatus.CRITICAL

    def test_evaluate_test_health_no_data(self, evaluator):
        """测试无测试数据时的评估"""
        result = evaluator.evaluate_test_health(None)

        assert result["score"] == 100  # 默认健康
        assert result["status"] == HealthStatus.HEALTHY

    def test_calculate_weighted_score(self, evaluator):
        """测试加权分数计算"""
        scores = [80, 90, 70]
        weights = [0.4, 0.4, 0.2]

        result = evaluator._calculate_weighted_score(scores, weights)

        expected = 80 * 0.4 + 90 * 0.4 + 70 * 0.2
        assert result == expected

    def test_calculate_weighted_score_empty(self, evaluator):
        """测试空分数列表的加权分数计算"""
        result = evaluator._calculate_weighted_score([], [])

        assert result == 0

    def test_generate_recommendations_healthy(self, evaluator):
        """测试健康系统的建议生成"""
        result = evaluator._generate_recommendations(HealthStatus.HEALTHY, [])

        assert "系统运行正常" in result
        assert len(result) >= 1

    def test_generate_recommendations_warning(self, evaluator):
        """测试警告系统的建议生成"""
        issues = ["CPU使用率较高", "内存使用率较高"]

        result = evaluator._generate_recommendations(HealthStatus.WARNING, issues)

        assert "优化系统性能" in result
        assert len(result) >= 2

    def test_generate_recommendations_critical(self, evaluator):
        """测试临界系统的建议生成"""
        issues = ["CPU使用率严重超标", "内存不足"]

        result = evaluator._generate_recommendations(HealthStatus.CRITICAL, issues)

        assert "立即采取措施" in result
        assert len(result) >= 3
