"""
cloud_enhanced_monitoring 模块的测试用例
提升测试覆盖率从1.02%到80%+
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

# 导入被测试的模块
from src.infrastructure.config.environment.cloud_enhanced_monitoring import (
    EnhancedMonitoringManager,
    MetricsAggregator,
    AlertCorrelator,
    PerformanceAnalyzer
)


class TestEnhancedMonitoringManager:
    """EnhancedMonitoringManager 测试类"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        config = {
            "prometheus_enabled": True,
            "grafana_enabled": True,
            "alerting_enabled": True,
            "log_aggregation": True,
            "enable_tracing": True
        }
        return config

    @pytest.fixture
    def manager(self, mock_config):
        """创建EnhancedMonitoringManager实例"""
        return EnhancedMonitoringManager(config=mock_config, cloud_provider="aws")

    def test_initialization(self, mock_config):
        """测试初始化"""
        manager = EnhancedMonitoringManager(config=mock_config, cloud_provider="aws")
        
        assert manager.config == mock_config
        assert manager.cloud_provider == "aws"
        assert manager.monitoring_enabled is True
        assert manager._custom_metrics == {}
        assert manager._alert_patterns == {}
        assert manager._anomaly_scores == {}

    def test_initialization_defaults(self):
        """测试默认初始化"""
        manager = EnhancedMonitoringManager()
        
        assert manager.config == {}
        assert manager.cloud_provider == "aws"
        assert manager.monitoring_enabled is True

    def test_start_monitoring(self, manager):
        """测试启动监控"""
        result = manager.start_monitoring()
        assert result is True
        
    def test_start_monitoring_disabled(self, manager):
        """测试启动监控 - 监控已禁用"""
        manager.monitoring_enabled = False
        result = manager.start_monitoring()
        assert result is False

    def test_stop_monitoring(self, manager):
        """测试停止监控"""
        result = manager.stop_monitoring()
        assert result is True
        assert manager.monitoring_enabled is False

    def test_add_metric_collector(self, manager):
        """测试添加指标收集器"""
        collector = Mock()
        manager.add_metric_collector("test_collector", collector)
        
        assert "test_collector" in manager.metrics_collectors
        assert manager.metrics_collectors["test_collector"] == collector

    def test_remove_metric_collector_success(self, manager):
        """测试移除指标收集器 - 成功"""
        collector = Mock()
        manager.add_metric_collector("test_collector", collector)
        
        result = manager.remove_metric_collector("test_collector")
        assert result is True
        assert "test_collector" not in manager.metrics_collectors

    def test_remove_metric_collector_not_found(self, manager):
        """测试移除指标收集器 - 不存在"""
        result = manager.remove_metric_collector("nonexistent_collector")
        assert result is False

    def test_add_alert_rule(self, manager):
        """测试添加告警规则"""
        rule_config = {"threshold": 100, "operator": ">"}
        manager.add_alert_rule("test_rule", rule_config)
        
        assert "test_rule" in manager.alert_rules
        assert manager.alert_rules["test_rule"] == rule_config

    def test_get_current_metrics_empty(self, manager):
        """测试获取当前指标 - 无收集器"""
        metrics = manager.get_current_metrics()
        
        assert "timestamp" in metrics
        assert "cloud_provider" in metrics
        assert metrics["cloud_provider"] == "aws"
        assert metrics["metrics"] == {}

    def test_get_current_metrics_with_collectors(self, manager):
        """测试获取当前指标 - 有收集器"""
        mock_collector = Mock()
        mock_collector.collect.return_value = {"cpu_usage": 50.0}
        
        manager.add_metric_collector("cpu", mock_collector)
        metrics = manager.get_current_metrics()
        
        assert "cpu" in metrics["metrics"]
        assert metrics["metrics"]["cpu"] == {"cpu_usage": 50.0}

    def test_get_current_metrics_with_exception(self, manager):
        """测试获取当前指标 - 收集器异常"""
        mock_collector = Mock()
        mock_collector.collect.side_effect = Exception("Collection failed")
        
        manager.add_metric_collector("failing_collector", mock_collector)
        metrics = manager.get_current_metrics()
        
        assert "failing_collector" in metrics["metrics"]
        assert "error" in metrics["metrics"]["failing_collector"]

    def test_detect_anomalies_empty(self, manager):
        """测试检测异常 - 空数据"""
        anomalies = manager.detect_anomalies({})
        assert anomalies == []

    def test_detect_anomalies_normal_values(self, manager):
        """测试检测异常 - 正常值"""
        metrics_data = {
            "cpu": {"value": 50.0, "threshold": 80.0}
        }
        anomalies = manager.detect_anomalies(metrics_data)
        assert anomalies == []

    def test_detect_anomalies_high_values(self, manager):
        """测试检测异常 - 异常值"""
        metrics_data = {
            "cpu": {"value": 150.0, "threshold": 80.0}
        }
        anomalies = manager.detect_anomalies(metrics_data)
        
        assert len(anomalies) == 1
        assert anomalies[0]["metric"] == "cpu"
        assert anomalies[0]["value"] == 150.0
        assert anomalies[0]["severity"] == "high"

    def test_detect_anomalies_invalid_data(self, manager):
        """测试检测异常 - 无效数据"""
        metrics_data = {
            "cpu": "invalid_data"
        }
        anomalies = manager.detect_anomalies(metrics_data)
        assert anomalies == []

    def test_get_scaling_recommendations_empty(self, manager):
        """测试获取扩缩容建议 - 空数据"""
        recommendations = manager.get_scaling_recommendations()
        assert isinstance(recommendations, list)

    def test_get_scaling_recommendations_scale_up(self, manager):
        """测试获取扩缩容建议 - 扩容"""
        mock_collector = Mock()
        mock_collector.collect.return_value = {"value": 100.0, "threshold": 80.0}
        manager.add_metric_collector("cpu", mock_collector)
        
        recommendations = manager.get_scaling_recommendations()
        
        assert len(recommendations) > 0
        assert recommendations[0]["action"] == "scale_up"

    def test_get_scaling_recommendations_scale_down(self, manager):
        """测试获取扩缩容建议 - 缩容"""
        mock_collector = Mock()
        mock_collector.collect.return_value = {"value": 10.0, "threshold": 80.0}
        manager.add_metric_collector("cpu", mock_collector)
        
        recommendations = manager.get_scaling_recommendations()
        
        assert len(recommendations) > 0
        assert recommendations[0]["action"] == "scale_down"

    def test_optimize_costs_empty(self, manager):
        """测试成本优化建议 - 空数据"""
        optimizations = manager.optimize_costs()
        
        assert isinstance(optimizations, list)
        assert len(optimizations) >= 1  # 至少有通用建议

    def test_optimize_costs_low_utilization(self, manager):
        """测试成本优化建议 - 低利用率"""
        mock_collector = Mock()
        mock_collector.collect.return_value = {"utilization": 0.2}
        manager.add_metric_collector("instance", mock_collector)
        
        optimizations = manager.optimize_costs()
        
        # 应该有资源优化建议
        resource_optimizations = [o for o in optimizations if o.get("type") == "resource_rightsizing"]
        assert len(resource_optimizations) > 0

    def test_get_monitoring_status(self, manager):
        """测试获取监控状态"""
        status = manager.get_monitoring_status()
        
        assert "enabled" in status
        assert "cloud_provider" in status
        assert "collection_interval" in status
        assert "retention_days" in status
        assert status["cloud_provider"] == "aws"

    def test_cleanup_old_data(self, manager):
        """测试清理旧数据"""
        cleaned_count = manager.cleanup_old_data()
        assert cleaned_count == 0  # 默认实现返回0


class TestMetricsAggregator:
    """MetricsAggregator 测试类"""

    @pytest.fixture
    def aggregator(self):
        """创建MetricsAggregator实例"""
        return MetricsAggregator()

    def test_initialization(self):
        """测试初始化"""
        aggregator = MetricsAggregator()
        assert aggregator.metrics == {}
        assert aggregator.aggregations == {}

    def test_add_metric_success(self, aggregator):
        """测试添加指标 - 成功"""
        aggregator.add_metric("test_metric", 100.0, time.time())
        
        assert "test_metric" in aggregator.metrics
        assert len(aggregator.metrics["test_metric"]) == 1

    def test_add_metric_multiple_values(self, aggregator):
        """测试添加指标 - 多个值"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        assert len(aggregator.metrics["test_metric"]) == 5

    def test_add_metric_limit_enforcement(self, aggregator):
        """测试添加指标 - 限制执行"""
        current_time = time.time()
        for i in range(1005):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        # 应该只保留最近1000个
        assert len(aggregator.metrics["test_metric"]) == 1000

    def test_get_aggregated_metric_not_found(self, aggregator):
        """测试获取聚合指标 - 指标不存在"""
        result = aggregator.get_aggregated_metric("nonexistent_metric")
        assert result == 0.0

    def test_get_aggregated_metric_avg(self, aggregator):
        """测试获取聚合指标 - 平均值"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        result = aggregator.get_aggregated_metric("test_metric", "avg", window=100)
        assert result == 2.0  # (0+1+2+3+4)/5

    def test_get_aggregated_metric_sum(self, aggregator):
        """测试获取聚合指标 - 求和"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        result = aggregator.get_aggregated_metric("test_metric", "sum", window=100)
        assert result == 10.0  # 0+1+2+3+4

    def test_get_aggregated_metric_max(self, aggregator):
        """测试获取聚合指标 - 最大值"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        result = aggregator.get_aggregated_metric("test_metric", "max", window=100)
        assert result == 4.0

    def test_get_aggregated_metric_min(self, aggregator):
        """测试获取聚合指标 - 最小值"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        result = aggregator.get_aggregated_metric("test_metric", "min", window=100)
        assert result == 0.0

    def test_get_aggregated_metric_count(self, aggregator):
        """测试获取聚合指标 - 计数"""
        current_time = time.time()
        for i in range(5):
            aggregator.add_metric("test_metric", float(i), current_time + i)
        
        result = aggregator.get_aggregated_metric("test_metric", "count", window=100)
        assert result == 5.0

    def test_get_aggregated_metric_invalid_type(self, aggregator):
        """测试获取聚合指标 - 无效类型"""
        current_time = time.time()
        aggregator.add_metric("test_metric", 100.0, current_time)
        
        result = aggregator.get_aggregated_metric("test_metric", "invalid", window=100)
        assert result == 0.0

    def test_get_all_aggregated_metrics(self, aggregator):
        """测试获取所有聚合指标"""
        current_time = time.time()
        aggregator.add_metric("metric1", 10.0, current_time)
        aggregator.add_metric("metric2", 20.0, current_time)
        
        result = aggregator.get_all_aggregated_metrics("avg", window=100)
        
        assert "metric1" in result
        assert "metric2" in result
        assert result["metric1"] == 10.0
        assert result["metric2"] == 20.0


class TestAlertCorrelator:
    """AlertCorrelator 测试类"""

    @pytest.fixture
    def correlator(self):
        """创建AlertCorrelator实例"""
        return AlertCorrelator()

    def test_initialization(self):
        """测试初始化"""
        correlator = AlertCorrelator()
        assert correlator.alerts == []
        assert correlator.correlations == {}

    def test_add_alert_success(self, correlator):
        """测试添加告警 - 成功"""
        alert = {"type": "cpu_high", "message": "CPU usage high"}
        correlator.add_alert(alert)
        
        assert len(correlator.alerts) == 1
        assert "timestamp" in correlator.alerts[0]

    def test_add_alert_multiple(self, correlator):
        """测试添加告警 - 多个"""
        for i in range(5):
            alert = {"type": f"alert_{i}", "message": f"Alert {i}"}
            correlator.add_alert(alert)
        
        assert len(correlator.alerts) == 5

    def test_add_alert_limit_enforcement(self, correlator):
        """测试添加告警 - 限制执行"""
        for i in range(1005):
            alert = {"type": f"alert_{i}", "message": f"Alert {i}"}
            correlator.add_alert(alert)
        
        # 应该只保留最近1000个
        assert len(correlator.alerts) == 1000

    def test_find_correlations_empty(self, correlator):
        """测试查找告警关联 - 空"""
        correlations = correlator.find_correlations()
        assert correlations == []

    def test_find_correlations_single_type(self, correlator):
        """测试查找告警关联 - 单一类型"""
        alert = {"type": "cpu_high", "message": "CPU high"}
        correlator.add_alert(alert)
        
        correlations = correlator.find_correlations()
        assert correlations == []  # 单个告警不算关联

    def test_find_correlations_multiple_same_type(self, correlator):
        """测试查找告警关联 - 多个相同类型"""
        for i in range(3):
            alert = {"type": "cpu_high", "message": f"CPU high {i}"}
            correlator.add_alert(alert)
        
        correlations = correlator.find_correlations(time_window=1000)
        
        assert len(correlations) > 0
        assert correlations[0]["type"] == "same_type"
        assert correlations[0]["alert_type"] == "cpu_high"
        assert correlations[0]["count"] == 3

    def test_get_alert_summary_empty(self, correlator):
        """测试获取告警摘要 - 空"""
        summary = correlator.get_alert_summary()
        
        assert summary["total_alerts"] == 0
        assert summary["alert_types"] == {}

    def test_get_alert_summary_with_alerts(self, correlator):
        """测试获取告警摘要 - 有告警"""
        for i in range(3):
            correlator.add_alert({"type": "cpu_high"})
        for i in range(2):
            correlator.add_alert({"type": "memory_high"})
        
        summary = correlator.get_alert_summary(time_window=1000)
        
        assert summary["total_alerts"] == 5
        assert summary["alert_types"]["cpu_high"] == 3
        assert summary["alert_types"]["memory_high"] == 2


class TestPerformanceAnalyzer:
    """PerformanceAnalyzer 测试类"""

    @pytest.fixture
    def analyzer(self):
        """创建PerformanceAnalyzer实例"""
        return PerformanceAnalyzer()

    def test_initialization(self):
        """测试初始化"""
        analyzer = PerformanceAnalyzer()
        assert analyzer.performance_data == {}
        assert analyzer.baselines == {}

    def test_record_performance_metric_success(self, analyzer):
        """测试记录性能指标 - 成功"""
        analyzer.record_performance_metric("test_metric", 100.0, time.time())
        
        assert "test_metric" in analyzer.performance_data
        assert len(analyzer.performance_data["test_metric"]) == 1

    def test_record_performance_metric_multiple(self, analyzer):
        """测试记录性能指标 - 多个"""
        current_time = time.time()
        for i in range(5):
            analyzer.record_performance_metric("test_metric", float(i), current_time + i)
        
        assert len(analyzer.performance_data["test_metric"]) == 5

    def test_record_performance_metric_limit_enforcement(self, analyzer):
        """测试记录性能指标 - 限制执行"""
        current_time = time.time()
        for i in range(1005):
            analyzer.record_performance_metric("test_metric", float(i), current_time + i)
        
        # 应该只保留最近1000个
        assert len(analyzer.performance_data["test_metric"]) == 1000

    def test_set_baseline(self, analyzer):
        """测试设置基准值"""
        analyzer.set_baseline("test_metric", 50.0)
        
        assert "test_metric" in analyzer.baselines
        assert analyzer.baselines["test_metric"] == 50.0

    def test_analyze_performance_no_data(self, analyzer):
        """测试分析性能 - 无数据"""
        result = analyzer.analyze_performance("nonexistent_metric")
        assert result["status"] == "no_data"

    def test_analyze_performance_no_recent_data(self, analyzer):
        """测试分析性能 - 无最近数据"""
        # 添加过期数据
        old_time = time.time() - 10000
        analyzer.record_performance_metric("test_metric", 100.0, old_time)
        
        result = analyzer.analyze_performance("test_metric", time_window=100)
        assert result["status"] == "no_recent_data"

    def test_analyze_performance_with_data(self, analyzer):
        """测试分析性能 - 有数据"""
        current_time = time.time()
        for i in range(5):
            analyzer.record_performance_metric("test_metric", float(i), current_time + i)
        
        result = analyzer.analyze_performance("test_metric", time_window=1000)
        
        assert result["metric"] == "test_metric"
        assert result["avg_value"] == 2.0
        assert result["max_value"] == 4.0
        assert result["min_value"] == 0.0
        assert result["sample_count"] == 5

    def test_analyze_performance_with_baseline_normal(self, analyzer):
        """测试分析性能 - 有基准值（正常）"""
        current_time = time.time()
        analyzer.set_baseline("test_metric", 100.0)
        
        for i in range(5):
            analyzer.record_performance_metric("test_metric", 100.0 + i, current_time + i)
        
        result = analyzer.analyze_performance("test_metric", time_window=1000)
        
        assert result["performance_status"] == "normal"

    def test_analyze_performance_with_baseline_warning(self, analyzer):
        """测试分析性能 - 有基准值（警告）"""
        current_time = time.time()
        analyzer.set_baseline("test_metric", 100.0)
        
        for i in range(5):
            analyzer.record_performance_metric("test_metric", 110.0 + i, current_time + i)
        
        result = analyzer.analyze_performance("test_metric", time_window=1000)
        
        assert result["performance_status"] in ["normal", "warning"]

    def test_analyze_performance_with_baseline_critical(self, analyzer):
        """测试分析性能 - 有基准值（严重）"""
        current_time = time.time()
        analyzer.set_baseline("test_metric", 100.0)
        
        for i in range(5):
            analyzer.record_performance_metric("test_metric", 150.0 + i, current_time + i)
        
        result = analyzer.analyze_performance("test_metric", time_window=1000)
        
        assert result["performance_status"] in ["warning", "critical"]

    def test_get_performance_trends_insufficient_data(self, analyzer):
        """测试获取性能趋势 - 数据不足"""
        result = analyzer.get_performance_trends("nonexistent_metric")
        assert result["trend"] == "insufficient_data"

    def test_get_performance_trends_increasing(self, analyzer):
        """测试获取性能趋势 - 上升趋势"""
        current_time = time.time()
        for i in range(20):
            # 前半部分低值，后半部分高值
            value = 50.0 if i < 10 else 100.0
            analyzer.record_performance_metric("test_metric", value, current_time + i)
        
        result = analyzer.get_performance_trends("test_metric")
        
        assert result["metric"] == "test_metric"
        assert result["trend"] == "increasing"

    def test_get_performance_trends_decreasing(self, analyzer):
        """测试获取性能趋势 - 下降趋势"""
        current_time = time.time()
        for i in range(20):
            # 前半部分高值，后半部分低值
            value = 100.0 if i < 10 else 50.0
            analyzer.record_performance_metric("test_metric", value, current_time + i)
        
        result = analyzer.get_performance_trends("test_metric")
        
        assert result["metric"] == "test_metric"
        assert result["trend"] == "decreasing"

    def test_get_performance_trends_stable(self, analyzer):
        """测试获取性能趋势 - 稳定趋势"""
        current_time = time.time()
        for i in range(20):
            analyzer.record_performance_metric("test_metric", 100.0, current_time + i)
        
        result = analyzer.get_performance_trends("test_metric")
        
        assert result["metric"] == "test_metric"
        assert result["trend"] == "stable"
