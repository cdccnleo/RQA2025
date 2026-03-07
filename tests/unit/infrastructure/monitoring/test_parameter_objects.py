#!/usr/bin/env python3
"""
RQA2025 基础设施层参数对象单元测试

测试参数对象类的功能和正确性。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from datetime import datetime
from dataclasses import is_dataclass

from src.infrastructure.monitoring.core.constants import DEFAULT_PAGE_SIZE
from src.infrastructure.monitoring.core.parameter_objects import (
    MetricRecordConfig,
    MetricsQueryConfig,
    AlertConditionConfig,
    AlertRuleConfig,
    MonitoringConfig,
    PerformanceMetricsConfig,
    CoverageCollectionConfig,
    ResourceUsageConfig,
    HealthCheckConfig,
    PrometheusExportConfig,
    OptimizationSuggestionConfig,
    DataPersistenceConfig,
    LoggerPoolStatsConfig
)


class TestParameterObjects(unittest.TestCase):
    """参数对象测试类"""

    def test_metric_record_config(self):
        """测试指标记录配置"""
        config = MetricRecordConfig(
            name="test_metric",
            value=42.0,
            tags={"service": "test"},
            app_name="test_app"
        )

        self.assertEqual(config.name, "test_metric")
        self.assertEqual(config.value, 42.0)
        self.assertEqual(config.tags, {"service": "test"})
        self.assertEqual(config.app_name, "test_app")
        self.assertIsInstance(config, MetricRecordConfig)
        self.assertTrue(is_dataclass(MetricRecordConfig))

    def test_metrics_query_config(self):
        """测试指标查询配置"""
        start_time = datetime.now()
        end_time = datetime.now()

        config = MetricsQueryConfig(
            metric_names=["cpu", "memory"],
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            tags_filter={"env": "prod"},
            aggregation_type="max",
            group_by_tags=["env", "service"],
            limit=50,
            offset=10
        )

        self.assertEqual(config.metric_names, ["cpu", "memory"])
        self.assertEqual(config.start_time, start_time.timestamp())
        self.assertEqual(config.end_time, end_time.timestamp())
        self.assertEqual(config.tags_filter, {"env": "prod"})
        self.assertEqual(config.aggregation_type, "max")
        self.assertEqual(config.group_by_tags, ["env", "service"])
        self.assertEqual(config.limit, 50)
        self.assertEqual(config.offset, 10)

    def test_alert_condition_config(self):
        """测试告警条件配置"""
        config = AlertConditionConfig(
            field="cpu_percent",
            value=80.0,
            operator="gt",
            threshold=85.0,
            comparison_type="percentage"
        )

        self.assertEqual(config.field, "cpu_percent")
        self.assertEqual(config.value, 80.0)
        self.assertEqual(config.operator, "gt")
        self.assertEqual(config.threshold, 85.0)
        self.assertEqual(config.comparison_type, "percentage")

    def test_alert_rule_config(self):
        """测试告警规则配置"""
        condition = AlertConditionConfig(
            field="memory_percent",
            value=90.0,
            operator="gt"
        )

        config = AlertRuleConfig(
            rule_id="high_memory",
            name="内存使用过高",
            description="系统内存使用率过高",
            condition=condition,
            level="warning",
            channels=["console", "email"],
            enabled=True,
            cooldown=300
        )

        self.assertEqual(config.rule_id, "high_memory")
        self.assertEqual(config.name, "内存使用过高")
        self.assertEqual(config.description, "系统内存使用率过高")
        self.assertEqual(config.level, "warning")
        self.assertEqual(config.channels, ["console", "email"])
        self.assertTrue(config.enabled)
        self.assertEqual(config.cooldown, 300)

    def test_monitoring_config(self):
        """测试监控配置"""
        config = MonitoringConfig(
            collection_interval=120,
            max_history_size=2000,
            alert_thresholds={
                'cpu_high': 85.0,
                'memory_high': 90.0
            }
        )

        self.assertEqual(config.collection_interval, 120)
        self.assertEqual(config.max_history_size, 2000)
        self.assertEqual(config.alert_thresholds['cpu_high'], 85.0)
        self.assertEqual(config.alert_thresholds['memory_high'], 90.0)

    def test_performance_metrics_config(self):
        """测试性能指标收集配置"""
        config = PerformanceMetricsConfig(
            include_detailed_stats=True,
            calculate_percentiles=True,
            include_memory_stats=True,
            include_thread_stats=False,
            include_queue_stats=True,
            include_error_stats=True,
            include_performance_trends=True,
            performance_thresholds={'response_time': 100.0},
            alert_on_anomalies=True,
            export_format="json",
            export_path="/tmp/metrics",
            create_charts=True,
            summary_report=True,
            enable_ai_insights=False
        )

        self.assertTrue(config.include_detailed_stats)
        self.assertTrue(config.calculate_percentiles)
        self.assertTrue(config.include_memory_stats)
        self.assertFalse(config.include_thread_stats)
        self.assertTrue(config.include_queue_stats)
        self.assertTrue(config.include_error_stats)
        self.assertTrue(config.include_performance_trends)
        self.assertEqual(config.performance_thresholds, {'response_time': 100.0})
        self.assertTrue(config.alert_on_anomalies)
        self.assertEqual(config.export_format, "json")
        self.assertEqual(config.export_path, "/tmp/metrics")
        self.assertTrue(config.create_charts)
        self.assertTrue(config.summary_report)
        self.assertFalse(config.enable_ai_insights)

    def test_coverage_collection_config(self):
        """测试覆盖率收集配置"""
        config = CoverageCollectionConfig(
            include_line_coverage=True,
            include_branch_coverage=True,
            include_function_coverage=True,
            include_class_coverage=True,
            exclude_patterns=["test_*", "*_mock.py"],
            source_dirs=["src", "lib"],
            report_formats=["html", "xml", "json"],
            minimum_coverage=85.0,
            fail_under_minimum=True
        )

        self.assertTrue(config.include_line_coverage)
        self.assertTrue(config.include_branch_coverage)
        self.assertTrue(config.include_function_coverage)
        self.assertTrue(config.include_class_coverage)
        self.assertEqual(config.exclude_patterns, ["test_*", "*_mock.py"])
        self.assertEqual(config.source_dirs, ["src", "lib"])
        self.assertEqual(config.report_formats, ["html", "xml", "json"])
        self.assertEqual(config.minimum_coverage, 85.0)
        self.assertTrue(config.fail_under_minimum)

    def test_resource_usage_config(self):
        """测试资源使用收集配置"""
        config = ResourceUsageConfig(
            include_cpu=True,
            include_memory=True,
            include_disk=True,
            include_network=True,
            include_processes=False,
            include_system_load=True,
            sampling_interval=2.0,
            collection_duration=120,
            enable_historical_tracking=True,
            alert_thresholds={
                'cpu_percent': 90.0,
                'memory_percent': 95.0,
                'disk_percent': 98.0
            }
        )

        self.assertTrue(config.include_cpu)
        self.assertTrue(config.include_memory)
        self.assertTrue(config.include_disk)
        self.assertTrue(config.include_network)
        self.assertFalse(config.include_processes)
        self.assertTrue(config.include_system_load)
        self.assertEqual(config.sampling_interval, 2.0)
        self.assertEqual(config.collection_duration, 120)
        self.assertTrue(config.enable_historical_tracking)
        self.assertEqual(config.alert_thresholds['cpu_percent'], 90.0)

    def test_health_check_config(self):
        """测试健康检查配置"""
        config = HealthCheckConfig(
            component_name="database",
            check_type="comprehensive",
            timeout_seconds=60,
            include_dependencies=True,
            include_performance_metrics=False,
            custom_checks=["replication", "latency"],
            environment_context={"region": "us-east-1"}
        )

        self.assertEqual(config.component_name, "database")
        self.assertEqual(config.check_type, "comprehensive")
        self.assertEqual(config.timeout_seconds, 60)
        self.assertTrue(config.include_dependencies)
        self.assertFalse(config.include_performance_metrics)
        self.assertEqual(config.custom_checks, ["replication", "latency"])
        self.assertEqual(config.environment_context, {"region": "us-east-1"})

    def test_prometheus_export_config(self):
        """测试Prometheus导出配置"""
        config = PrometheusExportConfig(
            include_help_text=True,
            include_type_info=True,
            metric_prefix="rqa2025",
            label_names=["service", "instance", "env"],
            default_labels={"env": "production"},
            export_timeout=60,
            enable_compression=True
        )

        self.assertTrue(config.include_help_text)
        self.assertTrue(config.include_type_info)
        self.assertEqual(config.metric_prefix, "rqa2025")
        self.assertEqual(config.label_names, ["service", "instance", "env"])
        self.assertEqual(config.default_labels, {"env": "production"})
        self.assertEqual(config.export_timeout, 60)
        self.assertTrue(config.enable_compression)

    def test_optimization_suggestion_config(self):
        """测试优化建议配置"""
        config = OptimizationSuggestionConfig(
            enable_ai_suggestions=True,
            max_suggestions_per_category=15,
            suggestion_thresholds={
                'coverage_drop': 10.0,
                'performance_degradation': 15.0,
                'memory_increase': 25.0
            },
            priority_weights={
                'critical': 5,
                'high': 3,
                'medium': 2,
                'low': 1
            }
        )

        self.assertTrue(config.enable_ai_suggestions)
        self.assertEqual(config.max_suggestions_per_category, 15)
        self.assertEqual(config.suggestion_thresholds['coverage_drop'], 10.0)
        self.assertEqual(config.priority_weights['critical'], 5)

    def test_data_persistence_config(self):
        """测试数据持久化配置"""
        config = DataPersistenceConfig(
            enable_file_storage=True,
            enable_database_storage=False,
            storage_path="data/monitoring",
            max_file_age_days=30,
            compression_enabled=True,
            backup_enabled=True,
            backup_interval_hours=24,
            retention_policy="time_based",
            max_storage_size_mb=2048
        )

        self.assertTrue(config.enable_file_storage)
        self.assertFalse(config.enable_database_storage)
        self.assertEqual(config.storage_path, "data/monitoring")
        self.assertEqual(config.max_file_age_days, 30)
        self.assertTrue(config.compression_enabled)
        self.assertTrue(config.backup_enabled)
        self.assertEqual(config.backup_interval_hours, 24)
        self.assertEqual(config.retention_policy, "time_based")
        self.assertEqual(config.max_storage_size_mb, 2048)

    def test_logger_pool_stats_config(self):
        """测试Logger池统计配置"""
        config = LoggerPoolStatsConfig(
            include_hit_rate=True,
            include_memory_usage=True,
            include_pool_size=True,
            include_access_times=True,
            include_error_counts=True,
            calculate_percentiles=True,
            percentile_values=[25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
            enable_trend_analysis=True,
            trend_window_size=15
        )

        self.assertTrue(config.include_hit_rate)
        self.assertTrue(config.include_memory_usage)
        self.assertTrue(config.include_pool_size)
        self.assertTrue(config.include_access_times)
        self.assertTrue(config.include_error_counts)
        self.assertTrue(config.calculate_percentiles)
        self.assertEqual(config.percentile_values, [25.0, 50.0, 75.0, 90.0, 95.0, 99.0])
        self.assertTrue(config.enable_trend_analysis)
        self.assertEqual(config.trend_window_size, 15)
        self.assertEqual(config.trend_window_size, 15)

    def test_config_defaults(self):
        """测试配置默认值"""
        # 测试各个配置类的默认值
        metric_config = MetricRecordConfig(name="test", value=1.0)
        self.assertIsNone(metric_config.tags)
        self.assertIsNone(metric_config.timestamp)
        self.assertIsNone(metric_config.app_name)

        query_config = MetricsQueryConfig(metric_names=["cpu"])
        self.assertEqual(query_config.metric_names, ["cpu"])
        self.assertEqual(query_config.aggregation_type, "avg")
        self.assertEqual(query_config.group_by_tags, None)
        self.assertEqual(query_config.limit, DEFAULT_PAGE_SIZE)
        self.assertEqual(query_config.offset, 0)

    def test_config_immutability(self):
        """测试配置对象的基本不变性"""
        config = AlertRuleConfig(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            condition=AlertConditionConfig(field="cpu", value=80.0)
        )

        # 验证对象创建成功
        self.assertEqual(config.rule_id, "test_rule")
        self.assertEqual(config.name, "Test Rule")
        self.assertEqual(config.description, "A test rule")
        self.assertEqual(config.condition.field, "cpu")
        self.assertEqual(config.condition.value, 80.0)


if __name__ == '__main__':
    unittest.main()
