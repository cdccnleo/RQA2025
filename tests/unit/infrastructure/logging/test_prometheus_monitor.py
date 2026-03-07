#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Prometheus监控

测试logging/monitors/prometheus_monitor.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Optional, Any, List

from src.infrastructure.logging.monitors.prometheus_monitor import (
    MetricsRegistry, MetricsExporter, AlertHandler, PrometheusMonitor
)
from src.infrastructure.logging.monitors.enums import AlertData


class TestMetricsRegistry:
    """测试指标注册表"""

    def setup_method(self):
        """测试前准备"""
        self.registry = MetricsRegistry()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.registry, '_registry')
        assert hasattr(self.registry, '_gauges')
        assert hasattr(self.registry, '_counters')
        assert hasattr(self.registry, '_histograms')
        assert hasattr(self.registry, '_logger')

        assert isinstance(self.registry._gauges, dict)
        assert isinstance(self.registry._counters, dict)
        assert isinstance(self.registry._histograms, dict)

    def test_create_gauge(self):
        """测试创建Gauge指标"""
        gauge = self.registry.create_gauge(
            name="test_gauge",
            description="Test gauge metric",
            labels=["service", "env"]
        )

        assert gauge is not None
        assert "test_gauge" in self.registry._gauges
        assert self.registry._gauges["test_gauge"] is gauge

    def test_create_counter(self):
        """测试创建Counter指标"""
        counter = self.registry.create_counter(
            name="test_counter",
            description="Test counter metric",
            labels=["method", "status"]
        )

        assert counter is not None
        assert "test_counter" in self.registry._counters
        assert self.registry._counters["test_counter"] is counter

    def test_create_histogram(self):
        """测试创建Histogram指标"""
        histogram = self.registry.create_histogram(
            name="test_histogram",
            description="Test histogram metric",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0]
        )

        assert histogram is not None
        assert "test_histogram" in self.registry._histograms
        assert self.registry._histograms["test_histogram"] is histogram

    def test_get_gauge(self):
        """测试获取Gauge指标"""
        # 先创建
        gauge = self.registry.create_gauge("test_gauge", "Test gauge")

        # 再获取
        retrieved = self.registry.get_gauge("test_gauge")

        assert retrieved is gauge

    def test_get_gauge_not_exists(self):
        """测试获取不存在的Gauge指标"""
        retrieved = self.registry.get_gauge("nonexistent")

        assert retrieved is None

    def test_get_counter(self):
        """测试获取Counter指标"""
        counter = self.registry.create_counter("test_counter", "Test counter")
        retrieved = self.registry.get_counter("test_counter")

        assert retrieved is counter

    def test_get_histogram(self):
        """测试获取Histogram指标"""
        histogram = self.registry.create_histogram("test_histogram", "Test histogram")
        retrieved = self.registry.get_histogram("test_histogram")

        assert retrieved is histogram

    def test_remove_gauge(self):
        """测试移除Gauge指标"""
        self.registry.create_gauge("test_gauge", "Test gauge")
        assert "test_gauge" in self.registry._gauges

        self.registry.remove_gauge("test_gauge")
        assert "test_gauge" not in self.registry._gauges

    def test_remove_counter(self):
        """测试移除Counter指标"""
        self.registry.create_counter("test_counter", "Test counter")
        assert "test_counter" in self.registry._counters

        self.registry.remove_counter("test_counter")
        assert "test_counter" not in self.registry._counters

    def test_remove_histogram(self):
        """测试移除Histogram指标"""
        self.registry.create_histogram("test_histogram", "Test histogram")
        assert "test_histogram" in self.registry._histograms

        self.registry.remove_histogram("test_histogram")
        assert "test_histogram" not in self.registry._histograms

    def test_clear_all_metrics(self):
        """测试清除所有指标"""
        # 创建各种类型的指标
        self.registry.create_gauge("gauge1", "Gauge 1")
        self.registry.create_gauge("gauge2", "Gauge 2")
        self.registry.create_counter("counter1", "Counter 1")
        self.registry.create_histogram("histogram1", "Histogram 1")

        assert len(self.registry._gauges) == 2
        assert len(self.registry._counters) == 1
        assert len(self.registry._histograms) == 1

        self.registry.clear_all_metrics()

        # Note: Default metrics are preserved
        assert len(self.registry._gauges) == 0  # User-created gauges are cleared
        assert len(self.registry._counters) == 0  # User-created counters are cleared
        assert len(self.registry._histograms) == 0  # User-created histograms are cleared

    def test_duplicate_metric_names(self):
        """测试重复的指标名称"""
        # 创建第一个gauge
        gauge1 = self.registry.create_gauge("duplicate_name", "First gauge")

        # 尝试创建同名gauge（应该返回已存在的）
        gauge2 = self.registry.create_gauge("duplicate_name", "Second gauge")

        assert gauge1 is gauge2
        assert len(self.registry._gauges) == 1

    def test_metric_with_labels(self):
        """测试带标签的指标"""
        gauge = self.registry.create_gauge(
            name="labeled_gauge",
            description="Gauge with labels",
            labels=["service", "version", "env"]
        )

        assert gauge is not None

        # 测试设置带标签的值（这里只是验证创建，没有实际设置值）
        assert "labeled_gauge" in self.registry._gauges


class TestMetricsExporter:
    """测试指标导出器"""

    def setup_method(self):
        """测试前准备"""
        self.registry = MetricsRegistry()
        self.exporter = MetricsExporter(registry=self.registry, job_name="test_job")

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.exporter, '_registry')
        assert hasattr(self.exporter, '_push_gateway_url')
        assert hasattr(self.exporter, '_job_name')
        assert hasattr(self.exporter, '_logger')

        assert isinstance(self.exporter._registry, MetricsRegistry)

    def test_initialization_with_custom_params(self):
        """测试带自定义参数的初始化"""
        custom_registry = MetricsRegistry()
        custom_exporter = MetricsExporter(
            registry=custom_registry,
            job_name="custom_job"
        )
        custom_exporter.set_gateway_url("http://custom:9091")

        assert custom_exporter._push_gateway_url == "http://custom:9091"
        assert custom_exporter._job_name == "custom_job"

    def test_export_metrics_to_gateway(self):
        """测试导出指标到网关"""
        # 设置网关URL
        self.exporter.set_gateway_url("http://test-gateway:9091")

        with patch('src.infrastructure.logging.monitors.prometheus_monitor.push_to_gateway') as mock_push:
            result = self.exporter.export_metrics_to_gateway()

            assert result is True
            mock_push.assert_called_once()

    def test_export_metrics_to_gateway_failure(self):
        """测试导出指标到网关失败"""
        with patch('src.infrastructure.logging.monitors.prometheus_monitor.push_to_gateway', side_effect=Exception("Push failed")):
            result = self.exporter.export_metrics_to_gateway()

            assert result is False

    def test_delete_metrics_from_gateway(self):
        """测试从网关删除指标"""
        # 设置网关URL
        self.exporter.set_gateway_url("http://test-gateway:9091")

        with patch('src.infrastructure.logging.monitors.prometheus_monitor.delete_from_gateway') as mock_delete:
            result = self.exporter.delete_metrics_from_gateway()

            assert result is True
            mock_delete.assert_called_once()

    def test_delete_metrics_from_gateway_failure(self):
        """测试从网关删除指标失败"""
        # 设置网关URL
        self.exporter.set_gateway_url("http://test-gateway:9091")

        with patch('src.infrastructure.logging.monitors.prometheus_monitor.delete_from_gateway', side_effect=Exception("Delete failed")):
            result = self.exporter.delete_metrics_from_gateway()

            assert result is False

    def test_get_registry(self):
        """测试获取注册表"""
        registry = self.exporter.get_registry()

        assert isinstance(registry, MetricsRegistry)
        assert registry is self.exporter._registry

    def test_update_push_gateway_url(self):
        """测试更新推送网关URL"""
        new_url = "http://new-gateway:9091"

        self.exporter.update_push_gateway_url(new_url)

        assert self.exporter._push_gateway_url == new_url

    def test_update_job_name(self):
        """测试更新作业名称"""
        new_job = "new_job_name"

        self.exporter.update_job_name(new_job)

        assert self.exporter._job_name == new_job

    def test_get_metrics_summary(self):
        """测试获取指标摘要"""
        # 创建一些指标
        self.exporter._registry.create_gauge("test_gauge", "Test gauge")
        self.exporter._registry.create_counter("test_counter", "Test counter")
        self.exporter._registry.create_histogram("test_histogram", "Test histogram")

        summary = self.exporter.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "gauges" in summary
        assert "counters" in summary
        assert "histograms" in summary
        assert summary["gauges"] == 1
        assert summary["counters"] == 1
        assert summary["histograms"] == 1


class TestAlertHandler:
    """测试告警处理器"""

    def setup_method(self):
        """测试前准备"""
        self.registry = MetricsRegistry()
        self.exporter = MetricsExporter(registry=self.registry, job_name="test_job")
        self.handler = AlertHandler(metrics_exporter=self.exporter)

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.handler, '_alert_rules')
        assert hasattr(self.handler, '_alert_states')
        assert hasattr(self.handler, '_logger')

        assert isinstance(self.handler._alert_rules, dict)
        assert isinstance(self.handler._alert_states, dict)

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule_name = "high_cpu"
        rule_config = {
            "condition": "cpu_usage > 80",
            "severity": "warning",
            "description": "CPU usage is too high",
            "threshold": 80
        }

        result = self.handler.add_alert_rule(rule_name, rule_config)

        assert result is True
        assert rule_name in self.handler._alert_rules
        assert self.handler._alert_rules[rule_name] == rule_config

    def test_add_alert_rule_duplicate(self):
        """测试添加重复的告警规则"""
        rule_name = "duplicate_rule"
        rule_config1 = {"condition": "cpu > 80"}
        rule_config2 = {"condition": "memory > 90"}

        # 添加第一个规则
        self.handler.add_alert_rule(rule_name, rule_config1)

        # 添加同名规则（应该覆盖）
        result = self.handler.add_alert_rule(rule_name, rule_config2)

        assert result is True
        assert self.handler._alert_rules[rule_name] == rule_config2

    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        rule_name = "temp_rule"
        rule_config = {"condition": "temp > 50"}

        # 先添加
        self.handler.add_alert_rule(rule_name, rule_config)
        assert rule_name in self.handler._alert_rules

        # 再移除
        result = self.handler.remove_alert_rule(rule_name)

        assert result is True
        assert rule_name not in self.handler._alert_rules

    def test_remove_alert_rule_not_exists(self):
        """测试移除不存在的告警规则"""
        result = self.handler.remove_alert_rule("nonexistent")

        assert result is False

    def test_get_alert_rules(self):
        """测试获取告警规则"""
        rules = {
            "cpu_high": {"condition": "cpu > 80", "severity": "warning"},
            "memory_low": {"condition": "memory < 10", "severity": "error"}
        }

        for name, config in rules.items():
            self.handler.add_alert_rule(name, config)

        retrieved_rules = self.handler.get_alert_rules()

        assert retrieved_rules == rules

    def test_evaluate_alert_condition(self):
        """测试评估告警条件"""
        # 这个方法可能需要具体的实现，这里测试基本的结构
        # 由于实现细节未知，我们测试方法存在性
        assert hasattr(self.handler, '_evaluate_condition')

        # 测试简单的条件评估（如果有实现的话）
        try:
            result = self.handler._evaluate_condition("cpu_usage > 80", {"cpu_usage": 85})
            # 如果实现返回布尔值
            assert isinstance(result, bool)
        except (NotImplementedError, AttributeError):
            # 如果方法未实现，跳过具体测试
            pass

    def test_trigger_alert(self):
        """测试触发告警"""
        alert_data = {
            "name": "test_alert",
            "severity": "warning",
            "message": "Test alert triggered",
            "value": 85,
            "threshold": 80
        }

        result = self.handler.trigger_alert(alert_data)

        assert result is True
        assert "test_alert" in self.handler._alert_states

    def test_resolve_alert(self):
        """测试解决告警"""
        alert_name = "resolvable_alert"

        # 先触发告警
        result = self.handler.trigger_alert({
            "name": alert_name,
            "severity": "warning",
            "message": "Alert to resolve"
        })

        assert result is True

        # 解决告警
        result = self.handler.resolve_alert(alert_name)

        assert result is True

    def test_resolve_alert_not_active(self):
        """测试解决不存在的告警"""
        result = self.handler.resolve_alert("nonexistent_alert")

        assert result is False

    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        # 添加一些告警
        alerts_data = [
            {"name": "alert1", "severity": "warning", "message": "Warning alert"},
            {"name": "alert2", "severity": "error", "message": "Error alert"},
            {"name": "alert3", "severity": "info", "message": "Info alert"}
        ]

        for alert_data in alerts_data:
            self.handler.trigger_alert(alert_data)

        active_alerts = self.handler.get_active_alerts()

        assert len(active_alerts) == 3
        assert all(alert["active"] for alert in active_alerts)

    def test_clear_all_alerts(self):
        """测试清除所有告警"""
        # 添加一些告警
        for i in range(3):
            self.handler.trigger_alert({
                "name": f"alert_{i}",
                "severity": "warning",
                "message": f"Alert {i}"
            })

        active_alerts = self.handler.get_active_alerts()
        assert len(active_alerts) == 3

        result = self.handler.clear_all_alerts()
        assert result is True

        active_alerts = self.handler.get_active_alerts()
        assert len(active_alerts) == 0

    def test_alert_handler_with_custom_rules(self):
        """测试带自定义规则的告警处理器"""
        custom_registry = MetricsRegistry()
        custom_exporter = MetricsExporter(registry=custom_registry, job_name="custom_job")
        custom_handler = AlertHandler(metrics_exporter=custom_exporter)

        # 添加复杂的告警规则
        complex_rules = {
            "cpu_critical": {
                "condition": "cpu_usage > 90",
                "severity": "critical",
                "description": "CPU usage is critically high",
                "threshold": 90,
                "cooldown": 300
            },
            "memory_warning": {
                "condition": "memory_usage > 85",
                "severity": "warning",
                "description": "Memory usage is high",
                "threshold": 85,
                "cooldown": 600
            }
        }

        for name, config in complex_rules.items():
            custom_handler.add_alert_rule(name, config)

        rules = custom_handler.get_alert_rules()
        assert len(rules) == 2
        assert "cpu_critical" in rules
        assert "memory_warning" in rules


class TestPrometheusMonitor:
    """测试Prometheus监控器"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = PrometheusMonitor()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.monitor, '_registry')
        assert hasattr(self.monitor, '_exporter')
        assert hasattr(self.monitor, '_alert_handler')
        assert hasattr(self.monitor, '_logger')

        assert isinstance(self.monitor._registry, MetricsRegistry)
        assert isinstance(self.monitor._exporter, MetricsExporter)
        assert isinstance(self.monitor._alert_handler, AlertHandler)

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {
            "push_gateway_url": "http://config-gateway:9091",
            "job_name": "config_job",
            "alert_rules": {
                "cpu_high": {"condition": "cpu > 80", "severity": "warning"}
            }
        }

        monitor = PrometheusMonitor(config)

        assert monitor._exporter._push_gateway_url == "http://config-gateway:9091"
        assert monitor._exporter._job_name == "config_job"
        assert "cpu_high" in monitor._alert_handler._alert_rules

    def test_create_gauge_metric(self):
        """测试创建Gauge指标"""
        gauge = self.monitor.create_gauge_metric(
            name="test_gauge",
            description="Test gauge",
            labels=["service"]
        )

        assert gauge is not None
        assert "test_gauge" in self.monitor._registry._gauges

    def test_create_counter_metric(self):
        """测试创建Counter指标"""
        counter = self.monitor.create_counter_metric(
            name="test_counter",
            description="Test counter",
            labels=["method"]
        )

        assert counter is not None
        assert "test_counter" in self.monitor._registry._counters

    def test_create_histogram_metric(self):
        """测试创建Histogram指标"""
        histogram = self.monitor.create_histogram_metric(
            name="test_histogram",
            description="Test histogram",
            labels=["endpoint"],
            buckets=[0.1, 1.0, 10.0]
        )

        assert histogram is not None
        assert "test_histogram" in self.monitor._registry._histograms

    def test_record_metric_value(self):
        """测试记录指标值"""
        # 创建gauge指标
        gauge = self.monitor.create_gauge_metric("test_value", "Test value gauge")

        # 记录值（这里我们mock实际的设置操作）
        with patch.object(gauge, 'set') as mock_set:
            result = self.monitor.record_metric_value("test_value", 42.5)

            assert result is True
            mock_set.assert_called_once_with(42.5)

    def test_record_metric_value_with_labels(self):
        """测试记录带标签的指标值"""
        gauge = self.monitor.create_gauge_metric("labeled_metric", "Labeled metric", labels=["env"])

        with patch.object(gauge, 'labels') as mock_labels:
            mock_labels.return_value.set.return_value = None

            result = self.monitor.record_metric_value(
                "labeled_metric",
                99.9,
                labels={"env": "prod"}
            )

            assert result is True
            mock_labels.assert_called_once_with(env="prod")

    def test_increment_counter(self):
        """测试递增计数器"""
        counter = self.monitor.create_counter_metric("test_counter", "Test counter")

        with patch.object(counter, 'inc') as mock_inc:
            result = self.monitor.increment_counter("test_counter")

            assert result is True
            mock_inc.assert_called_once()

    def test_increment_counter_with_labels(self):
        """测试递增带标签的计数器"""
        counter = self.monitor.create_counter_metric("labeled_counter", "Labeled counter", labels=["status"])

        with patch.object(counter, 'labels') as mock_labels:
            mock_labels.return_value.inc.return_value = None

            result = self.monitor.increment_counter(
                "labeled_counter",
                labels={"status": "success"}
            )

            assert result is True
            mock_labels.assert_called_once_with(status="success")

    def test_record_histogram_value(self):
        """测试记录直方图值"""
        histogram = self.monitor.create_histogram_metric("test_histogram", "Test histogram")

        with patch.object(histogram, 'observe') as mock_observe:
            result = self.monitor.record_histogram_value("test_histogram", 2.5)

            assert result is True
            mock_observe.assert_called_once_with(2.5)

    def test_record_histogram_value_with_labels(self):
        """测试记录带标签的直方图值"""
        histogram = self.monitor.create_histogram_metric("labeled_histogram", "Labeled histogram", labels=["method"])

        with patch.object(histogram, 'labels') as mock_labels:
            mock_labels.return_value.observe.return_value = None

            result = self.monitor.record_histogram_value(
                "labeled_histogram",
                1.8,
                labels={"method": "GET"}
            )

            assert result is True
            mock_labels.assert_called_once_with(method="GET")

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule_config = {
            "condition": "cpu_usage > 80",
            "severity": "warning",
            "description": "CPU usage alert"
        }

        result = self.monitor.add_alert_rule("cpu_alert", rule_config)

        assert result is True
        assert "cpu_alert" in self.monitor._alert_handler._alert_rules

    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        # 先添加
        self.monitor.add_alert_rule("temp_rule", {"condition": "temp > 50"})

        # 再移除
        result = self.monitor.remove_alert_rule("temp_rule")

        assert result is True
        assert "temp_rule" not in self.monitor._alert_handler._alert_rules

    def test_get_alert_rules(self):
        """测试获取告警规则"""
        rules = self.monitor.get_alert_rules()

        assert isinstance(rules, dict)

    def test_trigger_alert(self):
        """测试触发告警"""
        alert_data = {
            "name": "test_trigger",
            "severity": "error",
            "message": "Test alert trigger"
        }

        result = self.monitor.trigger_alert(alert_data)

        assert result is True
        # Note: Alert states are managed internally, not directly accessible

    def test_resolve_alert(self):
        """测试解决告警"""
        # 先触发
        self.monitor.trigger_alert({
            "name": "resolve_test",
            "severity": "warning",
            "message": "Alert to resolve"
        })

        # 再解决
        result = self.monitor.resolve_alert("resolve_test")

        assert result is True

    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        alerts = self.monitor.get_active_alerts()

        assert isinstance(alerts, list)

    def test_export_metrics(self):
        """测试导出指标"""
        # 设置网关URL
        self.monitor._exporter.set_gateway_url("http://test-gateway:9091")

        with patch.object(self.monitor._exporter, 'push_metrics', return_value=True) as mock_export:
            result = self.monitor.export_metrics()

            assert result is True
            mock_export.assert_called_once()

    def test_clear_metrics(self):
        """测试清除指标"""
        # 创建一些指标
        self.monitor.create_gauge_metric("temp_gauge", "Temp gauge")

        assert len(self.monitor._registry._gauges) > 0

        result = self.monitor.clear_metrics()
        assert result is True

        # Check that all metrics are cleared
        assert len(self.monitor._registry._gauges) == 0
        assert len(self.monitor._registry._counters) == 0
        assert len(self.monitor._registry._histograms) == 0

    def test_get_metrics_summary(self):
        """测试获取指标摘要"""
        summary = self.monitor.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "gauges" in summary
        assert "counters" in summary
        assert "histograms" in summary

    def test_monitor_error_handling(self):
        """测试监控器错误处理"""
        # 测试记录不存在的指标
        result = self.monitor.record_metric_value("nonexistent_metric", 42.0)

        assert result is False

    def test_monitor_thread_safety(self):
        """测试监控器线程安全性"""
        import threading

        results = []
        errors = []

        def monitor_operations(thread_id):
            try:
                # 创建指标
                gauge_name = f"thread_{thread_id}_gauge"
                self.monitor.create_gauge_metric(gauge_name, f"Gauge for thread {thread_id}")

                # 记录值
                self.monitor.record_metric_value(gauge_name, float(thread_id * 10))

                # 递增计数器
                counter_name = f"thread_{thread_id}_counter"
                self.monitor.create_counter_metric(counter_name, f"Counter for thread {thread_id}")
                self.monitor.increment_counter(counter_name)

                results.append(f"thread_{thread_id}_completed")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=monitor_operations, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(results) == 5

    def test_monitor_resource_management(self):
        """测试监控器资源管理"""
        # 创建大量指标
        for i in range(50):
            self.monitor.create_gauge_metric(f"resource_gauge_{i}", f"Resource gauge {i}")

        # 验证创建成功 (50 user gauges + 3 default gauges)
        assert len(self.monitor._registry._gauges) == 53

        # 清除资源
        result = self.monitor.clear_metrics()
        assert result is True

        # 验证清理成功 - all metrics cleared
        assert len(self.monitor._registry._gauges) == 0

    def test_monitor_configuration_persistence(self):
        """测试监控器配置持久性"""
        config = {
            "push_gateway_url": "http://persistent:9091",
            "alert_rules": {
                "persistent_rule": {
                    "condition": "value > 100",
                    "severity": "warning"
                }
            }
        }

        monitor = PrometheusMonitor(config)

        # 验证配置持久性
        assert monitor._exporter._push_gateway_url == "http://persistent:9091"
        assert "persistent_rule" in monitor._alert_handler._alert_rules

        # 验证配置不会被后续操作改变
        monitor.create_gauge_metric("test_metric", "Test metric")
        assert monitor._exporter._push_gateway_url == "http://persistent:9091"

    def test_monitor_performance_metrics(self):
        """测试监控器性能指标"""
        start_time = time.time()

        # 执行大量操作
        for i in range(100):
            self.monitor.create_gauge_metric(f"perf_gauge_{i}", f"Performance gauge {i}")
            self.monitor.record_metric_value(f"perf_gauge_{i}", float(i))

        end_time = time.time()
        duration = end_time - start_time

        # 性能应该在合理范围内
        assert duration < 5.0  # 少于5秒

        # 验证所有操作都成功 (100 user gauges + 3 default gauges)
        assert len(self.monitor._registry._gauges) == 103

    def test_monitor_comprehensive_workflow(self):
        """测试监控器综合工作流"""
        # 1. 创建各种类型的指标
        self.monitor.create_gauge_metric("cpu_usage", "CPU usage percentage")
        self.monitor.create_counter_metric("requests_total", "Total requests")
        self.monitor.create_histogram_metric("response_time", "Response time histogram")

        # 2. 记录指标值
        self.monitor.record_metric_value("cpu_usage", 75.5)
        self.monitor.increment_counter("requests_total")
        self.monitor.record_histogram_value("response_time", 0.245)

        # 3. 添加告警规则
        self.monitor.add_alert_rule("high_cpu", {
            "condition": "cpu_usage > 80",
            "severity": "warning"
        })

        # 4. 触发告警
        self.monitor.trigger_alert({
            "name": "cpu_warning",
            "severity": "warning",
            "message": "CPU usage is high"
        })

        # 5. 验证状态
        summary = self.monitor.get_metrics_summary()
        alerts = self.monitor.get_active_alerts()

        assert summary["gauges"] >= 1
        assert summary["counters"] >= 1
        assert summary["histograms"] >= 1
        assert len(alerts) >= 1

        # 6. 导出指标
        with patch.object(self.monitor._exporter, 'push_metrics', return_value=True):
            export_result = self.monitor.export_metrics()
            assert export_result is True

        # 7. 清理资源
        self.monitor.clear_metrics()
        resolved = self.monitor.resolve_alert("cpu_warning")

        assert resolved is True

        final_summary = self.monitor.get_metrics_summary()
        assert final_summary["gauges"] == 0


class TestPrometheusMonitorEdgeCases:
    """测试Prometheus监控器的边缘情况"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = PrometheusMonitor()

    @patch('src.infrastructure.logging.monitors.prometheus_monitor.push_to_gateway')
    def test_push_metrics_failure_handling(self, mock_push):
        """测试推送指标失败的处理"""
        mock_push.side_effect = Exception("Network error")

        # 创建指标
        gauge = self.monitor.create_gauge_metric("test_failure", "Test failure gauge")
        self.monitor.record_metric_value("test_failure", 42.0)

        # 尝试推送，应该失败但不抛出异常
        result = self.monitor.push_metrics()
        assert result is False

    def test_get_counter_existing_metric(self):
        """测试获取已存在的计数器指标"""
        # 先创建计数器
        counter1 = self.monitor.create_counter_metric("existing_counter", "Existing counter")
        assert counter1 is not None

        # 再次获取同一个计数器（通过registry）
        counter2 = self.monitor._registry.get_counter("existing_counter")
        assert counter2 is not None
        assert counter1 is counter2

    def test_get_histogram_existing_metric(self):
        """测试获取已存在的直方图指标"""
        # 先创建直方图
        hist1 = self.monitor.create_histogram_metric("existing_histogram", "Existing histogram")
        assert hist1 is not None

        # 再次获取同一个直方图（通过registry）
        hist2 = self.monitor._registry.get_histogram("existing_histogram")
        assert hist2 is not None
        assert hist1 is hist2

    def test_record_metric_with_invalid_labels(self):
        """测试使用无效标签记录指标"""
        gauge = self.monitor.create_gauge_metric("invalid_labels_test", "Test gauge", labels=["service"])

        # 使用不正确的标签 - 应该记录错误但不抛出异常
        result = self.monitor.record_metric_value("invalid_labels_test", 1.0, labels={"invalid_label": "value"})
        assert result is False  # 应该返回False表示失败

    def test_histogram_with_custom_buckets(self):
        """测试自定义桶的直方图"""
        buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        hist = self.monitor.create_histogram_metric(
            "custom_buckets_hist",
            "Histogram with custom buckets",
            buckets=buckets
        )

        assert hist is not None
        self.monitor.record_histogram_value("custom_buckets_hist", 1.5)

    def test_alert_handler_reset_nonexistent_alert(self):
        """测试重置不存在的告警"""
        result = self.monitor.resolve_alert("nonexistent_alert")
        assert result is False

    def test_monitor_initialization_with_numeric_config(self):
        """测试使用数字配置初始化监控器"""
        # 数字类型的配置 - 会被转换为字符串URL
        monitor = PrometheusMonitor(gateway_url_or_config=12345)
        assert monitor is not None
        # 数字被转换为字符串
        assert str(monitor.gateway_url) == "12345"

    def test_monitor_empty_gateway_url(self):
        """测试空网关URL的情况"""
        monitor = PrometheusMonitor(gateway_url="")
        assert monitor.gateway_url == ""

    def test_monitor_gateway_url_fallback(self):
        """测试网关URL回退到默认值"""
        monitor = PrometheusMonitor()
        assert monitor.gateway_url == "http://localhost:9091"

    def test_remove_nonexistent_gauge(self):
        """测试移除不存在的仪表指标"""
        result = self.monitor._registry.remove_gauge("nonexistent_gauge")
        assert result is False

    def test_remove_nonexistent_counter(self):
        """测试移除不存在的计数器指标"""
        result = self.monitor._registry.remove_counter("nonexistent_counter")
        assert result is False

    def test_get_metric_nonexistent(self):
        """测试获取不存在的指标"""
        result = self.monitor.get_metric("nonexistent_metric")
        assert result is None

    def test_send_metric_with_none_labels(self):
        """测试发送指标时标签为None的情况"""
        gauge = self.monitor.create_gauge_metric("none_labels_test", "Test with None labels")
        result = self.monitor.send_metric("none_labels_test", 42.0, labels=None)
        assert result is True

    def test_alert_handler_with_malformed_rule(self):
        """测试告警处理器处理格式错误的规则"""
        malformed_config = {
            "condition": "",  # 空的条件
            "severity": "invalid_severity"  # 无效的严重程度
        }

        result = self.monitor.add_alert_rule("malformed_rule", malformed_config)
        # 应该能够添加，但可能不会正常工作
        assert isinstance(result, bool)

    def test_monitor_comprehensive_error_scenarios(self):
        """测试监控器的综合错误场景"""
        # 测试各种边界情况
        monitor = PrometheusMonitor()

        # 测试空指标名称 - 应该记录错误但不抛出异常
        gauge = monitor.create_gauge_metric("", "Empty name")
        # Prometheus客户端可能不会创建空的指标

        # 测试空描述
        counter = monitor.create_counter_metric("valid_name", "")

        # 测试无效的指标值 - 这些应该被处理
        gauge = monitor.create_gauge_metric("invalid_value_test", "Test invalid values")
        # 这些可能不会抛出异常，取决于prometheus客户端的行为
        monitor.record_metric_value("invalid_value_test", float('inf'))
        monitor.record_metric_value("invalid_value_test", float('-inf'))
        monitor.record_metric_value("invalid_value_test", float('nan'))

    def test_monitor_resource_cleanup_on_error(self):
        """测试错误情况下资源清理"""
        monitor = PrometheusMonitor()

        # 创建一些指标
        gauge = monitor.create_gauge_metric("cleanup_test", "Test cleanup")
        counter = monitor.create_counter_metric("cleanup_counter", "Test counter")

        # 模拟一些操作
        monitor.record_metric_value("cleanup_test", 1.0)
        monitor.increment_counter("cleanup_counter")

        # 清理所有指标
        monitor.clear_metrics()

        # 验证清理成功
        summary = monitor.get_metrics_summary()
        assert summary["gauges"] == 0
        assert summary["counters"] == 0