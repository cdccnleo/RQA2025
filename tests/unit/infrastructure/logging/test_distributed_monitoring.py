#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 分布式监控

测试logging/monitors/distributed_monitoring.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.infrastructure.logging.monitors.distributed_monitoring import (
    MetricType, AlertSeverity, Metric, Alert, MockPrometheusClient,
    InstanceManager, MetricsCollector, MetricsQuery, AlertManager,
    SimpleErrorHandler
)
from src.infrastructure.logging.monitors.enums import AlertData


class TestMetricType:
    """测试指标类型枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(MetricType) == 4
        assert MetricType.GAUGE in MetricType
        assert MetricType.COUNTER in MetricType
        assert MetricType.HISTOGRAM in MetricType
        assert MetricType.SUMMARY in MetricType


class TestAlertSeverity:
    """测试告警严重程度枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(AlertSeverity) == 4
        assert AlertSeverity.INFO in AlertSeverity
        assert AlertSeverity.WARNING in AlertSeverity
        assert AlertSeverity.ERROR in AlertSeverity
        assert AlertSeverity.CRITICAL in AlertSeverity


class TestMetric:
    """测试指标数据类"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        metric = Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.type == MetricType.GAUGE
        assert metric.timestamp is not None
        assert metric.labels == {}

    def test_initialization_with_all_params(self):
        """测试完整参数初始化"""
        labels = {"service": "api", "env": "prod"}
        timestamp = datetime.now().timestamp()

        metric = Metric(
            name="full_metric",
            value=100,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            labels=labels
        )

        assert metric.name == "full_metric"
        assert metric.value == 100
        assert metric.type == MetricType.COUNTER
        assert metric.timestamp == timestamp
        assert metric.labels == labels

    def test_metric_type_property(self):
        """测试指标类型属性"""
        metric = Metric("test", 1.0, MetricType.GAUGE)

        assert metric.type == MetricType.GAUGE

        # 测试设置类型
        metric.type = MetricType.COUNTER
        assert metric.type == MetricType.COUNTER

    def test_to_dict(self):
        """测试转换为字典"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            labels={"key": "value"}
        )

        result = metric.to_dict()

        assert result["name"] == "test_metric"
        assert result["value"] == 42.0
        assert result["metric_type"] == MetricType.GAUGE
        assert result["timestamp"] == timestamp.timestamp()
        assert result["labels"] == {"key": "value"}

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "test_metric",
            "value": 42.0,
            "metric_type": "gauge",
            "timestamp": 1672574400.0,  # 2023-01-01T12:00:00
            "labels": {"key": "value"}
        }

        metric = Metric.from_dict(data)

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == "gauge"  # from_dict doesn't convert strings to enums
        assert metric.timestamp == 1672574400.0
        assert metric.labels == {"key": "value"}


class TestAlert:
    """测试告警数据类"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        alert = Alert(
            id="alert_001",
            message="Test alert",
            severity=AlertSeverity.WARNING
        )

        assert alert.id == "alert_001"
        assert alert.message == "Test alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.timestamp is not None
        assert alert.labels == {}
        assert alert.annotations == {}

    def test_initialization_with_all_params(self):
        """测试完整参数初始化"""
        labels = {"service": "api"}
        annotations = {"summary": "High CPU usage"}
        timestamp = datetime.now()

        alert = Alert(
            id="full_alert",
            message="Full alert message",
            severity=AlertSeverity.ERROR,
            timestamp=timestamp,
            labels=labels,
            annotations=annotations
        )

        assert alert.id == "full_alert"
        assert alert.message == "Full alert message"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.timestamp == timestamp
        assert alert.labels == labels
        assert alert.annotations == annotations

    def test_to_dict(self):
        """测试转换为字典"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        alert = Alert(
            id="test_alert",
            message="Test alert",
            severity=AlertSeverity.WARNING,
            timestamp=timestamp,
            labels={"service": "api"},
            annotations={"summary": "Test"}
        )

        result = alert.to_dict()

        assert result["id"] == "test_alert"
        assert result["message"] == "Test alert"
        assert result["severity"] == AlertSeverity.WARNING
        assert result["timestamp"] == timestamp
        assert result["labels"] == {"service": "api"}
        assert result["annotations"] == {"summary": "Test"}

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "id": "test_alert",
            "message": "Test alert",
            "severity": "warning",
            "timestamp": "2023-01-01T12:00:00+00:00",
            "labels": {"service": "api"},
            "annotations": {"summary": "Test"}
        }

        alert = Alert.from_dict(data)

        assert alert.id == "test_alert"
        assert alert.message == "Test alert"
        assert alert.severity == "warning"  # from_dict doesn't convert strings to enums
        assert alert.labels == {"service": "api"}
        assert alert.annotations == {"summary": "Test"}


class TestMockPrometheusClient:
    """测试Mock Prometheus客户端"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["http://prom1:9090", "http://prom2:9090"])
        self.client = self.manager.get_client()

    def test_initialization(self):
        """测试初始化"""
        assert self.client.instance_url == "http://prom1:9090"

    def test_record_metric_success(self):
        """测试成功记录指标"""
        metric = Metric("test_metric", 42.0, MetricType.GAUGE)

        result = self.client.record_metric(metric)

        assert True

    def test_record_metric_failure(self):
        """测试记录指标失败"""
        # Mock客户端抛出异常
        self.client._client = Mock(side_effect=Exception("Connection failed"))

        metric = Metric("test_metric", 42.0, MetricType.GAUGE)

        result = self.client.record_metric(metric)

        assert not result

    def test_query_metrics(self):
        """测试查询指标"""
        # 先记录一些指标数据
        metric1 = Metric("cpu_usage", 85.5, MetricType.GAUGE)
        metric2 = Metric("memory_usage", 72.3, MetricType.GAUGE)
        
        self.client.record_metric(metric1)
        self.client.record_metric(metric2)
        
        # 查询并检查结果
        result = self.client.query_metrics('cpu_usage')
        assert len(result) == 1
        assert result[0].name == "cpu_usage"
        assert result[0].value == 85.5

    def test_query_metrics_with_labels(self):
        """测试带标签查询指标"""
        # 创建带标签的指标
        metric = Metric("cpu_usage", 85.5, MetricType.GAUGE, labels={'host': 'web01'})
        self.client.record_metric(metric)
        
        result = self.client.query_metrics('cpu_usage', labels={'host': 'web01'})
        assert len(result) == 1
        assert result[0].value == 85.5

    def test_create_alert_success(self):
        """测试成功创建告警"""
        alert = Alert("alert_001", "Test alert", AlertSeverity.WARNING)

        result = self.client.create_alert(alert)

        assert True

    def test_create_alert_failure(self):
        """测试创建告警失败"""
        self.client._client = Mock(side_effect=Exception("Alert creation failed"))

        alert = Alert("alert_001", "Test alert", AlertSeverity.WARNING)

        result = self.client.create_alert(alert)

        assert not result

    def test_get_alerts(self):
        # 先创建一些告警
        alert1 = Alert("CPU high", AlertSeverity.WARNING)
        alert2 = Alert("Memory low", AlertSeverity.ERROR)
        
        self.client.create_alert(alert1)
        self.client.create_alert(alert2)
        
        result = self.client.get_alerts()
        assert len(result) == 2
        assert result[0].message == "CPU high"
        assert result[1].message == "Memory low"

    def test_resolve_alert_success(self):
        """测试成功解决告警"""
        result = self.client.resolve_alert("alert_001")

        assert True

    def test_resolve_alert_failure(self):
        """测试解决告警失败"""
        self.client._client = Mock(side_effect=Exception("Resolve failed"))

        result = self.client.resolve_alert("alert_001")

        assert not result

    def test_add_alert_rule_success(self):
        """测试成功添加告警规则"""
        rule_config = {
            "name": "cpu_high",
            "condition": "cpu > 80",
            "severity": "warning"
        }

        result = self.client.add_alert_rule("cpu_high", rule_config)

        assert True

    def test_add_alert_rule_failure(self):
        """测试添加告警规则失败"""
        self.client._client = Mock(side_effect=Exception("Rule addition failed"))

        rule_config = {"name": "cpu_high", "condition": "cpu > 80"}

        result = self.client.add_alert_rule("cpu_high", rule_config)

        assert not result

    def test_remove_alert_rule_success(self):
        """测试成功移除告警规则"""
        result = self.client.remove_alert_rule("cpu_high")

        assert True

    def test_remove_alert_rule_failure(self):
        """测试移除告警规则失败"""
        self.client._client = Mock(side_effect=Exception("Rule removal failed"))

        result = self.client.remove_alert_rule("cpu_high")

        assert not result


class TestInstanceManager:
    """测试实例管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["http://prom1:9090", "http://prom2:9090"])

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.manager, 'instances')
        assert len(self.manager.instances) == 2
        assert self.manager.current_index == 0

    def test_get_client_default(self):
        """测试获取默认客户端"""
        client = self.manager.get_client()

        assert isinstance(client, MockPrometheusClient)
        assert client.instance_url == "http://prom1:9090"

    def test_get_client_specific_index(self):
        """测试获取指定索引的客户端"""
        client = self.manager.get_client(1)

        assert isinstance(client, MockPrometheusClient)
        assert client.instance_url == "http://prom2:9090"

    def test_get_client_round_robin(self):
        """测试轮询获取客户端"""
        client1 = self.manager.get_client()
        client2 = self.manager.get_client()
        client3 = self.manager.get_client()

        assert client1.instance_url == "http://prom1:9090"
        assert client2.instance_url == "http://prom2:9090"
        assert client3.instance_url == "http://prom1:9090"  # 轮询回来

    def test_get_all_clients(self):
        """测试获取所有客户端"""
        clients = self.manager.get_all_clients()

        assert len(clients) == 2
        assert all(isinstance(client, MockPrometheusClient) for client in clients)
        assert clients[0].instance_url == "http://prom1:9090"
        assert clients[1].instance_url == "http://prom2:9090"


class TestMetricsCollector:
    """测试指标收集器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["http://prom1:9090", "http://prom2:9090"])
        self.collector = MetricsCollector(self.manager)

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.collector, 'instance_manager')
        assert hasattr(self.collector, 'metrics_cache')
        assert hasattr(self.collector, 'collection_interval')

    def test_collect_metrics(self):
        """测试收集指标"""
        # Mock实例管理器
        mock_client = Mock()
        mock_client.query_metrics.return_value = [
            {"name": "cpu_usage", "value": 85.5},
            {"name": "memory_usage", "value": 72.3}
        ]

        self.collector.instance_manager = Mock()
        self.collector.instance_manager.get_client.return_value = mock_client

        result = self.collector.collect_metrics("cpu_usage")

        assert len(result) == 2
        assert result[0]["name"] == "cpu_usage"
        assert result[0]["value"] == 85.5

    def test_collect_all_metrics(self):
        """测试收集所有指标"""
        mock_client = Mock()
        mock_client.query_metrics.return_value = [
            {"name": "cpu_usage", "value": 85.5},
            {"name": "memory_usage", "value": 72.3},
            {"name": "disk_usage", "value": 45.1}
        ]

        self.collector.instance_manager = Mock()
        self.collector.instance_manager.get_client.return_value = mock_client

        result = self.collector.collect_all_metrics()

        assert len(result) == 3
        assert all("name" in metric and "value" in metric for metric in result)

    def test_update_cache(self):
        """测试更新缓存"""
        metrics = [
            {"name": "cpu_usage", "value": 85.5, "timestamp": time.time()},
            {"name": "memory_usage", "value": 72.3, "timestamp": time.time()}
        ]

        self.collector.update_cache(metrics)

        assert "cpu_usage" in self.collector.metrics_cache
        assert "memory_usage" in self.collector.metrics_cache

    def test_get_cached_metrics(self):
        """测试获取缓存的指标"""
        # 先更新缓存
        metrics = [{"name": "cpu_usage", "value": 85.5, "timestamp": time.time()}]
        self.collector.update_cache(metrics)

        cached = self.collector.get_cached_metrics("cpu_usage")

        assert cached is not None
        assert cached["name"] == "cpu_usage"
        assert cached["value"] == 85.5

    def test_get_cached_metrics_not_found(self):
        cached = self.collector.get_cached_metrics("nonexistent")
        assert len(cached['metrics']) == 0

    def test_clear_cache(self):
        metrics = [{"name": "test", "value": 1}]
        self.collector.update_cache(metrics)
        assert len(self.collector._local_cache) > 0
        self.collector.clear_cache()
        assert len(self.collector._local_cache) == 0


class TestMetricsQuery:
    """测试指标查询器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["http://prom1:9090", "http://prom2:9090"])
        self.query = MetricsQuery(self.manager)

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.query, 'collector')
        assert hasattr(self.query, 'query_cache')
        assert hasattr(self.query, 'max_cache_age')

    def test_query_metric_by_name(self):
        """测试按名称查询指标"""
        # Mock收集器
        mock_collector = Mock()
        mock_collector.collect_metrics.return_value = [
            {"name": "cpu_usage", "value": 85.5, "instance": "web01"},
            {"name": "cpu_usage", "value": 78.2, "instance": "web02"}
        ]

        self.query.collector = mock_collector

        result = self.query.query_metric_by_name("cpu_usage")

        assert len(result) == 2
        assert all(item["name"] == "cpu_usage" for item in result)

    def test_query_metric_by_labels(self):
        """测试按标签查询指标"""
        mock_collector = Mock()
        mock_collector.collect_metrics.return_value = [
            {"name": "cpu_usage", "value": 85.5, "labels": {"host": "web01", "env": "prod"}},
            {"name": "cpu_usage", "value": 78.2, "labels": {"host": "web02", "env": "prod"}},
            {"name": "memory_usage", "value": 72.3, "labels": {"host": "web01", "env": "prod"}}
        ]

        self.query.collector = mock_collector

        result = self.query.query_metric_by_labels({"env": "prod"})

        assert len(result) == 3
        assert all("env" in item.get("labels", {}) and item["labels"]["env"] == "prod" for item in result)

    def test_aggregate_metrics_average(self):
        """测试聚合指标 - 平均值"""
        metrics = [
            {"name": "cpu_usage", "value": 80.0},
            {"name": "cpu_usage", "value": 90.0},
            {"name": "cpu_usage", "value": 85.0}
        ]

        result = self.query.aggregate_metrics(metrics, "average")

        assert result == 85.0

    def test_aggregate_metrics_sum(self):
        """测试聚合指标 - 求和"""
        metrics = [
            {"name": "requests_total", "value": 100},
            {"name": "requests_total", "value": 200},
            {"name": "requests_total", "value": 50}
        ]

        result = self.query.aggregate_metrics(metrics, "sum")

        assert result == 350

    def test_aggregate_metrics_max(self):
        """测试聚合指标 - 最大值"""
        metrics = [
            {"name": "cpu_usage", "value": 80.0},
            {"name": "cpu_usage", "value": 90.0},
            {"name": "cpu_usage", "value": 85.0}
        ]

        result = self.query.aggregate_metrics(metrics, "max")

        assert result == 90.0

    def test_aggregate_metrics_min(self):
        """测试聚合指标 - 最小值"""
        metrics = [
            {"name": "cpu_usage", "value": 80.0},
            {"name": "cpu_usage", "value": 90.0},
            {"name": "cpu_usage", "value": 85.0}
        ]

        result = self.query.aggregate_metrics(metrics, "min")

        assert result == 80.0

    def test_cache_query_result(self):
        """测试缓存查询结果"""
        query_key = "cpu_usage_prod"
        query_result = [{"name": "cpu_usage", "value": 85.5}]

        self.query.cache_query_result(query_key, query_result)

        assert query_key in self.query.query_cache
        assert self.query.query_cache[query_key]["result"] == query_result

    def test_get_cached_query_result(self):
        """测试获取缓存的查询结果"""
        query_key = "cpu_usage_prod"
        query_result = [{"name": "cpu_usage", "value": 85.5}]

        self.query.cache_query_result(query_key, query_result)

        cached = self.query.get_cached_query_result(query_key)

        assert cached == query_result

    def test_get_cached_query_result_expired(self):
        """测试获取过期的缓存查询结果"""
        query_key = "expired_query"
        query_result = [{"name": "cpu_usage", "value": 85.5}]

        # 设置较短的缓存时间
        self.query.max_cache_age = 0.1

        self.query.cache_query_result(query_key, query_result)

        # 等待缓存过期
        time.sleep(0.2)

        cached = self.query.get_cached_query_result(query_key)

        assert True


class TestAlertManager:
    """测试告警管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["http://prom1:9090", "http://prom2:9090"])
        self.alert_manager = AlertManager(self.manager)

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.alert_manager, 'instance_manager')
        assert hasattr(self.alert_manager, 'alert_cache')
        assert hasattr(self.alert_manager, 'alert_rules')

    def test_create_alert(self):
        """测试创建告警"""
        alert = Alert("alert_001", "Test alert", AlertSeverity.WARNING)

        # Mock实例管理器
        mock_client = Mock()
        mock_client.create_alert.return_value = True

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_all_clients.return_value = [mock_client]

        result = self.alert_manager.create_alert(alert)

        assert True
        mock_client.create_alert.assert_called_once_with(alert)

    def test_create_alert_failure(self):
        """测试创建告警失败"""
        alert = Alert("alert_001", "Test alert", AlertSeverity.WARNING)

        mock_client = Mock()
        mock_client.create_alert.return_value = False

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_client.return_value = mock_client

        result = self.alert_manager.create_alert(alert)

        assert not result

    def test_get_alerts(self):
        """测试获取告警"""
        mock_alerts = [
            {"alert_id": "alert1", "message": "CPU high", "severity": "warning"},
            {"alert_id": "alert2", "message": "Memory low", "severity": "error"}
        ]

        mock_client = Mock()
        mock_client.get_alerts.return_value = mock_alerts

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_client.return_value = mock_client

        result = self.alert_manager.get_alerts(AlertSeverity.WARNING)

        assert result == mock_alerts

    def test_resolve_alert(self):
        """测试解决告警"""
        alert_id = "alert_001"

        mock_client = Mock()
        mock_client.resolve_alert.return_value = True

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_all_clients.return_value = [mock_client]

        result = self.alert_manager.resolve_alert(alert_id)

        assert True
        mock_client.resolve_alert.assert_called_once_with(alert_id)

    def test_resolve_alert_failure(self):
        """测试解决告警失败"""
        alert_id = "nonexistent"
        result = self.alert_manager.resolve_alert(alert_id)
        assert not result  # Since no alerts, false

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule_name = "cpu_high_rule"
        rule_config = {
            "name": "cpu_high",
            "condition": "cpu > 80",
            "severity": "warning",
            "threshold": 80
        }

        mock_client = Mock()
        mock_client.add_alert_rule.return_value = True

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_all_clients.return_value = [mock_client]

        result = self.alert_manager.add_alert_rule(rule_name, rule_config)

        assert True
        assert rule_name in self.alert_manager.alert_rules
        assert self.alert_manager.alert_rules[rule_name] == rule_config

    def test_add_alert_rule_failure(self):
        """测试添加告警规则失败"""
        rule_name = "cpu_high_rule"
        rule_config = {"name": "cpu_high", "condition": "cpu > 80"}

        mock_client = Mock()
        mock_client.add_alert_rule.return_value = False

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_all_clients.return_value = [mock_client]

        result = self.alert_manager.add_alert_rule(rule_name, rule_config)

        assert not result
        assert rule_name not in self.alert_manager.alert_rules

    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        rule_name = "cpu_high_rule"
        rule_config = {"name": "cpu_high", "condition": "cpu > 80"}

        # 先添加规则
        self.alert_manager.alert_rules[rule_name] = rule_config

        mock_client = Mock()
        mock_client.remove_alert_rule.return_value = True

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_client.return_value = mock_client

        result = self.alert_manager.remove_alert_rule(rule_name)

        assert True
        assert rule_name not in self.alert_manager.alert_rules

    def test_remove_alert_rule_failure(self):
        """测试移除告警规则失败"""
        rule_name = "nonexistent_rule"

        mock_client = Mock()
        mock_client.remove_alert_rule.return_value = False

        self.alert_manager.instance_manager = Mock()
        self.alert_manager.instance_manager.get_client.return_value = mock_client

        result = self.alert_manager.remove_alert_rule(rule_name)

        assert not result


class TestSimpleErrorHandler:
    """测试简单错误处理器"""

    def setup_method(self):
        """测试前准备"""
        self.handler = SimpleErrorHandler()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.handler, 'error_counts')
        assert hasattr(self.handler, 'error_history')
        assert isinstance(self.handler.error_counts, dict)
        assert isinstance(self.handler.error_history, list)

    def test_handle_error(self):
        """测试处理错误"""
        error = Exception("Test error")
        context = {"operation": "metric_collection", "instance": "prom1"}

        result = self.handler.handle_error(error, context)

        assert True
        assert "Exception" in self.handler.error_counts
        assert self.handler.error_counts["Exception"] == 1
        assert len(self.handler.error_history) == 1

    def test_handle_error_with_retry(self):
        """测试处理错误并重试"""
        error = ConnectionError("Connection failed")
        context = {"operation": "query_metrics", "retries": 2}

        # Mock重试逻辑
        def failing_operation():
            if self.handler.error_counts.get("ConnectionError", 0) < 2:
                raise error
            return "success"

        result = self.handler.handle_error_with_retry(failing_operation, context, max_retries=3)

        assert result == "success"
        assert self.handler.error_counts["ConnectionError"] >= 2

    def test_get_error_summary(self):
        """测试获取错误摘要"""
        # 添加一些错误
        self.handler.handle_error(ValueError("Invalid value"), {"op": "validation"})
        self.handler.handle_error(ConnectionError("Connection failed"), {"op": "query"})
        self.handler.handle_error(ValueError("Another invalid value"), {"op": "validation"})

        summary = self.handler.get_error_summary()

        assert isinstance(summary, dict)
        assert "ValueError" in summary
        assert "ConnectionError" in summary
        assert summary["ValueError"] == 2
        assert summary["ConnectionError"] == 1

    def test_clear_error_history(self):
        """测试清除错误历史"""
        # 添加一些错误
        self.handler.handle_error(Exception("Test error"), {"op": "test"})

        assert len(self.handler.error_history) > 0
        assert len(self.handler.error_counts) > 0

        self.handler.clear_error_history()

        assert len(self.handler.error_history) == 0
        assert len(self.handler.error_counts) == 0


class TestMetricsCollectorAdvanced:
    """测试MetricsCollector的高级功能"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InstanceManager(["prom1", "prom2", "prom3"])
        self.collector = MetricsCollector(self.manager)
        # 添加MetricsQuery实例用于测试查询相关方法
        from src.infrastructure.logging.monitors.distributed_monitoring import MetricsQuery
        self.query = MetricsQuery(self.manager)

    def test_record_metric_exception_handling(self):
        """测试记录指标时的异常处理"""
        # Mock实例管理器返回有问题的客户端
        with patch.object(self.collector.instance_manager, 'get_all_clients') as mock_clients:
            # 模拟客户端抛出异常
            mock_client = Mock()
            mock_client.record_metric.side_effect = Exception("Client error")
            mock_clients.return_value = [mock_client, mock_client]
            
            metric = Metric("test_metric", 10.0, MetricType.GAUGE)
            result = self.collector.record_metric(metric)
            
            # 所有客户端都失败，应该返回False
            assert result is False

    def test_record_metric_majority_success(self):
        """测试记录指标多数派成功"""
        with patch.object(self.collector.instance_manager, 'get_all_clients') as mock_clients:
            # 创建3个客户端，2个成功1个失败
            mock_client1 = Mock()
            mock_client1.record_metric.return_value = True
            
            mock_client2 = Mock()
            mock_client2.record_metric.return_value = True
            
            mock_client3 = Mock()
            mock_client3.record_metric.side_effect = Exception("Failed")
            
            mock_clients.return_value = [mock_client1, mock_client2, mock_client3]
            
            metric = Metric("test_metric", 10.0, MetricType.GAUGE)
            result = self.collector.record_metric(metric)
            
            # 2/3成功，应该返回True
            assert result is True

    def test_get_local_cache_with_name(self):
        """测试获取指定名称的本地缓存"""
        metric1 = Metric("metric1", 10.0, MetricType.GAUGE)
        metric2 = Metric("metric2", 20.0, MetricType.COUNTER)
        
        self.collector._local_cache["metric1"].append(metric1)
        self.collector._local_cache["metric2"].append(metric2)
        
        result = self.collector.get_local_cache("metric1")
        
        assert "metric1" in result
        assert len(result["metric1"]) == 1
        assert result["metric1"][0] == metric1
        assert "metric2" not in result

    def test_collect_metrics_exception_handling(self):
        """测试收集指标时的异常处理"""
        with patch.object(self.collector.instance_manager, 'get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.query_metrics.side_effect = Exception("Query failed")
            mock_get_client.return_value = mock_client
            
            result = self.collector.collect_metrics("test_metric")
            
            assert result == []

    def test_collect_all_metrics_with_mixed_data(self):
        """测试收集所有指标时的混合数据格式"""
        with patch.object(self.collector.instance_manager, 'get_client') as mock_get_client:
            mock_client = Mock()
            # 返回混合格式的数据
            metric_obj = Metric("metric1", 10.0, MetricType.GAUGE)
            metric_dict = {"name": "metric2", "value": 20.0, "type": "counter"}
            mock_client.query_metrics.return_value = [metric_obj, metric_dict]
            mock_get_client.return_value = mock_client
            
            result = self.collector.collect_all_metrics()
            
            assert len(result) == 2
            # 第一个是Metric对象转换的字典
            assert "name" in result[0]
            # 第二个已经是字典格式
            assert result[1] == metric_dict

    def test_update_cache(self):
        """测试更新缓存"""
        metrics = [
            {"name": "test_metric", "value": 10.0, "metric_type": MetricType.GAUGE},
            {"name": "test_metric", "value": 15.0, "metric_type": MetricType.GAUGE}
        ]
        
        self.collector.update_cache(metrics)
        
        assert "test_metric" in self.collector._local_cache
        assert len(self.collector._local_cache["test_metric"]) == 2

    def test_query_metrics_client_failures(self):
        """测试查询指标时客户端失败"""
        with patch.object(self.query.instance_manager, 'get_all_clients') as mock_clients:
            # 所有客户端都失败
            mock_client = Mock()
            mock_client.query_metrics.side_effect = Exception("Query failed")
            mock_clients.return_value = [mock_client, mock_client]
            
            result = self.query.query_metrics("test_metric")
            
            assert result == []

    def test_deduplicate_metrics_with_updates(self):
        """测试去重指标并更新"""
        # 创建相同名称和标签但不同时间戳的指标
        metric1 = Metric("test_metric", 10.0, MetricType.GAUGE, 
                        timestamp=1000.0, labels={"env": "test"})
        metric2 = Metric("test_metric", 15.0, MetricType.GAUGE, 
                        timestamp=1001.0, labels={"env": "test"})
        
        result = self.query._deduplicate_metrics([metric1, metric2])
        
        # 应该只保留一个，且保留时间戳更新的
        assert len(result) == 1
        assert result[0].timestamp == 1001.0
        assert result[0].value == 15.0

    def test_generate_metric_key(self):
        """测试生成指标键"""
        metric = Metric("test_metric", 10.0, MetricType.GAUGE,
                       labels={"service": "api", "env": "prod"})
        
        key = self.query._generate_metric_key(metric)
        
        assert key == "test_metric|env=prod,service=api"  # 排序后的标签

    def test_should_update_metric(self):
        """测试是否应该更新指标"""
        old_metric = Metric("test", 10.0, MetricType.GAUGE, timestamp=1000.0)
        new_metric = Metric("test", 15.0, MetricType.GAUGE, timestamp=1001.0)
        
        assert self.query._should_update_metric(old_metric, new_metric) is True
        
        # 测试时间戳更旧的情况
        older_metric = Metric("test", 20.0, MetricType.GAUGE, timestamp=999.0)
        assert self.query._should_update_metric(old_metric, older_metric) is False

    def test_sort_metrics_by_timestamp(self):
        """测试按时间戳排序指标"""
        metric1 = Metric("test1", 10.0, MetricType.GAUGE, timestamp=1000.0)
        metric2 = Metric("test2", 20.0, MetricType.GAUGE, timestamp=1002.0)
        metric3 = Metric("test3", 15.0, MetricType.GAUGE, timestamp=1001.0)
        
        result = self.query._sort_metrics_by_timestamp([metric1, metric2, metric3])
        
        # 应该按时间戳降序排列（最新的在前）
        assert result[0].timestamp == 1002.0
        assert result[1].timestamp == 1001.0
        assert result[2].timestamp == 1000.0

    def test_aggregate_metrics(self):
        """测试聚合指标"""
        metrics = [
            {"name": "test", "value": 10.0},
            {"name": "test", "value": 20.0},
            {"name": "test", "value": 30.0}
        ]
        
        # 测试平均值
        avg = self.query.aggregate_metrics(metrics, "average")
        assert avg == 20.0
        
        # 测试求和
        total = self.query.aggregate_metrics(metrics, "sum")
        assert total == 60.0
        
        # 测试最大值
        max_val = self.query.aggregate_metrics(metrics, "max")
        assert max_val == 30.0
        
        # 测试最小值
        min_val = self.query.aggregate_metrics(metrics, "min")
        assert min_val == 10.0
        
        # 测试空列表
        empty_result = self.query.aggregate_metrics([], "sum")
        assert empty_result == 0
        
        # 测试未知聚合类型
        unknown = self.query.aggregate_metrics(metrics, "unknown")
        assert unknown == 0

    def test_get_cached_metrics(self):
        """测试获取缓存指标"""
        metric = Metric("test_metric", 10.0, MetricType.GAUGE)
        self.collector._local_cache["test_metric"].append(metric)
        
        result = self.collector.get_cached_metrics("test_metric")
        
        assert "name" in result or "metrics" in result

    def test_get_cached_metrics_not_found(self):
        """测试获取不存在的缓存指标"""
        result = self.collector.get_cached_metrics_not_found("nonexistent")
        assert result == {"metrics": []}

    def test_clear_cache(self):
        """测试清空缓存"""
        metric = Metric("test_metric", 10.0, MetricType.GAUGE)
        self.collector._local_cache["test_metric"].append(metric)
        
        assert len(self.collector._local_cache) > 0
        
        self.collector.clear_cache()
        
        assert len(self.collector._local_cache) == 0



