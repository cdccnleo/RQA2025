#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分布式监控深度测试 - Week 2 Day 1
针对: monitors/distributed_monitoring.py (387行未覆盖，高价值目标)
目标: 从21.82%提升至50%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


# =====================================================
# 1. 枚举和数据类测试
# =====================================================

class TestDistributedMonitoringEnums:
    """测试枚举类"""
    
    def test_metric_type_enum(self):
        """测试MetricType枚举"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MetricType
        
        assert hasattr(MetricType, 'GAUGE')
        assert hasattr(MetricType, 'COUNTER')
        assert hasattr(MetricType, 'HISTOGRAM')
        assert hasattr(MetricType, 'SUMMARY')
    
    def test_alert_severity_enum(self):
        """测试AlertSeverity枚举"""
        from src.infrastructure.logging.monitors.distributed_monitoring import AlertSeverity
        
        assert hasattr(AlertSeverity, 'INFO')
        assert hasattr(AlertSeverity, 'WARNING')
        assert hasattr(AlertSeverity, 'ERROR')
        assert hasattr(AlertSeverity, 'CRITICAL')


class TestMetricDataClass:
    """测试Metric数据类"""
    
    def test_metric_creation(self):
        """测试创建指标"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Metric, MetricType
        
        metric = Metric(
            name='cpu_usage',
            value=75.5,
            metric_type=MetricType.GAUGE
        )
        assert metric.name == 'cpu_usage'
        assert metric.value == 75.5
        assert metric.metric_type == MetricType.GAUGE
    
    def test_metric_with_labels(self):
        """测试带标签的指标"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Metric, MetricType
        
        metric = Metric(
            name='request_count',
            value=100,
            metric_type=MetricType.COUNTER,
            labels={'endpoint': '/api/health', 'method': 'GET'}
        )
        assert metric.labels['endpoint'] == '/api/health'
        assert metric.labels['method'] == 'GET'
    
    def test_metric_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Metric, MetricType
        
        metric = Metric(name='test', value=10, metric_type=MetricType.GAUGE)
        data = metric.to_dict()
        assert isinstance(data, dict)
        assert data['name'] == 'test'
        assert data['value'] == 10
    
    def test_metric_from_dict(self):
        """测试从字典创建"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Metric, MetricType
        
        data = {
            'name': 'memory_usage',
            'value': 80,
            'metric_type': MetricType.GAUGE,
            'labels': {},
            'timestamp': 1234567890.0
        }
        metric = Metric.from_dict(data)
        assert metric.name == 'memory_usage'
        assert metric.value == 80
    
    def test_metric_type_property(self):
        """测试metric_type属性"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Metric, MetricType
        
        metric = Metric(name='test', value=1, metric_type=MetricType.COUNTER)
        assert metric.type == MetricType.COUNTER
        
        metric.type = MetricType.GAUGE
        assert metric.metric_type == MetricType.GAUGE


class TestAlertDataClass:
    """测试Alert数据类"""
    
    def test_alert_creation(self):
        """测试创建告警"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Alert, AlertSeverity
        
        alert = Alert(
            message='CPU usage high',
            severity=AlertSeverity.WARNING
        )
        assert alert.message == 'CPU usage high'
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == 'active'
    
    def test_alert_with_labels(self):
        """测试带标签的告警"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Alert, AlertSeverity
        
        alert = Alert(
            message='Disk space low',
            severity=AlertSeverity.ERROR,
            labels={'server': 'web-1', 'disk': '/dev/sda1'}
        )
        assert alert.labels['server'] == 'web-1'
        assert alert.labels['disk'] == '/dev/sda1'
    
    def test_alert_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Alert, AlertSeverity
        
        alert = Alert(message='Test alert', severity=AlertSeverity.INFO)
        data = alert.to_dict()
        assert isinstance(data, dict)
        assert data['message'] == 'Test alert'
    
    def test_alert_from_dict(self):
        """测试从字典创建"""
        from src.infrastructure.logging.monitors.distributed_monitoring import Alert, AlertSeverity
        
        data = {
            'message': 'Memory leak detected',
            'severity': AlertSeverity.CRITICAL,
            'labels': {},
            'annotations': {},
            'timestamp': datetime.now(),
            'status': 'active'
        }
        alert = Alert.from_dict(data)
        assert alert.message == 'Memory leak detected'


# =====================================================
# 2. MockPrometheusClient测试
# =====================================================

class TestMockPrometheusClient:
    """测试Mock Prometheus客户端"""
    
    def test_mock_client_initialization(self):
        """测试初始化"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient
        
        client = MockPrometheusClient('http://localhost:9090')
        assert client.instance_url == 'http://localhost:9090'
    
    def test_record_metric(self):
        """测试记录指标"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, MetricType
        
        client = MockPrometheusClient('http://localhost:9090')
        metric = Metric(name='test_metric', value=100, metric_type=MetricType.COUNTER)
        
        result = client.record_metric(metric)
        assert result is True
    
    def test_query_metrics_by_name(self):
        """测试按名称查询指标"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, MetricType
        
        client = MockPrometheusClient('http://localhost:9090')
        
        # 记录几个指标
        client.record_metric(Metric(name='cpu', value=50, metric_type=MetricType.GAUGE))
        client.record_metric(Metric(name='cpu', value=60, metric_type=MetricType.GAUGE))
        client.record_metric(Metric(name='memory', value=80, metric_type=MetricType.GAUGE))
        
        # 查询cpu指标
        cpu_metrics = client.query_metrics(name='cpu')
        assert len(cpu_metrics) == 2
    
    def test_query_metrics_by_labels(self):
        """测试按标签查询指标"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, MetricType
        
        client = MockPrometheusClient('http://localhost:9090')
        
        client.record_metric(Metric(
            name='requests',
            value=100,
            metric_type=MetricType.COUNTER,
            labels={'endpoint': '/api/users'}
        ))
        client.record_metric(Metric(
            name='requests',
            value=50,
            metric_type=MetricType.COUNTER,
            labels={'endpoint': '/api/orders'}
        ))
        
        # 按标签查询
        metrics = client.query_metrics(labels={'endpoint': '/api/users'})
        assert len(metrics) >= 1
    
    def test_query_metrics_by_time_range(self):
        """测试按时间范围查询"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, MetricType
        import time
        
        client = MockPrometheusClient('http://localhost:9090')
        
        start_time = time.time()
        client.record_metric(Metric(name='test', value=1, metric_type=MetricType.GAUGE))
        time.sleep(0.1)
        mid_time = time.time()
        time.sleep(0.1)
        client.record_metric(Metric(name='test', value=2, metric_type=MetricType.GAUGE))
        
        # 查询时间范围
        metrics = client.query_metrics(name='test', start_time=mid_time)
        assert len(metrics) >= 1
    
    def test_create_alert(self):
        """测试创建告警"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Alert, AlertSeverity
        
        client = MockPrometheusClient('http://localhost:9090')
        alert = Alert(message='Test alert', severity=AlertSeverity.WARNING)
        
        result = client.create_alert(alert)
        assert result is True
    
    def test_query_alerts(self):
        """测试查询告警"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Alert, AlertSeverity
        
        client = MockPrometheusClient('http://localhost:9090')
        
        # 创建告警
        client.create_alert(Alert(message='Alert 1', severity=AlertSeverity.INFO))
        client.create_alert(Alert(message='Alert 2', severity=AlertSeverity.WARNING))
        
        # 查询告警
        if hasattr(client, 'query_alerts'):
            alerts = client.query_alerts()
            assert len(alerts) >= 2
    
    def test_record_metric_with_error(self):
        """测试记录指标时发生错误"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, MetricType
        
        client = MockPrometheusClient('http://localhost:9090')
        
        # 设置会抛出异常的Mock client
        client._client = Mock(side_effect=Exception("Connection error"))
        
        metric = Metric(name='test', value=1, metric_type=MetricType.GAUGE)
        result = client.record_metric(metric)
        assert result is False
    
    def test_create_alert_with_error(self):
        """测试创建告警时发生错误"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Alert, AlertSeverity
        
        client = MockPrometheusClient('http://localhost:9090')
        client._client = Mock(side_effect=Exception("Connection error"))
        
        alert = Alert(message='Test', severity=AlertSeverity.INFO)
        result = client.create_alert(alert)
        assert result is False


# =====================================================
# 3. 分布式监控系统集成测试
# =====================================================

class TestDistributedMonitoringIntegration:
    """测试分布式监控系统集成"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        from src.infrastructure.logging.monitors.distributed_monitoring import MockPrometheusClient, Metric, Alert
        from src.infrastructure.logging.monitors.distributed_monitoring import MetricType, AlertSeverity
        
        # 创建客户端
        client = MockPrometheusClient('http://localhost:9090')
        
        # 记录多个指标
        for i in range(5):
            metric = Metric(
                name='request_duration',
                value=i * 0.1,
                metric_type=MetricType.HISTOGRAM,
                labels={'service': 'api'}
            )
            client.record_metric(metric)
        
        # 查询指标
        metrics = client.query_metrics(name='request_duration')
        assert len(metrics) == 5
        
        # 创建告警
        alert = Alert(
            message='High latency detected',
            severity=AlertSeverity.WARNING,
            labels={'service': 'api'}
        )
        client.create_alert(alert)

