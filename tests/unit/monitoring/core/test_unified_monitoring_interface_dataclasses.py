#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一监控系统接口dataclass测试
补充unified_monitoring_interface.py中dataclass的__post_init__方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    core_unified_monitoring_interface_module = importlib.import_module('src.monitoring.core.unified_monitoring_interface')
    Alert = getattr(core_unified_monitoring_interface_module, 'Alert', None)
    HealthCheck = getattr(core_unified_monitoring_interface_module, 'HealthCheck', None)
    PerformanceMetrics = getattr(core_unified_monitoring_interface_module, 'PerformanceMetrics', None)
    MonitoringConfig = getattr(core_unified_monitoring_interface_module, 'MonitoringConfig', None)
    AlertLevel = getattr(core_unified_monitoring_interface_module, 'AlertLevel', None)
    AlertStatus = getattr(core_unified_monitoring_interface_module, 'AlertStatus', None)
    HealthStatus = getattr(core_unified_monitoring_interface_module, 'HealthStatus', None)
    MonitorType = getattr(core_unified_monitoring_interface_module, 'MonitorType', None)
    MetricType = getattr(core_unified_monitoring_interface_module, 'MetricType', None)
    
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestAlertDataclass:
    """测试Alert dataclass的__post_init__方法"""

    def test_alert_tags_defaults_to_empty_list(self):
        """测试Alert的tags默认为空列表"""
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert alert.tags == []
        assert isinstance(alert.tags, list)

    def test_alert_tags_with_explicit_none(self):
        """测试Alert的tags显式设置为None时初始化为空列表"""
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=None
        )
        
        assert alert.tags == []
        assert isinstance(alert.tags, list)

    def test_alert_tags_with_custom_list(self):
        """测试Alert的tags可以设置为自定义列表"""
        custom_tags = ['tag1', 'tag2', 'tag3']
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=custom_tags
        )
        
        assert alert.tags == custom_tags
        assert len(alert.tags) == 3

    def test_alert_metadata_defaults_to_none(self):
        """测试Alert的metadata默认为None"""
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert alert.metadata is None

    def test_alert_metadata_with_custom_dict(self):
        """测试Alert的metadata可以设置为自定义字典"""
        custom_metadata = {'key1': 'value1', 'key2': 123}
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=custom_metadata
        )
        
        assert alert.metadata == custom_metadata


class TestHealthCheckDataclass:
    """测试HealthCheck dataclass的__post_init__方法"""

    def test_health_check_details_defaults_to_empty_dict(self):
        """测试HealthCheck的details默认为空字典"""
        health_check = HealthCheck(
            component='test_component',
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now()
        )
        
        assert health_check.details == {}
        assert isinstance(health_check.details, dict)

    def test_health_check_details_with_explicit_none(self):
        """测试HealthCheck的details显式设置为None时初始化为空字典"""
        health_check = HealthCheck(
            component='test_component',
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            details=None
        )
        
        assert health_check.details == {}
        assert isinstance(health_check.details, dict)

    def test_health_check_details_with_custom_dict(self):
        """测试HealthCheck的details可以设置为自定义字典"""
        custom_details = {'key1': 'value1', 'key2': 123}
        health_check = HealthCheck(
            component='test_component',
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            details=custom_details
        )
        
        assert health_check.details == custom_details

    def test_health_check_optional_fields(self):
        """测试HealthCheck的可选字段"""
        health_check = HealthCheck(
            component='test_component',
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            response_time=1.5,
            message='Test message'
        )
        
        assert health_check.response_time == 1.5
        assert health_check.message == 'Test message'
        assert health_check.details == {}


class TestPerformanceMetricsDataclass:
    """测试PerformanceMetrics dataclass"""

    def test_performance_metrics_creation(self):
        """测试PerformanceMetrics创建"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000}
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == 70.0
        assert metrics.network_io['bytes_sent'] == 1000

    def test_performance_metrics_optional_fields(self):
        """测试PerformanceMetrics的可选字段"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={'bytes_sent': 1000},
            response_time=0.5,
            throughput=100.0,
            error_rate=0.01,
            active_connections=10
        )
        
        assert metrics.response_time == 0.5
        assert metrics.throughput == 100.0
        assert metrics.error_rate == 0.01
        assert metrics.active_connections == 10

    def test_performance_metrics_optional_fields_default_none(self):
        """测试PerformanceMetrics的可选字段默认为None"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={}
        )
        
        assert metrics.response_time is None
        assert metrics.throughput is None
        assert metrics.error_rate is None
        assert metrics.active_connections is None


class TestMonitoringConfigDataclass:
    """测试MonitoringConfig dataclass的__post_init__方法"""

    def test_monitoring_config_alert_thresholds_defaults_to_empty_dict(self):
        """测试MonitoringConfig的alert_thresholds默认为空字典"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        assert config.alert_thresholds == {}
        assert isinstance(config.alert_thresholds, dict)

    def test_monitoring_config_alert_thresholds_with_explicit_none(self):
        """测试MonitoringConfig的alert_thresholds显式设置为None时初始化为空字典"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM,
            alert_thresholds=None
        )
        
        assert config.alert_thresholds == {}
        assert isinstance(config.alert_thresholds, dict)

    def test_monitoring_config_alert_thresholds_with_custom_dict(self):
        """测试MonitoringConfig的alert_thresholds可以设置为自定义字典"""
        custom_thresholds = {'cpu': 80.0, 'memory': 90.0}
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM,
            alert_thresholds=custom_thresholds
        )
        
        assert config.alert_thresholds == custom_thresholds

    def test_monitoring_config_notification_channels_defaults_to_empty_list(self):
        """测试MonitoringConfig的notification_channels默认为空列表"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        assert config.notification_channels == []
        assert isinstance(config.notification_channels, list)

    def test_monitoring_config_notification_channels_with_explicit_none(self):
        """测试MonitoringConfig的notification_channels显式设置为None时初始化为空列表"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM,
            notification_channels=None
        )
        
        assert config.notification_channels == []
        assert isinstance(config.notification_channels, list)

    def test_monitoring_config_notification_channels_with_custom_list(self):
        """测试MonitoringConfig的notification_channels可以设置为自定义列表"""
        custom_channels = ['email', 'sms', 'webhook']
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM,
            notification_channels=custom_channels
        )
        
        assert config.notification_channels == custom_channels
        assert len(config.notification_channels) == 3

    def test_monitoring_config_default_values(self):
        """测试MonitoringConfig的默认值"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        assert config.enabled == True
        assert config.interval == 60
        assert config.timeout == 30
        assert config.retries == 3
        assert config.alert_thresholds == {}
        assert config.notification_channels == []

    def test_monitoring_config_custom_values(self):
        """测试MonitoringConfig的自定义值"""
        config = MonitoringConfig(
            monitor_type=MonitorType.PERFORMANCE,
            enabled=False,
            interval=30,
            timeout=15,
            retries=5,
            alert_thresholds={'cpu': 80.0},
            notification_channels=['email']
        )
        
        assert config.monitor_type == MonitorType.PERFORMANCE
        assert config.enabled == False
        assert config.interval == 30
        assert config.timeout == 15
        assert config.retries == 5
        assert config.alert_thresholds == {'cpu': 80.0}
        assert config.notification_channels == ['email']


class TestDataclassPostInitIntegration:
    """测试dataclass __post_init__方法集成场景"""

    def test_alert_tags_mutable(self):
        """测试Alert的tags是可变的"""
        alert = Alert(
            alert_id='test_alert',
            title='Test Alert',
            description='Test description',
            level=AlertLevel.INFO,
            status=AlertStatus.ACTIVE,
            source='test',
            component='test_component',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 可以添加标签
        alert.tags.append('new_tag')
        assert 'new_tag' in alert.tags

    def test_health_check_details_mutable(self):
        """测试HealthCheck的details是可变的"""
        health_check = HealthCheck(
            component='test_component',
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now()
        )
        
        # 可以添加详情
        health_check.details['new_key'] = 'new_value'
        assert health_check.details['new_key'] == 'new_value'

    def test_monitoring_config_thresholds_mutable(self):
        """测试MonitoringConfig的alert_thresholds是可变的"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        # 可以添加阈值
        config.alert_thresholds['cpu'] = 80.0
        assert config.alert_thresholds['cpu'] == 80.0

    def test_monitoring_config_channels_mutable(self):
        """测试MonitoringConfig的notification_channels是可变的"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        # 可以添加通道
        config.notification_channels.append('email')
        assert 'email' in config.notification_channels

