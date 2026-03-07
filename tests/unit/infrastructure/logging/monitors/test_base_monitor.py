"""
测试基础日志监控器

覆盖 base.py 中的 ILogMonitor 接口和 BaseMonitor 类
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.infrastructure.logging.monitors.base import ILogMonitor, BaseMonitor
from src.infrastructure.logging.core.exceptions import LogMonitorError


class TestILogMonitor:
    """ILogMonitor 接口测试"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            ILogMonitor()

    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        # 检查必需的方法是否存在
        required_methods = ['check_health', 'collect_metrics', 'detect_anomalies', 'get_status']

        for method_name in required_methods:
            assert hasattr(ILogMonitor, method_name), f"Missing method: {method_name}"

            # 检查方法是否是抽象的
            method = getattr(ILogMonitor, method_name)
            assert hasattr(method, '__isabstractmethod__'), f"Method {method_name} should be abstract"


class ConcreteMonitor(BaseMonitor):
    """用于测试的BaseMonitor具体实现"""

    def _check_health(self):
        return {'status': 'healthy'}

    def _collect_metrics(self):
        return {'test_metric': 42}

    def _detect_anomalies(self):
        return [{'type': 'test_anomaly'}]


class TestBaseMonitor:
    """BaseMonitor 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        monitor = ConcreteMonitor()

        # config会包含同步的字段值
        assert monitor.config['interval'] == 60
        assert monitor.config['retention_days'] == 7
        assert monitor.name == "ConcreteMonitor"
        assert monitor.enabled == True
        assert monitor.interval == 60
        assert monitor.retention_days == 7
        assert monitor.last_check_time is None
        assert monitor.metrics_history == []
        assert monitor.anomalies == []
        assert monitor.health_status == 'unknown'

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'name': 'CustomMonitor',
            'enabled': False,
            'interval': 30,
            'retention_days': 14
        }
        monitor = ConcreteMonitor(config)

        assert monitor.config == config
        assert monitor.name == "CustomMonitor"
        assert monitor.enabled == False
        assert monitor.interval == 30
        assert monitor.retention_days == 14

    def test_init_invalid_interval(self):
        """测试无效interval的处理"""
        config = {'interval': -5}
        monitor = ConcreteMonitor(config)

        assert monitor.interval == 60  # 应该使用默认值
        assert monitor.config['interval'] == 60

    def test_init_invalid_retention(self):
        """测试无效retention_days的处理"""
        config = {'retention_days': 0}
        monitor = ConcreteMonitor(config)

        assert monitor.retention_days == 7  # 应该使用默认值
        assert monitor.config['retention_days'] == 7

    def test_check_health_disabled(self):
        """测试禁用状态下的健康检查"""
        monitor = ConcreteMonitor()
        monitor.enabled = False

        result = monitor.check_health()

        assert result == {'status': 'disabled', 'enabled': False}

    def test_check_health_enabled(self):
        """测试启用状态下的健康检查"""
        monitor = ConcreteMonitor()

        result = monitor.check_health()

        assert result == {'status': 'healthy'}
        assert monitor.health_status == 'healthy'
        assert monitor.last_check_time is not None
        assert isinstance(monitor.last_check_time, datetime)

    def test_check_health_exception(self):
        """测试健康检查异常处理"""
        monitor = ConcreteMonitor()

        with patch.object(monitor, '_check_health', side_effect=Exception("Test error")):
            with pytest.raises(LogMonitorError) as exc_info:
                monitor.check_health()

            assert "Health check failed: Test error" in str(exc_info.value)
            assert monitor.health_status == 'error'

    def test_collect_metrics_disabled(self):
        """测试禁用状态下的指标收集"""
        monitor = ConcreteMonitor()
        monitor.enabled = False

        result = monitor.collect_metrics()
        assert result == {}

    def test_collect_metrics_enabled(self):
        """测试启用状态下的指标收集"""
        monitor = ConcreteMonitor()

        result = monitor.collect_metrics()

        assert result['test_metric'] == 42
        assert 'timestamp' in result
        assert result['monitor_name'] == 'ConcreteMonitor'
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] == result

    def test_collect_metrics_exception(self):
        """测试指标收集异常处理"""
        monitor = ConcreteMonitor()

        with patch.object(monitor, '_collect_metrics', side_effect=Exception("Collection error")):
            with pytest.raises(LogMonitorError) as exc_info:
                monitor.collect_metrics()

            assert "Metrics collection failed: Collection error" in str(exc_info.value)

    def test_detect_anomalies_disabled(self):
        """测试禁用状态下的异常检测"""
        monitor = ConcreteMonitor()
        monitor.enabled = False

        result = monitor.detect_anomalies()
        assert result == []

    def test_detect_anomalies_enabled(self):
        """测试启用状态下的异常检测"""
        monitor = ConcreteMonitor()

        result = monitor.detect_anomalies()

        assert len(result) == 1
        assert result[0]['type'] == 'test_anomaly'
        assert len(monitor.anomalies) == 1

        # 检查时间戳和监控器名称是否被添加
        anomaly = monitor.anomalies[0]
        assert 'timestamp' in anomaly
        assert anomaly['monitor_name'] == 'ConcreteMonitor'

    def test_detect_anomalies_exception(self):
        """测试异常检测异常处理"""
        monitor = ConcreteMonitor()

        with patch.object(monitor, '_detect_anomalies', side_effect=Exception("Detection error")):
            with pytest.raises(LogMonitorError) as exc_info:
                monitor.detect_anomalies()

            assert "Anomaly detection failed: Detection error" in str(exc_info.value)

    def test_get_status(self):
        """测试获取状态"""
        monitor = ConcreteMonitor()

        # 设置一些状态
        monitor.health_status = 'healthy'
        monitor.last_check_time = datetime(2023, 1, 1, 12, 0, 0)
        monitor.metrics_history = [{'metric1': 1}, {'metric2': 2}]
        monitor.anomalies = [{'anomaly1': 1}]

        status = monitor.get_status()

        assert status['name'] == 'ConcreteMonitor'
        assert status['enabled'] == True
        assert status['health_status'] == 'healthy'
        assert status['interval'] == 60
        assert status['retention_days'] == 7
        assert status['metrics_count'] == 2
        assert status['anomalies_count'] == 1
        assert status['type'] == 'ConcreteMonitor'
        assert 'last_check_time' in status

    def test_cleanup_old_data(self):
        """测试清理过期数据"""
        monitor = ConcreteMonitor()
        monitor.retention_days = 1  # 1天保留期

        # 创建一些测试数据
        old_time = (datetime.now() - timedelta(days=2)).isoformat()
        new_time = datetime.now().isoformat()

        monitor.metrics_history = [
            {'timestamp': old_time, 'value': 1},
            {'timestamp': new_time, 'value': 2}
        ]
        monitor.anomalies = [
            {'timestamp': old_time, 'type': 'old'},
            {'timestamp': new_time, 'type': 'new'}
        ]

        monitor._cleanup_old_data()

        # 应该只保留新数据
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0]['value'] == 2
        assert len(monitor.anomalies) == 1
        assert monitor.anomalies[0]['type'] == 'new'
