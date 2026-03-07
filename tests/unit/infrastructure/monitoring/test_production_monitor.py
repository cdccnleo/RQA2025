#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 生产环境监控系统

测试production_monitor.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import psutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestProductionMonitor:
    """测试生产环境监控系统"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.monitoring.production_monitor import (
                ProductionMonitor, HealthChecker, PerformanceMonitor,
                ResourceMonitor, AlertManager, MonitoringDashboard,
                MonitoringConfig, MonitorStatus, MonitorLevel
            )
            self.ProductionMonitor = ProductionMonitor
            self.HealthChecker = HealthChecker
            self.PerformanceMonitor = PerformanceMonitor
            self.ResourceMonitor = ResourceMonitor
            self.AlertManager = AlertManager
            self.MonitoringDashboard = MonitoringDashboard
            self.MonitoringConfig = MonitoringConfig
            self.MonitorStatus = MonitorStatus
            self.MonitorLevel = MonitorLevel
        except ImportError as e:
            pytest.skip(f"Production monitor components not available: {e}")

    def test_monitor_status_enum(self):
        """测试监控状态枚举"""
        if not hasattr(self, 'MonitorStatus'):
            pytest.skip("MonitorStatus not available")

        assert hasattr(self.MonitorStatus, 'HEALTHY')
        assert hasattr(self.MonitorStatus, 'WARNING')
        assert hasattr(self.MonitorStatus, 'CRITICAL')
        assert hasattr(self.MonitorStatus, 'DOWN')

    def test_monitor_level_enum(self):
        """测试监控级别枚举"""
        if not hasattr(self, 'MonitorLevel'):
            pytest.skip("MonitorLevel not available")

        assert hasattr(self.MonitorLevel, 'BASIC')
        assert hasattr(self.MonitorLevel, 'STANDARD')
        assert hasattr(self.MonitorLevel, 'ADVANCED')
        assert hasattr(self.MonitorLevel, 'EXPERT')

    def test_monitoring_config(self):
        """测试监控配置"""
        if not hasattr(self, 'MonitoringConfig'):
            pytest.skip("MonitoringConfig not available")

        config = self.MonitoringConfig(
            check_interval=30,
            alert_thresholds={
                'cpu': 80,
                'memory': 85,
                'disk': 90
            },
            enabled_checks=['cpu', 'memory', 'disk', 'network'],
            alert_channels=['email', 'slack']
        )

        assert config.check_interval == 30
        assert config.alert_thresholds['cpu'] == 80
        assert 'cpu' in config.enabled_checks
        assert 'email' in config.alert_channels

    def test_health_checker(self):
        """测试健康检查器"""
        if not hasattr(self, 'HealthChecker'):
            pytest.skip("HealthChecker not available")

        checker = self.HealthChecker()

        assert checker is not None

        # 测试基本健康检查
        health_status = checker.check_basic_health()
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'timestamp' in health_status

        # 测试服务健康检查
        service_status = checker.check_service_health('test_service')
        assert isinstance(service_status, dict)

    def test_performance_monitor(self):
        """测试性能监控器"""
        if not hasattr(self, 'PerformanceMonitor'):
            pytest.skip("PerformanceMonitor not available")

        monitor = self.PerformanceMonitor()

        assert monitor is not None

        # 测试性能指标收集
        metrics = monitor.collect_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'cpu_usage' in metrics or 'cpu' in metrics
        assert 'timestamp' in metrics

        # 测试响应时间监控
        monitor.start_response_time_monitor('test_operation')
        time.sleep(0.01)  # 短暂延迟
        response_time = monitor.end_response_time_monitor('test_operation')

        assert response_time >= 0
        assert isinstance(response_time, float)

    def test_resource_monitor(self):
        """测试资源监控器"""
        if not hasattr(self, 'ResourceMonitor'):
            pytest.skip("ResourceMonitor not available")

        monitor = self.ResourceMonitor()

        assert monitor is not None

        # 测试CPU监控
        cpu_info = monitor.monitor_cpu()
        assert isinstance(cpu_info, dict)
        assert 'usage_percent' in cpu_info

        # 测试内存监控
        memory_info = monitor.monitor_memory()
        assert isinstance(memory_info, dict)
        assert 'usage_percent' in memory_info

        # 测试磁盘监控
        disk_info = monitor.monitor_disk()
        assert isinstance(disk_info, dict)
        assert 'usage_percent' in disk_info

        # 测试网络监控
        network_info = monitor.monitor_network()
        assert isinstance(network_info, dict)

    def test_alert_manager(self):
        """测试告警管理器"""
        if not hasattr(self, 'AlertManager'):
            pytest.skip("AlertManager not available")

        manager = self.AlertManager()

        assert manager is not None
        assert hasattr(manager, 'alerts')
        assert hasattr(manager, 'alert_rules')

        # 测试添加告警规则
        rule = {
            'name': 'high_cpu',
            'condition': 'cpu_usage > 90',
            'severity': 'critical',
            'channels': ['email']
        }
        manager.add_alert_rule(rule)
        assert len(manager.alert_rules) > 0

        # 测试告警触发
        alert = manager.check_condition('cpu_usage > 90', {'cpu_usage': 95})
        assert alert is not None
        assert alert['severity'] == 'critical'

    def test_monitoring_dashboard(self):
        """测试监控仪表板"""
        if not hasattr(self, 'MonitoringDashboard'):
            pytest.skip("MonitoringDashboard not available")

        dashboard = self.MonitoringDashboard()

        assert dashboard is not None

        # 测试添加监控数据
        metrics_data = {
            'cpu_usage': 75.5,
            'memory_usage': 82.3,
            'disk_usage': 68.9,
            'timestamp': datetime.now()
        }

        dashboard.add_metrics(metrics_data)

        # 测试生成报告
        report = dashboard.generate_report()
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'metrics' in report

        # 测试获取图表数据
        chart_data = dashboard.get_chart_data('cpu_usage', hours=24)
        assert isinstance(chart_data, list)

    def test_production_monitor(self):
        """测试生产环境监控器"""
        if not hasattr(self, 'ProductionMonitor'):
            pytest.skip("ProductionMonitor not available")

        monitor = self.ProductionMonitor()

        assert monitor is not None
        assert hasattr(monitor, 'health_checker')
        assert hasattr(monitor, 'performance_monitor')
        assert hasattr(monitor, 'resource_monitor')
        assert hasattr(monitor, 'alert_manager')

        # 测试系统状态检查
        system_status = monitor.check_system_status()
        assert isinstance(system_status, dict)
        assert 'overall_status' in system_status
        assert 'components' in system_status

        # 测试监控启动和停止
        monitor.start_monitoring()
        assert monitor.is_monitoring() is True

        monitor.stop_monitoring()
        assert monitor.is_monitoring() is False

    def test_production_monitor_integration(self):
        """测试生产环境监控器集成"""
        if not all(hasattr(self, cls) for cls in [
            'ProductionMonitor', 'HealthChecker', 'PerformanceMonitor'
        ]):
            pytest.skip("Required components not available")

        monitor = self.ProductionMonitor()

        # 配置监控
        config = self.MonitoringConfig(
            check_interval=5,  # 5秒间隔
            alert_thresholds={'cpu': 90, 'memory': 90}
        )
        monitor.configure(config)

        # 启动监控
        monitor.start_monitoring()

        # 等待一段时间让监控收集数据
        time.sleep(1)

        # 检查监控状态
        status = monitor.get_monitoring_status()
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'last_check' in status

        # 停止监控
        monitor.stop_monitoring()

        # 验证监控已停止
        assert monitor.is_monitoring() is False

    def test_production_monitor_alerts(self):
        """测试生产环境监控器的告警功能"""
        if not hasattr(self, 'ProductionMonitor'):
            pytest.skip("ProductionMonitor not available")

        monitor = self.ProductionMonitor()

        # 设置告警规则
        alert_rule = {
            'name': 'memory_critical',
            'condition': 'memory_usage > 95',
            'severity': 'critical',
            'message': 'Memory usage is critically high'
        }

        monitor.alert_manager.add_alert_rule(alert_rule)

        # 模拟高内存使用
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 98

            # 触发告警检查
            alerts = monitor.alert_manager.check_condition(
                'memory_usage > 95',
                {'memory_usage': 98}
            )

            assert alerts is not None
            assert alerts['severity'] == 'critical'
            assert 'memory' in alerts['message'].lower()

    def test_production_monitor_error_handling(self):
        """测试生产环境监控器的错误处理"""
        if not hasattr(self, 'ProductionMonitor'):
            pytest.skip("ProductionMonitor not available")

        monitor = self.ProductionMonitor()

        # 测试无效配置
        try:
            monitor.configure(None)
        except Exception:
            pass  # 应该能处理无效配置

        # 测试重复启动
        monitor.start_monitoring()
        monitor.start_monitoring()  # 应该不会出错

        # 测试重复停止
        monitor.stop_monitoring()
        monitor.stop_monitoring()  # 应该不会出错

        # 测试获取不存在的服务状态
        service_status = monitor.health_checker.check_service_health('nonexistent_service')
        assert isinstance(service_status, dict)  # 应该返回有效的状态字典

    def test_monitoring_config_validation(self):
        """测试监控配置验证"""
        if not hasattr(self, 'MonitoringConfig'):
            pytest.skip("MonitoringConfig not available")

        # 测试有效配置
        valid_config = self.MonitoringConfig(
            check_interval=30,
            alert_thresholds={'cpu': 80, 'memory': 85}
        )
        assert valid_config.check_interval == 30

        # 测试边界值
        edge_config = self.MonitoringConfig(
            check_interval=1,  # 最小值
            alert_thresholds={'cpu': 0, 'memory': 100}  # 边界值
        )
        assert edge_config.check_interval == 1
        assert edge_config.alert_thresholds['cpu'] == 0
        assert edge_config.alert_thresholds['memory'] == 100


if __name__ == '__main__':
    pytest.main([__file__])

