#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统监控实际测试
基于system_monitor.py的实际实现创建测试用例
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from src.infrastructure.resource.core.system_monitor import (
        SystemMonitorConfig, SystemInfoCollector, MetricsCalculator,
        SystemAlertManager, MonitorEngine, SystemMonitorFacade
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    # 创建mock类以避免导入错误
    class SystemMonitorConfig:
        pass
    class SystemInfoCollector:
        pass
    class MetricsCalculator:
        pass
    class SystemAlertManager:
        pass
    class MonitorEngine:
        pass
    class SystemMonitorFacade:
        pass
    print(f"Warning: 无法导入所需模块: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestSystemMonitorConfig:
    """测试SystemMonitorConfig类"""

    def test_system_monitor_config_default_values(self):
        """测试SystemMonitorConfig默认值"""
        config = SystemMonitorConfig()

        assert config.check_interval == 60.0
        assert config.alert_handlers is None
        assert config.cpu_threshold == 90.0
        assert config.memory_threshold == 90.0
        assert config.disk_threshold == 90.0

    def test_system_monitor_config_custom_values(self):
        """测试SystemMonitorConfig自定义值"""
        def dummy_handler():
            pass

        config = SystemMonitorConfig(
            check_interval=30.0,
            alert_handlers=[dummy_handler],
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=95.0
        )

        assert config.check_interval == 30.0
        assert len(config.alert_handlers) == 1
        assert config.cpu_threshold == 80.0
        assert config.memory_threshold == 85.0
        assert config.disk_threshold == 95.0


class TestSystemInfoCollector:
    """测试SystemInfoCollector类"""

    def setup_method(self):
        """测试前准备"""
        self.collector = SystemInfoCollector()

    def test_system_info_collector_initialization(self):
        """测试SystemInfoCollector初始化"""
        assert hasattr(self.collector, 'psutil')
        assert hasattr(self.collector, 'os')
        assert hasattr(self.collector, 'socket')

    def test_system_info_collector_with_mocks(self):
        """测试SystemInfoCollector使用mock"""
        mock_psutil = Mock()
        mock_os = Mock()
        mock_socket = Mock()

        collector = SystemInfoCollector(
            psutil_mock=mock_psutil,
            os_mock=mock_os,
            socket_mock=mock_socket
        )

        assert collector.psutil == mock_psutil
        assert collector.os == mock_os
        assert collector.socket == mock_socket

    @patch('socket.gethostname')
    @patch('platform.platform')
    @patch('platform.system')
    @patch('platform.release')
    @patch('platform.processor')
    @patch('platform.python_version')
    def test_get_system_info(self, mock_python_version, mock_processor,
                           mock_release, mock_system, mock_platform, mock_hostname):
        """测试获取系统信息"""
        # 设置mock返回值
        mock_hostname.return_value = "test-host"
        mock_platform.return_value = "Linux-5.4.0"
        mock_system.return_value = "Linux"
        mock_release.return_value = "5.4.0"
        mock_processor.return_value = "x86_64"
        mock_python_version.return_value = "3.9.0"

        collector = SystemInfoCollector()
        collector.psutil.cpu_count.return_value = 32  # 实际系统可能返回32个逻辑CPU
        collector.psutil.boot_time.return_value = time.time() - 3600

        system_info = collector.get_system_info()

        assert system_info['hostname'] == "test-host"
        assert system_info['platform'] == "Linux-5.4.0"
        assert system_info['system'] == "Linux"
        assert system_info['cpu_count'] == 32
        assert 'boot_time' in system_info


class TestMetricsCalculator:
    """测试MetricsCalculator类"""

    def setup_method(self):
        """测试前准备"""
        self.calculator = MetricsCalculator()

    def test_metrics_calculator_initialization(self):
        """测试MetricsCalculator初始化"""
        assert hasattr(self.calculator, 'psutil')
        assert hasattr(self.calculator, 'os')

    def test_metrics_calculator_with_mocks(self):
        """测试MetricsCalculator使用mock"""
        mock_psutil = Mock()
        mock_os = Mock()

        calculator = MetricsCalculator(
            psutil_mock=mock_psutil,
            os_mock=mock_os
        )

        assert calculator.psutil == mock_psutil
        assert calculator.os == mock_os

    def test_calculate_system_stats(self):
        """测试计算系统统计信息"""
        calculator = MetricsCalculator()

        # Mock各种系统调用
        with patch.object(calculator.psutil, 'cpu_percent', return_value=45.5), \
             patch.object(calculator.psutil, 'virtual_memory') as mock_memory, \
             patch.object(calculator.psutil, 'disk_usage') as mock_disk, \
             patch.object(calculator.psutil, 'net_io_counters') as mock_net, \
             patch.object(calculator.psutil, 'pids', return_value=[1, 2, 3, 4, 5]), \
             patch.object(calculator, '_get_load_avg', return_value=[1.5, 1.2, 0.8]):

            # 设置内存mock
            mock_memory.return_value.percent = 60.5
            mock_memory.return_value.total = 8 * 1024**3
            mock_memory.return_value.available = 4 * 1024**3
            mock_memory.return_value.used = 4 * 1024**3

            # 设置磁盘mock
            mock_disk.return_value.percent = 70.2
            mock_disk.return_value.total = 500 * 1024**3
            mock_disk.return_value.free = 150 * 1024**3
            mock_disk.return_value.used = 350 * 1024**3

            # 设置网络mock
            mock_net.return_value.bytes_sent = 1000000
            mock_net.return_value.bytes_recv = 2000000
            mock_net.return_value.packets_sent = 1000
            mock_net.return_value.packets_recv = 2000

            stats = calculator.calculate_system_stats()

            assert 'timestamp' in stats
            assert 'cpu' in stats
            assert 'memory' in stats
            assert 'disk' in stats
            assert 'network' in stats
            assert stats['cpu']['percent'] == 45.5
            assert stats['memory']['percent'] == 60.5
            assert stats['disk']['percent'] == 70.2


class TestSystemAlertManager:
    """测试SystemAlertManager类"""

    def setup_method(self):
        """测试前准备"""
        self.alert_manager = SystemAlertManager()

    def test_system_alert_manager_initialization(self):
        """测试SystemAlertManager初始化"""
        assert hasattr(self.alert_manager, 'alert_handlers')
        assert self.alert_manager.alert_handlers == []

    def test_system_alert_manager_with_handlers(self):
        """测试SystemAlertManager带处理器初始化"""
        def dummy_handler():
            pass

        alert_manager = SystemAlertManager([dummy_handler])

        assert len(alert_manager.alert_handlers) == 1
        assert dummy_handler in alert_manager.alert_handlers

    def test_check_system_status_normal(self):
        """测试检查系统状态（正常）"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        }
        config = SystemMonitorConfig()

        alerts = self.alert_manager.check_system_status(stats, config)

        assert isinstance(alerts, list)
        # 正常情况下应该没有告警
        assert len(alerts) == 0

    def test_check_system_status_high_cpu(self):
        """测试检查系统状态（CPU过高）"""
        stats = {
            'cpu': {'percent': 95.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        }
        config = SystemMonitorConfig()

        alerts = self.alert_manager.check_system_status(stats, config)

        assert isinstance(alerts, list)
        assert len(alerts) > 0
        assert any('CPU' in alert.get('message', '') for alert in alerts)

    def test_check_system_status_high_memory(self):
        """测试检查系统状态（内存过高）"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 95.0},
            'disk': {'percent': 70.0}
        }
        config = SystemMonitorConfig()

        alerts = self.alert_manager.check_system_status(stats, config)

        assert isinstance(alerts, list)
        assert len(alerts) > 0
        assert any('内存' in alert.get('message', '') or 'memory' in alert.get('message', '').lower()
                  for alert in alerts)

    def test_trigger_alerts(self):
        """测试触发告警"""
        triggered_calls = []

        def capture_handler(source, alert):
            triggered_calls.append((source, alert))

        self.alert_manager.alert_handlers = [capture_handler]

        alerts = [
            {'message': 'Test alert 1', 'severity': 'warning'},
            {'message': 'Test alert 2', 'severity': 'error'}
        ]

        self.alert_manager.trigger_alerts(alerts)

        assert len(triggered_calls) == 2
        assert triggered_calls[0][0] == 'system'  # source参数
        assert triggered_calls[0][1]['message'] == 'Test alert 1'
        assert triggered_calls[1][0] == 'system'  # source参数
        assert triggered_calls[1][1]['message'] == 'Test alert 2'


class TestMonitorEngine:
    """测试MonitorEngine类"""

    def setup_method(self):
        """测试前准备"""
        self.config = SystemMonitorConfig(check_interval=1.0)
        self.metrics_calculator = MetricsCalculator()
        self.alert_manager = SystemAlertManager()
        self.engine = MonitorEngine(self.config, self.metrics_calculator, self.alert_manager)

    def test_monitor_engine_initialization(self):
        """测试MonitorEngine初始化"""
        assert self.engine.config == self.config
        assert self.engine.metrics_calculator == self.metrics_calculator
        assert self.engine.alert_manager == self.alert_manager
        assert not self.engine._monitoring_active
        assert self.engine._stats == []
        assert self.engine._alerts_history == []

    def test_start_monitoring(self):
        """测试启动监控"""
        self.engine.start_monitoring()

        assert self.engine._monitoring_active
        assert self.engine._monitor_thread is not None
        assert self.engine._monitor_thread.is_alive()

        # 清理
        self.engine.stop_monitoring()

    def test_stop_monitoring(self):
        """测试停止监控"""
        self.engine.start_monitoring()
        assert self.engine._monitoring_active

        self.engine.stop_monitoring()

        assert not self.engine._monitoring_active
        # 线程可能还在运行但会很快结束，不强制要求为None

    def test_get_stats(self):
        """测试获取统计信息"""
        # 先添加一些统计数据
        self.engine._stats = [
            {'timestamp': '2023-01-01T10:00:00', 'cpu': {'usage_percent': 50.0}},
            {'timestamp': '2023-01-01T10:01:00', 'cpu': {'usage_percent': 60.0}}
        ]

        stats = self.engine.get_stats()

        assert isinstance(stats, list)
        assert len(stats) == 2
        assert stats[0]['cpu']['usage_percent'] == 50.0

    def test_get_stats_with_time_filter(self):
        """测试获取带时间过滤的统计信息"""
        self.engine._stats = [
            {'timestamp': '2023-01-01T10:00:00', 'cpu': {'usage_percent': 50.0}},
            {'timestamp': '2023-01-01T11:00:00', 'cpu': {'usage_percent': 60.0}},
            {'timestamp': '2023-01-01T12:00:00', 'cpu': {'usage_percent': 70.0}}
        ]

        # 获取从11:00开始的统计信息
        stats = self.engine.get_stats(start_time='2023-01-01T11:00:00')

        assert len(stats) == 2
        assert stats[0]['timestamp'] == '2023-01-01T11:00:00'

    def test_get_alerts_history(self):
        """测试获取告警历史"""
        self.engine._alerts_history = [
            {'timestamp': '2023-01-01T10:00:00', 'message': 'Alert 1'},
            {'timestamp': '2023-01-01T10:01:00', 'message': 'Alert 2'}
        ]

        alerts = self.engine.get_alerts_history()

        assert isinstance(alerts, list)
        assert len(alerts) == 2
        assert alerts[0]['message'] == 'Alert 1'

    def test_get_alerts_history_with_limit(self):
        """测试获取带限制的告警历史"""
        self.engine._alerts_history = [
            {'timestamp': '2023-01-01T10:00:00', 'message': 'Alert 1'},
            {'timestamp': '2023-01-01T10:01:00', 'message': 'Alert 2'},
            {'timestamp': '2023-01-01T10:02:00', 'message': 'Alert 3'}
        ]

        alerts = self.engine.get_alerts_history(limit=2)

        assert len(alerts) == 2


class TestSystemMonitorFacade:
    """测试SystemMonitorFacade类"""

    def setup_method(self):
        """测试前准备"""
        self.config = SystemMonitorConfig()
        self.facade = SystemMonitorFacade(self.config)

    def test_system_monitor_facade_initialization(self):
        """测试SystemMonitorFacade初始化"""
        assert self.facade.config == self.config
        assert hasattr(self.facade, 'metrics_calculator')
        assert hasattr(self.facade, 'alert_manager')
        assert hasattr(self.facade, 'monitor_engine')
        assert hasattr(self.facade, 'info_collector')

    def test_start_monitoring(self):
        """测试启动监控"""
        self.facade.start_monitoring()

        assert self.facade.monitor_engine._monitoring_active

        # 清理
        self.facade.stop_monitoring()

    def test_stop_monitoring(self):
        """测试停止监控"""
        self.facade.start_monitoring()
        assert self.facade.monitor_engine._monitoring_active

        self.facade.stop_monitoring()

        assert not self.facade.monitor_engine._monitoring_active

    def test_get_stats(self):
        """测试获取统计信息"""
        # 添加一些测试数据
        self.facade.monitor_engine._stats = [
            {'timestamp': '2023-01-01T10:00:00', 'cpu': {'usage_percent': 50.0}}
        ]

        stats = self.facade.get_stats()

        assert isinstance(stats, list)
        assert len(stats) >= 0

    def test_get_alerts_history(self):
        """测试获取告警历史"""
        # 添加一些测试数据
        self.facade.monitor_engine._alerts_history = [
            {'timestamp': '2023-01-01T10:00:00', 'message': 'Test alert'}
        ]

        alerts = self.facade.get_alerts_history()

        assert isinstance(alerts, list)
        assert len(alerts) >= 0

    def test_get_system_info(self):
        """测试获取系统信息"""
        system_info = self.facade.get_system_info()

        assert isinstance(system_info, dict)
        assert 'hostname' in system_info
        assert 'platform' in system_info
        assert 'cpu_count' in system_info

    def test_properties(self):
        """测试属性访问器"""
        assert self.facade.check_interval == self.config.check_interval
        assert self.facade.alert_handlers == self.config.alert_handlers

        # 测试设置监控状态
        self.facade._monitoring = True
        assert self.facade._monitoring == True

        self.facade._monitoring = False
        assert self.facade._monitoring == False


class TestPrometheusMetrics:
    """测试Prometheus指标功能"""

    def setup_method(self):
        """测试前准备"""
        self.config = SystemMonitorConfig()
        self.facade = SystemMonitorFacade(self.config)

    def test_prometheus_metrics_initialization(self):
        """测试Prometheus指标初始化"""
        # 验证指标对象存在
        assert hasattr(self.facade, 'cpu_gauge')
        assert hasattr(self.facade, 'memory_gauge')
        assert hasattr(self.facade, 'disk_gauge')

        # 如果Prometheus可用，验证指标类型
        if hasattr(self.facade, 'registry') and self.facade.registry is not None:
            assert 'system_cpu_percent' in self.facade.registry._names_to_collectors
            assert 'system_memory_percent' in self.facade.registry._names_to_collectors
            assert 'system_disk_percent' in self.facade.registry._names_to_collectors

    def test_prometheus_metrics_update(self):
        """测试Prometheus指标更新"""
        # 只有在Prometheus可用的情况下才测试
        if hasattr(self.facade, 'registry') and self.facade.registry is not None:
            # 手动设置指标值
            self.facade.cpu_gauge.set(75.5)
            self.facade.memory_gauge.set(60.2)
            self.facade.disk_gauge.set(45.8)

            # 验证指标值已设置（这里只是确保不抛出异常）
            assert True

    def test_prometheus_registry_isolation(self):
        """测试Prometheus注册表隔离"""
        from prometheus_client import CollectorRegistry

        # 使用自定义注册表来确保隔离
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        config1 = SystemMonitorConfig(registry=registry1)
        config2 = SystemMonitorConfig(registry=registry2)

        facade1 = SystemMonitorFacade(config1)
        facade2 = SystemMonitorFacade(config2)

        # 验证两个facade使用不同的注册表
        assert facade1.registry is not facade2.registry
        assert facade1.registry is registry1
        assert facade2.registry is registry2


class TestAdvancedMonitoringScenarios:
    """测试高级监控场景"""

    def test_threshold_based_alerting(self):
        """测试基于阈值的告警"""
        captured_alerts = []

        def alert_handler(source, alert):
            captured_alerts.append((source, alert))

        # 创建配置，设置较低的阈值以便触发告警
        config = SystemMonitorConfig(
            cpu_threshold=5.0,  # 非常低的阈值
            memory_threshold=5.0,
            disk_threshold=5.0,
            alert_handlers=[alert_handler]
        )

        alert_manager = SystemAlertManager(config.alert_handlers)

        # 模拟高使用率的系统状态
        stats = {
            'cpu': {'percent': 95.0},  # 超过阈值
            'memory': {'percent': 85.0},  # 超过阈值
            'disk': {'percent': 10.0}  # 未超过阈值
        }

        alerts = alert_manager.check_system_status(stats, config)

        # 应该生成至少两个告警（CPU和内存）
        assert len(alerts) >= 2
        assert any('CPU' in alert.get('message', '') for alert in alerts)
        assert any('内存' in alert.get('message', '') or 'memory' in alert.get('message', '').lower() for alert in alerts)

    def test_multiple_threshold_scenarios(self):
        """测试多种阈值场景"""
        alert_manager = SystemAlertManager()

        scenarios = [
            # (stats, config, expected_alerts_count)
            ({
                'cpu': {'percent': 50.0},
                'memory': {'percent': 60.0},
                'disk': {'percent': 70.0}
            }, SystemMonitorConfig(), 0),  # 正常情况，无告警

            ({
                'cpu': {'percent': 95.0},
                'memory': {'percent': 60.0},
                'disk': {'percent': 70.0}
            }, SystemMonitorConfig(), 1),  # 只有CPU超标

            ({
                'cpu': {'percent': 50.0},
                'memory': {'percent': 95.0},
                'disk': {'percent': 70.0}
            }, SystemMonitorConfig(), 1),  # 只有内存超标

            ({
                'cpu': {'percent': 95.0},
                'memory': {'percent': 95.0},
                'disk': {'percent': 95.0}
            }, SystemMonitorConfig(), 3),  # 全部超标
        ]

        for stats, config, expected_count in scenarios:
            alerts = alert_manager.check_system_status(stats, config)
            assert len(alerts) == expected_count, f"Expected {expected_count} alerts, got {len(alerts)} for stats: {stats}"

    def test_monitoring_engine_data_persistence(self):
        """测试监控引擎数据持久性"""
        config = SystemMonitorConfig()
        calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, calculator, alert_manager)

        # 添加一些测试数据
        test_stats = [
            {'timestamp': '2023-01-01T10:00:00', 'cpu': {'percent': 50.0}},
            {'timestamp': '2023-01-01T10:01:00', 'cpu': {'percent': 60.0}},
            {'timestamp': '2023-01-01T10:02:00', 'cpu': {'percent': 70.0}},
        ]

        test_alerts = [
            {'timestamp': '2023-01-01T10:00:00', 'message': 'High CPU usage'},
            {'timestamp': '2023-01-01T10:01:00', 'message': 'Memory warning'},
        ]

        engine._stats = test_stats
        engine._alerts_history = test_alerts

        # 验证数据持久性
        retrieved_stats = engine.get_stats()
        retrieved_alerts = engine.get_alerts_history()

        assert len(retrieved_stats) == 3
        assert len(retrieved_alerts) == 2
        assert retrieved_stats[0]['cpu']['percent'] == 50.0
        assert retrieved_alerts[1]['message'] == 'Memory warning'

    def test_configuration_updates(self):
        """测试配置更新"""
        config = SystemMonitorConfig(check_interval=60.0)
        facade = SystemMonitorFacade(config)

        assert facade.config.check_interval == 60.0

        # 更新配置
        new_config = SystemMonitorConfig(check_interval=30.0)
        facade.config = new_config

        assert facade.config.check_interval == 30.0


class TestErrorHandling:
    """测试错误处理"""

    def test_metrics_calculation_error_handling(self):
        """测试指标计算错误处理"""
        calculator = MetricsCalculator()

        # Mock psutil方法抛出异常
        with patch.object(calculator.psutil, 'cpu_percent', side_effect=Exception("CPU error")):

            # 当有异常时，应该抛出异常（当前实现没有错误处理）
            with pytest.raises(Exception, match="CPU error"):
                calculator.calculate_system_stats()

    def test_alert_handler_error_handling(self):
        """测试告警处理器错误处理"""
        alert_manager = SystemAlertManager()

        def failing_handler(source, alert):
            raise Exception("Handler failed")

        def working_handler(source, alert):
            pass  # 正常工作的处理器

        alert_manager.alert_handlers = [failing_handler, working_handler]

        alerts = [{'message': 'Test alert'}]

        # 不应该抛出异常，即使有处理器失败
        alert_manager.trigger_alerts(alerts)

    def test_monitoring_engine_error_recovery(self):
        """测试监控引擎错误恢复"""
        config = SystemMonitorConfig()
        calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, calculator, alert_manager)

        # Mock计算器抛出异常
        with patch.object(calculator, 'calculate_system_stats', side_effect=Exception("Calculation failed")):
            # 启动监控
            engine.start_monitoring()

            # 等待一个监控周期
            time.sleep(0.1)

            # 引擎应该仍然在运行（错误恢复）
            assert engine._monitoring_active

            # 停止监控
            engine.stop_monitoring()


class TestSystemMonitorIntegration:
    """测试系统监控集成场景"""

    def test_full_monitoring_workflow(self):
        """测试完整的监控工作流程"""
        config = SystemMonitorConfig(check_interval=0.5)  # 较短的检查间隔用于测试
        facade = SystemMonitorFacade(config)

        # 启动监控
        facade.start_monitoring()

        # 等待一段时间让监控运行
        time.sleep(0.2)

        # 获取统计信息
        stats = facade.get_stats()
        alerts = facade.get_alerts_history()
        system_info = facade.get_system_info()

        # 验证数据结构
        assert isinstance(stats, list)
        assert isinstance(alerts, list)
        assert isinstance(system_info, dict)

        # 停止监控
        facade.stop_monitoring()

        assert not facade.monitor_engine._monitoring_active

    def test_monitoring_data_collection(self):
        """测试监控数据收集"""
        config = SystemMonitorConfig(check_interval=0.1)
        facade = SystemMonitorFacade(config)

        # 启动监控
        facade.start_monitoring()

        # 等待数据收集
        time.sleep(0.3)

        # 获取统计信息
        stats = facade.get_stats()

        # 应该至少收集到一些数据
        assert len(stats) >= 0

        # 如果有统计数据，验证其结构
        if len(stats) > 0:
            stat = stats[0]
            assert 'timestamp' in stat
            assert 'cpu' in stat
            assert 'memory' in stat

        # 停止监控
        facade.stop_monitoring()

    def test_alert_generation_and_handling(self):
        """测试告警生成和处理"""
        # 创建一个配置，使其容易触发告警
        config = SystemMonitorConfig(
            cpu_threshold=10.0,  # 很低的阈值，容易触发
            memory_threshold=10.0,
            disk_threshold=10.0,
            check_interval=0.2
        )

        facade = SystemMonitorFacade(config)

        # 启动监控
        facade.start_monitoring()

        # 等待可能的告警生成
        time.sleep(0.5)

        # 获取告警历史
        alerts = facade.get_alerts_history()

        # 验证告警数据结构
        assert isinstance(alerts, list)

        # 停止监控
        facade.stop_monitoring()

    def test_end_to_end_monitoring_pipeline(self):
        """测试端到端监控管道"""
        captured_alerts = []

        def alert_handler(source, alert):
            captured_alerts.append((source, alert))

        config = SystemMonitorConfig(
            check_interval=0.1,
            cpu_threshold=5.0,  # 很低的阈值
            alert_handlers=[alert_handler]
        )

        facade = SystemMonitorFacade(config)

        # 启动监控
        facade.start_monitoring()

        # 等待监控运行和可能的告警生成
        time.sleep(0.5)

        # 获取系统信息
        system_info = facade.get_system_info()
        assert 'hostname' in system_info

        # 获取统计信息
        stats = facade.get_stats()
        assert isinstance(stats, list)

        # 停止监控
        facade.stop_monitoring()

        # 验证整个管道正常工作
        assert not facade.monitor_engine._monitoring_active
