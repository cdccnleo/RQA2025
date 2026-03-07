"""
测试系统监控组件 - 完整版本

验证SystemMonitor、SystemInfoCollector、MetricsCalculator等类的完整功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

try:
    from src.infrastructure.resource.core.system_monitor import (
        SystemMonitorConfig, SystemInfoCollector, MetricsCalculator,
        MonitorEngine, SystemMonitorFacade, SystemAlertManager
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
    class MonitorEngine:
        pass
    class SystemMonitorFacade:
        pass
    print(f"Warning: 无法导入所需模块: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestSystemMonitorConfig:
    """测试SystemMonitorConfig配置类"""

    def test_system_monitor_config_defaults(self):
        """测试系统监控配置默认值"""
        config = SystemMonitorConfig()

        assert config.check_interval == 60.0
        assert config.alert_handlers is None
        assert config.psutil_mock is None
        assert config.os_mock is None
        assert config.socket_mock is None
        assert config.registry is None
        assert config.cpu_threshold == 90.0
        assert config.memory_threshold == 90.0
        assert config.disk_threshold == 90.0

    def test_system_monitor_config_custom_values(self):
        """测试系统监控配置自定义值"""
        alert_handlers = [MagicMock()]
        config = SystemMonitorConfig(
            check_interval=30.0,
            alert_handlers=alert_handlers,
            cpu_threshold=85.0,
            memory_threshold=80.0,
            disk_threshold=75.0
        )

        assert config.check_interval == 30.0
        assert config.alert_handlers == alert_handlers
        assert config.cpu_threshold == 85.0
        assert config.memory_threshold == 80.0
        assert config.disk_threshold == 75.0


class TestSystemInfoCollector:
    """测试SystemInfoCollector类"""

    def test_system_info_collector_initialization(self):
        """测试系统信息收集器初始化"""
        collector = SystemInfoCollector()

        assert hasattr(collector, 'psutil')
        assert hasattr(collector, 'os')
        assert hasattr(collector, 'socket')

    def test_system_info_collector_with_mocks(self):
        """测试系统信息收集器带mock初始化"""
        mock_psutil = MagicMock()
        mock_os = MagicMock()
        mock_socket = MagicMock()

        collector = SystemInfoCollector(
            psutil_mock=mock_psutil,
            os_mock=mock_os,
            socket_mock=mock_socket
        )

        assert collector.psutil == mock_psutil
        assert collector.os == mock_os
        assert collector.socket == mock_socket

    def test_collect_system_info(self):
        """测试收集系统信息"""
        # Mock platform
        mock_platform = MagicMock()
        mock_platform.platform.return_value = "Windows"
        mock_platform.node.return_value = "test-server"
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.version.return_value = "10.0.19041"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel"

        # Mock socket
        mock_socket = MagicMock()
        mock_socket.gethostname.return_value = "test-server"
        mock_socket.gethostbyname.return_value = "192.168.1.100"

        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.boot_time.return_value = 1609459200  # 2021-01-01 00:00:00

        collector = SystemInfoCollector(
            platform_mock=mock_platform,
            socket_mock=mock_socket,
            psutil_mock=mock_psutil
        )

        info = collector.collect_system_info()

        assert info['hostname'] == "test-server"
        assert info['platform'] == "Windows"
        assert info['system'] == "Windows"
        assert info['release'] == "10"
        assert info['version'] == "10.0.19041"
        assert info['machine'] == "AMD64"
        assert info['processor'] == "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel"
        assert info['ip_address'] == "192.168.1.100"

    def test_collect_system_info_with_mocks(self):
        """测试收集系统信息（带mock）"""
        mock_platform = MagicMock()
        mock_platform.node.return_value = "mock-server"
        mock_platform.system.return_value = "Linux"
        mock_platform.platform.return_value = "Linux"
        mock_platform.release.return_value = "Ubuntu 20.04"
        mock_platform.processor.return_value = "x86_64"
        mock_platform.version.return_value = "20.04"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.python_version.return_value = "3.8.0"

        mock_socket = MagicMock()
        mock_socket.gethostname.return_value = "mock-server"
        mock_socket.gethostbyname.return_value = "10.0.0.1"

        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.boot_time.return_value = 1609459200  # 2021-01-01 00:00:00

        collector = SystemInfoCollector(
            os_mock=mock_platform,
            socket_mock=mock_socket,
            psutil_mock=mock_psutil
        )

        info = collector.collect_system_info()

        assert info['hostname'] == "mock-server"
        assert info['platform'] == "Linux"
        assert info['ip_address'] == "10.0.0.1"

    def test_collect_system_info_exception_handling(self):
        """测试收集系统信息异常处理"""
        mock_platform = MagicMock()
        mock_platform.node.side_effect = Exception("Platform error")

        collector = SystemInfoCollector(os_mock=mock_platform)

        info = collector.collect_system_info()

        # 应该返回包含错误信息的结果
        assert 'error' in info or 'hostname' in info


class TestMetricsCalculator:
    """测试MetricsCalculator类"""

    def test_metrics_calculator_initialization(self):
        """测试指标计算器初始化"""
        calculator = MetricsCalculator()

        assert hasattr(calculator, 'psutil')

    def test_metrics_calculator_with_mock(self):
        """测试指标计算器带mock初始化"""
        mock_psutil = MagicMock()
        calculator = MetricsCalculator(psutil_mock=mock_psutil)

        assert calculator.psutil == mock_psutil

    def test_calculate_stats(self):
        """测试计算统计信息"""
        mock_psutil = MagicMock()

        # Mock CPU
        mock_cpu = MagicMock()
        mock_cpu.percent.return_value = 45.5
        mock_psutil.cpu_percent.return_value = mock_cpu.percent.return_value
        mock_psutil.cpu_count.return_value = 8

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.percent = 67.8
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory

        # Mock disk
        mock_disk = MagicMock()
        mock_disk.percent = 55.2
        mock_disk.total = 500 * 1024**3  # 500GB
        mock_disk.free = 200 * 1024**3  # 200GB
        mock_psutil.disk_usage.return_value = mock_disk

        # Mock network
        mock_net = MagicMock()
        mock_net.bytes_sent = 1024 * 1024  # 1MB
        mock_net.bytes_recv = 2 * 1024 * 1024  # 2MB
        mock_psutil.net_io_counters.return_value = mock_net

        calculator = MetricsCalculator(psutil_mock=mock_psutil)

        stats = calculator.calculate_stats()

        assert stats['cpu_percent'] == 45.5
        assert stats['cpu_count'] == 8
        assert stats['memory_percent'] == 67.8
        assert stats['memory_total'] == 16 * 1024**3
        assert stats['memory_available'] == 8 * 1024**3
        assert stats['disk_percent'] == 55.2
        assert stats['disk_total'] == 500 * 1024**3
        assert stats['disk_free'] == 200 * 1024**3
        assert stats['network_bytes_sent'] == 1024 * 1024
        assert stats['network_bytes_recv'] == 2 * 1024 * 1024

    def test_calculate_stats_exception_handling(self):
        """测试计算统计信息异常处理"""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.side_effect = Exception("PSUtil error")

        calculator = MetricsCalculator(psutil_mock=mock_psutil)

        stats = calculator.calculate_stats()

        # 应该返回包含默认值的结果
        assert isinstance(stats, dict)
        assert 'cpu_percent' in stats


class TestMonitorEngine:
    """测试MonitorEngine类"""

    def test_monitor_engine_initialization(self):
        """测试监控引擎初始化"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        assert engine.config == config
        assert engine.monitoring_active is False
        assert engine.monitor_thread is None
        assert len(engine.metrics_history) == 0

    def test_start_monitoring(self):
        """测试启动监控"""
        config = SystemMonitorConfig(check_interval=0.1)  # 快速测试
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        with patch.object(engine, '_monitor_loop', side_effect=KeyboardInterrupt):
            engine.start_monitoring()

            assert engine.monitoring_active is True
            assert engine.monitor_thread is not None

            # 停止监控
            engine.stop_monitoring()
            assert engine.monitoring_active is False

    def test_stop_monitoring_without_start(self):
        """测试在未启动时停止监控"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        engine.stop_monitoring()
        assert engine.monitoring_active is False

    def test_get_performance_report(self):
        """测试获取性能报告"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        # 直接设置_stats数组来模拟有数据的情况
        engine._stats = [{
            'cpu_percent': 50.0,
            'memory_percent': 60.0,
            'disk_percent': 40.0
        }]

        report = engine.get_performance_report()

        assert report.cpu_usage == 50.0
        assert report.memory_usage == 60.0
        assert report.disk_usage == 40.0

    def test_get_system_resources(self):
        """测试获取系统资源"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        with patch.object(engine.system_info_collector, 'collect_system_info') as mock_collect:
            mock_collect.return_value = {
                'hostname': 'test-server',
                'cpu_count': 8,
                'memory_total': 16 * 1024**3
            }

            resources = engine.get_system_resources()

            assert resources['hostname'] == 'test-server'
            assert resources['cpu_count'] == 8
            mock_collect.assert_called_once()

    def test_monitor_resources(self):
        """测试监控资源"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        with patch.object(engine.metrics_calculator, 'calculate_stats') as mock_calculate:
            mock_calculate.return_value = {
                'cpu_percent': 45.0,
                'memory_percent': 60.0,
                'disk_percent': 50.0
            }

            metrics = engine.monitor_resources()

            assert 'cpu' in metrics
            assert 'memory' in metrics
            assert 'disk' in metrics
            assert metrics['cpu']['usage'] == 45.0
            assert metrics['memory']['usage'] == 60.0
            assert metrics['disk']['usage'] == 50.0

            # 验证历史记录
            assert len(engine.metrics_history) == 1

    def test_check_alert_thresholds_no_alerts(self):
        """测试检查告警阈值（无告警）"""
        config = SystemMonitorConfig(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        metrics = {
            'cpu': {'usage': 70.0},
            'memory': {'usage': 75.0},
            'disk': {'usage': 80.0}
        }

        alerts = engine.check_alert_thresholds(metrics)

        assert len(alerts) == 0

    def test_check_alert_thresholds_with_alerts(self):
        """测试检查告警阈值（有告警）"""
        config = SystemMonitorConfig(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        metrics = {
            'cpu': {'usage': 85.0},  # 超过阈值
            'memory': {'usage': 90.0},  # 超过阈值
            'disk': {'usage': 80.0}  # 未超过阈值
        }

        alerts = engine.check_alert_thresholds(metrics)

        assert len(alerts) == 2
        alert_messages = [alert['message'] for alert in alerts]
        assert any('CPU' in msg for msg in alert_messages)
        assert any('内存' in msg for msg in alert_messages)

    def test_get_monitoring_history(self):
        """测试获取监控历史"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        # 添加一些历史数据
        current_time = datetime.now()
        old_time = (current_time - timedelta(hours=8)).replace(minute=0, second=0, microsecond=0)
        recent_time = current_time.replace(minute=0, second=0, microsecond=0)

        engine._stats = [
            {'timestamp': old_time, 'cpu_usage': 50.0, 'memory_usage': 60.0},
            {'timestamp': recent_time, 'cpu_usage': 70.0, 'memory_usage': 75.0}
        ]

        history = engine.get_monitoring_history(hours=6)

        assert len(history) == 1
        assert history[0]['cpu_usage'] == 70.0

    def test_get_health_status(self):
        """测试获取健康状态"""
        config = SystemMonitorConfig()
        metrics_calculator = MetricsCalculator()
        alert_manager = SystemAlertManager()
        engine = MonitorEngine(config, metrics_calculator, alert_manager)

        with patch.object(engine.metrics_calculator, 'calculate_stats') as mock_calculate:
            mock_calculate.return_value = {
                'cpu_percent': 85.0,  # 高负载
                'memory_percent': 75.0,  # 正常
                'disk_percent': 60.0  # 正常
            }

            health = engine.get_health_status()

            assert 'overall' in health
            assert 'components' in health
            assert health['components']['cpu'] == 'warning'  # CPU高负载
            assert health['components']['memory'] == 'good'
            assert health['components']['disk'] == 'good'


class TestSystemMonitorFacade:
    """测试SystemMonitorFacade类"""

    def test_facade_initialization(self):
        """测试门面初始化"""
        facade = SystemMonitorFacade()

        assert facade.system_info_collector is not None
        assert facade.metrics_calculator is not None
        assert facade.monitor_engine is not None
        assert facade.alert_manager is not None

    def test_get_system_info_delegation(self):
        """测试获取系统信息委托"""
        facade = SystemMonitorFacade()

        with patch.object(facade.system_info_collector, 'collect_system_info') as mock_collect:
            mock_collect.return_value = {'hostname': 'test-server'}

            info = facade.get_system_info()

            assert info['hostname'] == 'test-server'
            mock_collect.assert_called_once()

    def test_get_stats_delegation(self):
        """测试获取统计信息委托"""
        facade = SystemMonitorFacade()

        with patch.object(facade.metrics_calculator, 'calculate_stats') as mock_calculate:
            mock_calculate.return_value = {'cpu_percent': 50.0}

            stats = facade.get_stats(current=True)

            assert stats['cpu_percent'] == 50.0
            mock_calculate.assert_called_once()

    def test_start_monitoring_delegation(self):
        """测试启动监控委托"""
        facade = SystemMonitorFacade()

        with patch.object(facade.monitor_engine, 'start_monitoring') as mock_start:
            facade.start_monitoring()

            mock_start.assert_called_once()

    def test_stop_monitoring_delegation(self):
        """测试停止监控委托"""
        facade = SystemMonitorFacade()

        with patch.object(facade.monitor_engine, 'stop_monitoring') as mock_stop:
            facade.stop_monitoring()

            mock_stop.assert_called_once()

    def test_get_performance_report_delegation(self):
        """测试获取性能报告委托"""
        facade = SystemMonitorFacade()

        expected_metrics = {
            'cpu_usage': 75.5,
            'memory_usage': 82.3,
            'disk_usage': 65.0,
            'network_latency': 25.0,
            'test_execution_time': 120.5,
            'test_success_rate': 95.0,
            'active_threads': 8,
            'timestamp': datetime.now()
        }

        with patch.object(facade.monitor_engine, 'get_performance_report') as mock_report:
            mock_report.return_value = expected_metrics

            report = facade.get_performance_report()

            assert report == expected_metrics
            mock_report.assert_called_once()

    def test_get_system_resources_delegation(self):
        """测试获取系统资源委托"""
        facade = SystemMonitorFacade()

        with patch.object(facade.monitor_engine, 'get_system_resources') as mock_resources:
            mock_resources.return_value = {'hostname': 'test-server'}

            resources = facade.get_system_resources()

            assert resources['hostname'] == 'test-server'
            mock_resources.assert_called_once()

    def test_configure_monitoring(self):
        """测试配置监控"""
        facade = SystemMonitorFacade()

        config = {
            'cpu_threshold': 85.0,
            'memory_threshold': 80.0,
            'check_interval': 30.0
        }

        with patch.object(facade.monitor_engine, 'configure') as mock_config:
            facade.configure_monitoring(config)

            mock_config.assert_called_once_with(config)

    def test_reset_monitoring(self):
        """测试重置监控"""
        facade = SystemMonitorFacade()

        with patch.object(facade.monitor_engine, 'reset') as mock_reset, \
             patch.object(facade.alert_manager, 'reset') as mock_alert_reset:

            facade.reset()

            mock_reset.assert_called_once()
            mock_alert_reset.assert_called_once()

    def test_monitoring_integration_workflow(self):
        """测试监控集成工作流"""
        facade = SystemMonitorFacade()

        # 1. 获取系统信息
        with patch.object(facade.system_info_collector, 'collect_system_info') as mock_info:
            mock_info.return_value = {'hostname': 'test-server'}
            info = facade.get_system_info()
            assert info['hostname'] == 'test-server'

        # 2. 获取统计信息
        with patch.object(facade.metrics_calculator, 'calculate_stats') as mock_stats:
            mock_stats.return_value = {'cpu_percent': 60.0}
            stats = facade.get_stats(current=True)
            assert stats['cpu_percent'] == 60.0

        # 3. 启动监控
        with patch.object(facade.monitor_engine, 'start_monitoring') as mock_start:
            facade.start_monitoring()
            mock_start.assert_called_once()

        # 4. 获取性能报告
        with patch.object(facade.monitor_engine, 'get_performance_report') as mock_report:
            mock_report.return_value = {'cpu_usage': 55.0}
            report = facade.get_performance_report()
            assert report['cpu_usage'] == 55.0

        # 5. 停止监控
        with patch.object(facade.monitor_engine, 'stop_monitoring') as mock_stop:
            facade.stop_monitoring()
            mock_stop.assert_called_once()

    def test_error_handling_in_facade(self):
        """测试门面中的错误处理"""
        facade = SystemMonitorFacade()

        # 测试获取系统信息失败
        with patch.object(facade.system_info_collector, 'collect_system_info') as mock_info:
            mock_info.side_effect = Exception("Info collection failed")

            # 不应该抛出异常
            info = facade.get_system_info()
            assert isinstance(info, dict)  # 应该返回错误信息或默认值

        # 测试获取统计失败
        with patch.object(facade.metrics_calculator, 'calculate_stats') as mock_stats:
            mock_stats.side_effect = Exception("Stats calculation failed")

            # 不应该抛出异常
            stats = facade.get_stats(current=True)
            assert isinstance(stats, dict)  # 应该返回错误信息或默认值

