#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
system_monitor 模块测试
测试系统监控器的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

try:
    from src.infrastructure.resource.core.system_monitor import (
        SystemMonitorConfig, SystemInfoCollector, SystemAlertManager,
        MonitorEngine, MetricsCalculator, SystemMonitor
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestSystemMonitorConfig(unittest.TestCase):
    """测试系统监控配置"""

    def test_config_default_values(self):
        """测试配置默认值"""
        config = SystemMonitorConfig()
        
        self.assertEqual(config.check_interval, 60.0)
        self.assertIsNone(config.alert_handlers)
        self.assertEqual(config.cpu_threshold, 90.0)
        self.assertEqual(config.memory_threshold, 90.0)
        self.assertEqual(config.disk_threshold, 90.0)

    def test_config_custom_values(self):
        """测试自定义配置值"""
        def alert_handler(source, data):
            pass
        
        config = SystemMonitorConfig(
            check_interval=30.0,
            alert_handlers=[alert_handler],
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=95.0
        )
        
        self.assertEqual(config.check_interval, 30.0)
        self.assertEqual(len(config.alert_handlers), 1)
        self.assertEqual(config.cpu_threshold, 80.0)
        self.assertEqual(config.memory_threshold, 85.0)
        self.assertEqual(config.disk_threshold, 95.0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestSystemInfoCollector(unittest.TestCase):
    """测试系统信息收集器"""

    def setUp(self):
        """测试前准备"""
        # 创建模拟对象
        self.mock_psutil = Mock()
        self.mock_os = Mock()
        self.mock_socket = Mock()
        
        # 设置模拟返回值
        self.setup_mock_returns()
        
        self.collector = SystemInfoCollector(
            psutil_mock=self.mock_psutil,
            os_mock=self.mock_os,
            socket_mock=self.mock_socket
        )

    def setup_mock_returns(self):
        """设置模拟对象的返回值"""
        # CPU模拟
        mock_cpu = Mock()
        mock_cpu.percent = 50.0
        self.mock_psutil.cpu_percent.return_value = 50.0
        self.mock_psutil.cpu_count.return_value = 8
        
        # 内存模拟
        mock_memory = Mock()
        mock_memory.total = 8589934592  # 8GB
        mock_memory.available = 4294967296  # 4GB
        mock_memory.used = 4294967296  # 4GB
        mock_memory.percent = 50.0
        self.mock_psutil.virtual_memory.return_value = mock_memory
        
        # 磁盘模拟
        mock_disk = Mock()
        mock_disk.total = 1000000000000  # 1TB
        mock_disk.used = 500000000000  # 500GB
        mock_disk.free = 500000000000  # 500GB
        mock_disk.percent = 50.0
        self.mock_psutil.disk_usage.return_value = mock_disk
        
        # 网络模拟
        mock_net_io = Mock()
        mock_net_io.bytes_sent = 1000000
        mock_net_io.bytes_recv = 2000000
        mock_net_io.packets_sent = 1000
        mock_net_io.packets_recv = 2000
        self.mock_psutil.net_io_counters.return_value = mock_net_io
        
        # 进程模拟
        self.mock_psutil.pids.return_value = list(range(100))  # 100个进程

    def test_collector_initialization(self):
        """测试收集器初始化"""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.psutil, self.mock_psutil)

    def test_collector_initialization_with_real_psutil(self):
        """测试使用真实psutil初始化"""
        # 不传递mock，应该使用真实的psutil
        collector = SystemInfoCollector()
        self.assertIsNotNone(collector)

    def test_get_system_stats(self):
        """测试获取系统统计信息"""
        stats = self.collector.get_system_stats()
        
        # 验证返回结构
        self.assertIsInstance(stats, dict)
        self.assertIn('cpu', stats)
        self.assertIn('memory', stats)
        self.assertIn('disk', stats)
        self.assertIn('network', stats)
        self.assertIn('process', stats)

    def test_get_system_stats_cpu_info(self):
        """测试获取CPU信息"""
        stats = self.collector.get_system_stats()
        
        cpu_info = stats['cpu']
        self.assertIn('percent', cpu_info)
        self.assertIn('count', cpu_info)
        self.assertEqual(cpu_info['percent'], 50.0)
        self.assertEqual(cpu_info['count'], 8)

    def test_get_system_stats_memory_info(self):
        """测试获取内存信息"""
        stats = self.collector.get_system_stats()
        
        memory_info = stats['memory']
        self.assertIn('total', memory_info)
        self.assertIn('available', memory_info)
        self.assertIn('used', memory_info)
        self.assertIn('percent', memory_info)

    def test_get_system_stats_disk_info(self):
        """测试获取磁盘信息"""
        stats = self.collector.get_system_stats()
        
        disk_info = stats['disk']
        self.assertIn('total', disk_info)
        self.assertIn('used', disk_info)
        self.assertIn('free', disk_info)
        self.assertIn('percent', disk_info)

    def test_get_system_stats_network_info(self):
        """测试获取网络信息"""
        stats = self.collector.get_system_stats()
        
        network_info = stats['network']
        self.assertIn('bytes_sent', network_info)
        self.assertIn('bytes_recv', network_info)
        self.assertIn('packets_sent', network_info)
        self.assertIn('packets_recv', network_info)

    def test_get_system_stats_process_info(self):
        """测试获取进程信息"""
        stats = self.collector.get_system_stats()
        
        process_info = stats['process']
        self.assertIn('count', process_info)
        self.assertEqual(process_info['count'], 100)

    def test_get_load_avg_available(self):
        """测试获取负载平均值 - 可用时"""
        # 模拟os.getloadavg可用
        self.mock_os.getloadavg.return_value = (1.5, 1.2, 1.0)
        
        load_avg = self.collector._get_load_avg()
        self.assertIsNotNone(load_avg)
        self.assertEqual(load_avg, [1.5, 1.2, 1.0])

    def test_get_load_avg_unavailable(self):
        """测试获取负载平均值 - 不可用时"""
        # 模拟os没有getloadavg方法
        delattr(self.mock_os, 'getloadavg')
        
        load_avg = self.collector._get_load_avg()
        self.assertIsNone(load_avg)

    def test_get_load_avg_exception(self):
        """测试获取负载平均值时发生异常"""
        # 模拟getloadavg抛出异常
        self.mock_os.getloadavg.side_effect = Exception("Load avg error")
        
        load_avg = self.collector._get_load_avg()
        self.assertIsNone(load_avg)


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestSystemAlertManager(unittest.TestCase):
    """测试系统告警管理器"""

    def setUp(self):
        """测试前准备"""
        self.mock_handlers = [Mock(), Mock()]
        self.alert_manager = SystemAlertManager(self.mock_handlers)

    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        self.assertIsNotNone(self.alert_manager)
        self.assertEqual(len(self.alert_manager.alert_handlers), 2)

    def test_alert_manager_no_handlers(self):
        """测试无处理器的告警管理器"""
        alert_manager = SystemAlertManager()
        self.assertEqual(len(alert_manager.alert_handlers), 0)

    def test_check_system_status_no_alerts(self):
        """测试检查系统状态 - 无告警"""
        config = SystemMonitorConfig(
            cpu_threshold=90.0,
            memory_threshold=90.0,
            disk_threshold=90.0
        )
        
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 40.0},
            'disk': {'percent': 30.0}
        }
        
        alerts = self.alert_manager.check_system_status(stats, config)
        self.assertEqual(len(alerts), 0)

    def test_check_system_status_cpu_alert(self):
        """测试检查系统状态 - CPU告警"""
        config = SystemMonitorConfig(cpu_threshold=80.0)
        
        stats = {
            'cpu': {'percent': 85.0},
            'memory': {'percent': 40.0},
            'disk': {'percent': 30.0}
        }
        
        alerts = self.alert_manager.check_system_status(stats, config)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'cpu')
        self.assertEqual(alerts[0]['level'], 'critical')
        self.assertIn('High CPU usage', alerts[0]['message'])

    def test_check_system_status_memory_alert(self):
        """测试检查系统状态 - 内存告警"""
        config = SystemMonitorConfig(memory_threshold=80.0)
        
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 85.0},
            'disk': {'percent': 30.0}
        }
        
        alerts = self.alert_manager.check_system_status(stats, config)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'memory')
        self.assertEqual(alerts[0]['level'], 'critical')

    def test_check_system_status_disk_alert(self):
        """测试检查系统状态 - 磁盘告警"""
        config = SystemMonitorConfig(disk_threshold=80.0)
        
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 40.0},
            'disk': {'percent': 85.0}
        }
        
        alerts = self.alert_manager.check_system_status(stats, config)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'disk')
        self.assertEqual(alerts[0]['level'], 'warning')

    def test_check_system_status_multiple_alerts(self):
        """测试检查系统状态 - 多个告警"""
        config = SystemMonitorConfig(
            cpu_threshold=80.0,
            memory_threshold=80.0,
            disk_threshold=80.0
        )
        
        stats = {
            'cpu': {'percent': 85.0},
            'memory': {'percent': 85.0},
            'disk': {'percent': 85.0}
        }
        
        alerts = self.alert_manager.check_system_status(stats, config)
        self.assertEqual(len(alerts), 3)

    def test_trigger_alerts(self):
        """测试触发告警"""
        alerts = [
            {'type': 'cpu', 'level': 'critical', 'message': 'High CPU'},
            {'type': 'memory', 'level': 'warning', 'message': 'High memory'}
        ]
        
        self.alert_manager.trigger_alerts(alerts)
        
        # 验证所有处理器被调用
        for handler in self.mock_handlers:
            self.assertEqual(handler.call_count, 2)

    def test_trigger_single_alert(self):
        """测试触发单个告警"""
        alert = {'type': 'cpu', 'level': 'critical', 'message': 'High CPU'}
        
        self.alert_manager._trigger_single_alert(alert)
        
        # 验证所有处理器被调用
        for handler in self.mock_handlers:
            handler.assert_called_once_with('system', alert)

    def test_trigger_alert_handler_exception(self):
        """测试告警处理器异常"""
        # 设置一个处理器抛出异常
        failing_handler = Mock(side_effect=Exception("Handler failed"))
        alert_manager = SystemAlertManager([failing_handler, Mock()])
        
        alert = {'type': 'cpu', 'message': 'test'}
        
        # 不应该抛出异常
        alert_manager._trigger_single_alert(alert)


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestMetricsCalculator(unittest.TestCase):
    """测试指标计算器"""

    def setUp(self):
        """测试前准备"""
        self.calculator = MetricsCalculator()

    def test_calculator_initialization(self):
        """测试计算器初始化"""
        self.assertIsNotNone(self.calculator)

    def test_calculate_metrics(self):
        """测试计算指标"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 40.0},
            'network': {'bytes_sent': 1000, 'bytes_recv': 2000},
            'process': {'count': 50}
        }
        
        metrics = self.calculator.calculate_metrics(stats)
        
        self.assertIsNotNone(metrics)
        # 验证metrics的基本属性
        self.assertIsNotNone(metrics.timestamp)


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestMonitorEngine(unittest.TestCase):
    """测试监控引擎"""

    def setUp(self):
        """测试前准备"""
        self.config = SystemMonitorConfig(check_interval=0.1)  # 短间隔用于测试
        self.mock_metrics_calculator = Mock()
        self.mock_alert_manager = Mock()
        self.mock_info_collector = Mock()
        
        self.engine = MonitorEngine(
            self.config,
            self.mock_metrics_calculator,
            self.mock_alert_manager
        )
        
        # 设置模拟返回值
        self.mock_info_collector.get_system_stats.return_value = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 40.0}
        }
        self.mock_alert_manager.check_system_status.return_value = []

    def test_engine_initialization(self):
        """测试引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertFalse(self.engine._monitoring_active)

    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 启动监控
        self.engine.start_monitoring(self.mock_info_collector)
        self.assertTrue(self.engine._monitoring_active)
        self.assertIsNotNone(self.engine._monitor_thread)
        
        # 等待一小段时间
        time.sleep(0.1)
        
        # 停止监控
        self.engine.stop_monitoring()
        self.assertFalse(self.engine._monitoring_active)

    def test_start_already_running(self):
        """测试重复启动监控"""
        self.engine.start_monitoring(self.mock_info_collector)
        initial_thread = self.engine._monitor_thread
        
        # 再次启动应该不创建新线程
        self.engine.start_monitoring(self.mock_info_collector)
        self.assertEqual(self.engine._monitor_thread, initial_thread)
        
        self.engine.stop_monitoring()

    def test_stop_not_running(self):
        """测试停止未运行的监控"""
        # 应该不会抛出异常
        self.engine.stop_monitoring()

    def test_get_recent_stats(self):
        """测试获取最近统计"""
        stats = self.engine.get_recent_stats()
        self.assertIsInstance(stats, list)

    def test_get_recent_stats_with_limit(self):
        """测试获取有限数量的最近统计"""
        stats = self.engine.get_recent_stats(count=5)
        self.assertIsInstance(stats, list)
        self.assertLessEqual(len(stats), 5)


@unittest.skipUnless(IMPORTS_AVAILABLE, "system_monitor模块导入失败")
class TestSystemMonitor(unittest.TestCase):
    """测试系统监控器主类"""

    def setUp(self):
        """测试前准备"""
        self.config = SystemMonitorConfig(check_interval=0.1)
        self.monitor = SystemMonitor(self.config)

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertIsNotNone(self.monitor.config)
        self.assertIsNotNone(self.monitor.info_collector)
        self.assertIsNotNone(self.monitor.alert_manager)
        self.assertIsNotNone(self.monitor.engine)

    def test_start_stop_system_monitoring(self):
        """测试启动和停止系统监控"""
        # 启动监控
        self.monitor.start()
        self.assertTrue(self.monitor.engine._monitoring_active)
        
        time.sleep(0.1)
        
        # 停止监控
        self.monitor.stop()
        self.assertFalse(self.monitor.engine._monitoring_active)

    def test_get_system_info(self):
        """测试获取系统信息"""
        info = self.monitor.get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn('cpu', info)
        self.assertIn('memory', info)

    def test_get_alerts_summary(self):
        """测试获取告警摘要"""
        summary = self.monitor.get_alerts_summary()
        self.assertIsInstance(summary, dict)

    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        status = self.monitor.get_monitoring_status()
        self.assertIsInstance(status, dict)
        self.assertIn('active', status)
        self.assertIn('start_time', status)

    def test_set_alert_thresholds(self):
        """测试设置告警阈值"""
        self.monitor.set_alert_thresholds({
            'cpu': 80.0,
            'memory': 85.0,
            'disk': 95.0
        })
        
        # 验证配置已更新
        self.assertEqual(self.monitor.config.cpu_threshold, 80.0)
        self.assertEqual(self.monitor.config.memory_threshold, 85.0)
        self.assertEqual(self.monitor.config.disk_threshold, 95.0)

    def test_add_alert_handler(self):
        """测试添加告警处理器"""
        def test_handler(source, data):
            pass
        
        initial_count = len(self.monitor.alert_manager.alert_handlers)
        self.monitor.add_alert_handler(test_handler)
        
        self.assertEqual(len(self.monitor.alert_manager.alert_handlers), initial_count + 1)

    def test_monitor_graceful_shutdown(self):
        """测试监控器优雅关闭"""
        self.monitor.start()
        time.sleep(0.1)
        
        # 调用shutdown应该停止监控
        if hasattr(self.monitor, 'shutdown'):
            self.monitor.shutdown()
            self.assertFalse(self.monitor.engine._monitoring_active)
        else:
            # 如果没有shutdown方法，使用stop
            self.monitor.stop()
            self.assertFalse(self.monitor.engine._monitoring_active)


if __name__ == '__main__':
    unittest.main()