#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
资源管理系统核心组件测试
为已实现的资源管理组件创建基础单元测试，提高覆盖率
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta


class TestResourceManager(unittest.TestCase):
    """Test resource manager"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.resource.resource_manager import ResourceManager
        self.manager = ResourceManager()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        # 验证基本属性存在
        self.assertTrue(hasattr(self.manager, '_monitoring'))
        self.assertTrue(hasattr(self.manager, '_resource_history'))
        self.assertTrue(hasattr(self.manager, '_lock'))

    def test_start_monitoring(self):
        """测试启动监控"""
        # 停止当前的监控
        self.manager._monitoring = False
        if self.manager._monitor_thread:
            self.manager._monitor_thread = None

        # 重新启动监控
        with patch('threading.Thread') as mock_thread:
            self.manager.start_monitoring()

            # 验证监控线程被创建
            mock_thread.assert_called_once()
            args, kwargs = mock_thread.call_args
            self.assertEqual(kwargs['target'], self.manager._monitor_resources)
            self.assertTrue(kwargs['daemon'])

    def test_get_current_usage(self):
        """测试获取当前资源使用情况"""
        with patch('psutil.cpu_percent') as mock_cpu, \
            patch('psutil.virtual_memory') as mock_memory, \
            patch('psutil.disk_usage') as mock_disk:

            # 设置模拟返回值
            mock_cpu.return_value = 45.5
            mock_memory.return_value = Mock(percent=60.2, used=1024*1024*1024, total=2*1024*1024*1024)
            mock_disk.return_value = Mock(percent=75.8, used=50*1024*1024*1024, total=200*1024*1024*1024)

            usage = self.manager.get_current_usage()

        # 验证返回结果
        self.assertIn('cpu_percent', usage)
        self.assertIn('memory_percent', usage)
        self.assertIn('disk_percent', usage)
        self.assertEqual(usage['cpu_percent'], 45.5)
        self.assertEqual(usage['memory_percent'], 60.2)

    def test_get_usage_history(self):
        """测试获取资源历史"""
        # 添加一些测试历史数据
        test_data = {'timestamp': datetime.now().isoformat(), 'cpu_percent': 50.0}
        self.manager._resource_history.append(test_data)

        history = self.manager.get_usage_history()

        # 验证历史数据
        self.assertIsInstance(history, dict)
        self.assertIn('history', history)
        self.assertIn('count', history)


class TestSystemMonitor(unittest.TestCase):
    """Test system monitor"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.resource.system_monitor import SystemMonitor
        self.monitor = SystemMonitor()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.monitor)
        # 验证基本属性存在
        self.assertTrue(hasattr(self.monitor, 'check_interval'))
        self.assertTrue(hasattr(self.monitor, 'alert_handlers'))
        self.assertTrue(hasattr(self.monitor, '_monitoring'))
        # SystemMonitor可能没有_metrics属性，检查实际存在的属性
        self.assertTrue(hasattr(self.monitor, '_stats') or hasattr(self.monitor, '_metrics'))

    def test_get_system_info(self):
        """测试获取系统信息"""
        with patch('platform.system') as mock_system, \
            patch('platform.release') as mock_release, \
            patch('platform.processor') as mock_processor:

            mock_system.return_value = 'Linux'
            mock_release.return_value = '5.4.0'
            mock_processor.return_value = 'Intel Core i7'

            info = self.monitor._get_system_info()

        # 验证返回结果
        self.assertIn('system', info)
        self.assertIn('release', info)
        self.assertIn('processor', info)

    def test_start_monitoring(self):
        """测试启动监控"""
        with patch('threading.Thread') as mock_thread:
            self.monitor.start_monitoring()

        # 验证监控已启动
        self.assertTrue(self.monitor._monitoring)
        mock_thread.assert_called_once()

    def test_stop_monitoring(self):
        """测试停止监控"""
        # 先启动监控
        self.monitor._monitoring = True
        mock_thread = Mock()
        self.monitor._monitor_thread = mock_thread

        self.monitor.stop_monitoring()

        # 验证监控已停止
        self.assertFalse(self.monitor._monitoring)

    def test_collect_system_stats(self):
        """测试收集系统统计数据"""
        with patch('psutil.cpu_percent') as mock_cpu, \
            patch('psutil.virtual_memory') as mock_memory, \
            patch('psutil.disk_usage') as mock_disk, \
            patch('psutil.net_io_counters') as mock_net, \
            patch('psutil.pids') as mock_pids:

            mock_cpu.return_value = 45.5
            mock_memory.return_value = Mock(total=8*1024**3, available=4*1024**3, used=4*1024**3, percent=50.0)
            mock_disk.return_value = Mock(total=500*1024**3, used=250*1024**3, free=250*1024**3, percent=50.0)
            mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
            mock_pids.return_value = [1, 2, 3, 4, 5]

            stats = self.monitor._collect_system_stats()

        # 验证返回结果
        self.assertIn('cpu', stats)
        self.assertIn('memory', stats)
        self.assertIn('disk', stats)
        self.assertEqual(stats['cpu']['percent'], 45.5)


class TestResourceOptimization(unittest.TestCase):
    """测试资源优化器"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.resource.resource_optimization import ResourceOptimizer
        self.optimizer = ResourceOptimizer()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.optimizer)
        # 验证基本属性存在
        self.assertTrue(hasattr(self.optimizer, 'monitoring_data'))
        self.assertTrue(hasattr(self.optimizer, 'optimization_suggestions'))

    def test_get_system_resources(self):
        """测试获取系统资源"""
        resources = self.optimizer.get_system_resources()

        # 验证返回结果
        self.assertIsInstance(resources, dict)
        # 检查实际返回的字段
        self.assertIn('cpu_percent', resources)

    def test_detect_memory_leaks(self):
        """测试内存泄漏检测"""
        result = self.optimizer.detect_memory_leaks()

        # 验证结果
        self.assertIsInstance(result, list)

    def test_generate_optimization_report(self):
        """测试生成优化报告"""
        report = self.optimizer.generate_optimization_report()

        # 验证结果
        self.assertIsInstance(report, dict)
        self.assertIn('timestamp', report)


class TestTaskScheduler(unittest.TestCase):
    """测试任务调度器"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.resource.task_scheduler import TaskScheduler
        self.scheduler = TaskScheduler()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.scheduler)
        # 验证基本属性存在
        self.assertTrue(hasattr(self.scheduler, 'tasks'))
        self.assertTrue(hasattr(self.scheduler, 'running'))

    def test_submit_task(self):
        """测试提交任务"""
        def test_task():
            return "executed"

        task_id = self.scheduler.submit_task("test_task", test_task)

        # 验证任务已添加
        self.assertIn(task_id, self.scheduler.tasks)

    def test_start_scheduler(self):
        """测试启动调度器"""
        self.scheduler.start()

        # 验证调度器已启动
        self.assertTrue(self.scheduler.running)
        self.assertIsNotNone(self.scheduler.workers)

    def test_stop_scheduler(self):
        """测试停止调度器"""
        # 先启动调度器
        self.scheduler.start()

        # 停止调度器
        self.scheduler.stop()

        # 验证调度器已停止
        self.assertFalse(self.scheduler.running)

    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.scheduler.get_stats()

        # 验证统计信息
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tasks', stats)


class TestUnifiedMonitor(unittest.TestCase):
    """测试统一监控器"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.resource.unified_monitor_adapter import UnifiedMonitor
        self.monitor = UnifiedMonitor()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.monitor)
        # 验证基本属性存在
        self.assertTrue(hasattr(self.monitor, 'is_running'))
        self.assertTrue(hasattr(self.monitor, '_metrics'))

    def test_start_stop(self):
        """测试启动和停止"""
        # 测试启动
        self.monitor.start()
        self.assertTrue(self.monitor.is_running)

        # 测试停止
        self.monitor.stop()
        self.assertFalse(self.monitor.is_running)

    def test_get_metrics(self):
        """测试获取指标"""
        # 添加一些测试指标
        self.monitor._metrics['test'] = [{'value': 50.0, 'timestamp': datetime.now().isoformat()}]

        # 获取指标
        metrics = self.monitor.get_metrics('test')

        # 验证结果
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 1)

    def test_get_all_metrics(self):
        """测试获取所有指标"""
        # 添加一些测试指标
        self.monitor._metrics['cpu'] = [{'value': 50.0}]
        self.monitor._metrics['memory'] = [{'value': 60.0}]

        # 获取所有指标
        metrics = self.monitor.get_metrics(None)

        # 验证结果
        self.assertIsInstance(metrics, list)


if __name__ == '__main__':
    unittest.main()
