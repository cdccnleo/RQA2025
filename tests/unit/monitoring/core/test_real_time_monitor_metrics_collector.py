#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor MetricsCollector测试
补充MetricsCollector类的方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
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
    core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
    MetricsCollector = getattr(core_real_time_monitor_module, 'MetricsCollector', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if MetricsCollector is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricsCollector:
    """测试MetricsCollector类"""

    @pytest.fixture
    def collector(self):
        """创建MetricsCollector实例"""
        return MetricsCollector()

    def test_init(self, collector):
        """测试初始化"""
        assert collector.metrics == {}
        assert collector.collectors == {}
        assert collector.collection_interval == 5
        assert collector._running == False
        assert collector._thread is None

    def test_register_collector(self, collector):
        """测试注册指标收集器"""
        def mock_collector():
            return {'test_metric': 1.0}
        
        collector.register_collector('test_collector', mock_collector)
        
        assert 'test_collector' in collector.collectors
        assert collector.collectors['test_collector'] == mock_collector

    def test_register_collector_multiple(self, collector):
        """测试注册多个收集器"""
        def collector1():
            return {'metric1': 1.0}
        def collector2():
            return {'metric2': 2.0}
        
        collector.register_collector('collector1', collector1)
        collector.register_collector('collector2', collector2)
        
        assert len(collector.collectors) == 2

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_collect_system_metrics_success(self, mock_pids, mock_net, mock_disk, mock_mem, mock_cpu, collector):
        """测试收集系统指标成功"""
        mock_cpu.return_value = 50.0
        mock_mem.return_value = Mock(percent=60.0, used=1024*1024*1024, available=512*1024*1024)
        mock_disk.return_value = Mock(percent=70.0)
        mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        mock_pids.return_value = [1, 2, 3]
        
        with patch('psutil.getloadavg', return_value=(1.0, 2.0, 3.0)):
            metrics = collector.collect_system_metrics()
            
            assert metrics['cpu_percent'] == 50.0
            assert metrics['memory_percent'] == 60.0
            assert metrics['disk_usage_percent'] == 70.0
            assert metrics['network_bytes_sent'] == 1000
            assert metrics['network_bytes_recv'] == 2000
            assert metrics['num_processes'] == 3

    @patch('psutil.cpu_percent')
    def test_collect_system_metrics_exception(self, mock_cpu, collector):
        """测试收集系统指标异常处理"""
        mock_cpu.side_effect = Exception("Test error")
        
        metrics = collector.collect_system_metrics()
        
        assert metrics == {}

    @patch('psutil.getloadavg', side_effect=AttributeError)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_collect_system_metrics_no_loadavg(self, mock_pids, mock_net, mock_disk, mock_mem, mock_cpu, mock_loadavg, collector):
        """测试收集系统指标无loadavg"""
        mock_cpu.return_value = 50.0
        mock_mem.return_value = Mock(percent=60.0, used=1024*1024*1024, available=512*1024*1024)
        mock_disk.return_value = Mock(percent=70.0)
        mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        mock_pids.return_value = [1, 2, 3]
        
        metrics = collector.collect_system_metrics()
        
        assert metrics['load_average_1min'] == 0

    @patch('psutil.Process')
    def test_collect_application_metrics_success(self, mock_process_class, collector):
        """测试收集应用指标成功"""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*100, vms=1024*1024*200)
        mock_process.num_threads.return_value = 5
        mock_process.num_fds = Mock(return_value=10)
        mock_process.cpu_times.return_value = Mock(user=1.0, system=0.5)
        mock_process_class.return_value = mock_process
        
        metrics = collector.collect_application_metrics()
        
        assert 'app_cpu_percent' in metrics
        assert 'app_memory_rss_mb' in metrics
        assert 'app_memory_vms_mb' in metrics
        assert 'app_num_threads' in metrics

    @patch('psutil.Process')
    def test_collect_application_metrics_exception(self, mock_process_class, collector):
        """测试收集应用指标异常处理"""
        mock_process_class.side_effect = Exception("Test error")
        
        metrics = collector.collect_application_metrics()
        
        assert metrics == {}

    def test_collect_business_metrics_default(self, collector):
        """测试收集业务指标默认值"""
        metrics = collector.collect_business_metrics()
        
        assert metrics['requests_total'] == 0
        assert metrics['requests_per_second'] == 0.0
        assert metrics['errors_total'] == 0
        assert metrics['error_rate'] == 0.0
        assert metrics['avg_response_time_ms'] == 0.0

    def test_update_business_metric_request(self, collector):
        """测试更新业务指标-请求"""
        collector.update_business_metric('request', 1.0)
        
        metrics = collector.collect_business_metrics()
        assert metrics['requests_total'] == 1
        
        collector.update_business_metric('request', 1.0)
        metrics = collector.collect_business_metrics()
        assert metrics['requests_total'] == 2

    def test_update_business_metric_error(self, collector):
        """测试更新业务指标-错误"""
        collector.update_business_metric('error', 1.0)
        
        metrics = collector.collect_business_metrics()
        assert metrics['errors_total'] == 1

    def test_update_business_metric_response_time(self, collector):
        """测试更新业务指标-响应时间"""
        collector.update_business_metric('response_time', 100.0)
        
        metrics = collector.collect_business_metrics()
        assert metrics['avg_response_time_ms'] == 100.0
        
        collector.update_business_metric('response_time', 200.0)
        metrics = collector.collect_business_metrics()
        # 平均值应该是 (100 + 200) / 2 = 150
        assert metrics['avg_response_time_ms'] == 150.0

    @patch.object(MetricsCollector, 'collect_system_metrics')
    @patch.object(MetricsCollector, 'collect_application_metrics')
    @patch.object(MetricsCollector, 'collect_business_metrics')
    def test_collect_all_metrics(self, mock_business, mock_app, mock_system, collector):
        """测试收集所有指标"""
        mock_system.return_value = {'cpu_percent': 50.0}
        mock_app.return_value = {'app_cpu_percent': 10.0}
        mock_business.return_value = {'requests_total': 100}
        
        all_metrics = collector.collect_all_metrics()
        
        assert 'cpu_percent' in all_metrics
        assert isinstance(all_metrics['cpu_percent'], MetricData)
        assert all_metrics['cpu_percent'].tags['type'] == 'system'
        
        assert 'app_cpu_percent' in all_metrics
        assert all_metrics['app_cpu_percent'].tags['type'] == 'application'
        
        assert 'requests_total' in all_metrics
        assert all_metrics['requests_total'].tags['type'] == 'business'

    @patch.object(MetricsCollector, 'collect_system_metrics')
    @patch.object(MetricsCollector, 'collect_application_metrics')
    def test_collect_all_metrics_empty(self, mock_app, mock_system, collector):
        """测试收集所有指标为空"""
        mock_system.return_value = {}
        mock_app.return_value = {}
        
        all_metrics = collector.collect_all_metrics()
        
        # 应该只有business metrics（如果有的话）
        assert isinstance(all_metrics, dict)

    def test_collect_all_metrics_with_custom_collector(self, collector):
        """测试收集所有指标包含自定义收集器"""
        def custom_collector():
            return {'custom_metric': 42.0}
        
        collector.register_collector('custom', custom_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                assert 'custom_custom_metric' in all_metrics
                assert all_metrics['custom_custom_metric'].tags['type'] == 'custom'
                assert all_metrics['custom_custom_metric'].tags['collector'] == 'custom'

    def test_collect_all_metrics_custom_collector_exception(self, collector):
        """测试自定义收集器异常处理"""
        def failing_collector():
            raise Exception("Collector error")
        
        collector.register_collector('failing', failing_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                # 不应该抛出异常
                all_metrics = collector.collect_all_metrics()
                assert isinstance(all_metrics, dict)

    @patch('threading.Thread')
    def test_start_collection_normal(self, mock_thread_class, collector):
        """测试启动收集正常"""
        collector._running = False
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        collector.start_collection()
        
        assert collector._running == True
        mock_thread.start.assert_called_once()

    def test_start_collection_already_running(self, collector):
        """测试启动收集已在运行"""
        collector._running = True
        original_thread = collector._thread
        
        collector.start_collection()
        
        # 应该不改变状态
        assert collector._running == True
        assert collector._thread == original_thread

    def test_stop_collection_normal(self, collector):
        """测试停止收集正常"""
        collector._running = True
        mock_thread = Mock()
        mock_thread.join = Mock()
        collector._thread = mock_thread
        
        collector.stop_collection()
        
        assert collector._running == False
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_collection_no_thread(self, collector):
        """测试停止收集无线程"""
        collector._running = True
        collector._thread = None
        
        # 不应该抛出异常
        collector.stop_collection()
        assert collector._running == False

