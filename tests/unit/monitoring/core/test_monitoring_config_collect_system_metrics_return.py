#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig收集系统指标返回值测试
补充collect_system_metrics函数返回值结构和字段的详细测试
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
import importlib
from pathlib import Path
import pytest

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
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    collect_system_metrics = getattr(core_monitoring_config_module, 'collect_system_metrics', None)
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    if collect_system_metrics is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestCollectSystemMetricsReturn:
    """测试collect_system_metrics函数返回值"""

    @pytest.fixture(autouse=True)
    def reset_monitoring(self):
        """每次测试前重置monitoring实例"""
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []
        yield
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_returns_dict(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试函数返回字典"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        assert isinstance(result, dict)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_has_required_keys(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试返回值包含必需的键"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        assert 'cpu_percent' in result
        assert 'memory_percent' in result
        assert 'disk_percent' in result

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_cpu_percent(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试cpu_percent值"""
        mock_cpu.return_value = 75.5
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        assert result['cpu_percent'] == 75.5

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_memory_percent(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试memory_percent值"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=85.5)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        assert result['memory_percent'] == 85.5

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_disk_percent(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试disk_percent值"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=90.5)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        assert result['disk_percent'] == 90.5

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_all_metrics_recorded(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试所有指标都被记录"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        collect_system_metrics()
        
        # 验证所有指标都被记录
        assert 'cpu_usage' in monitoring.metrics
        assert 'memory_usage' in monitoring.metrics
        assert 'disk_usage' in monitoring.metrics
        assert 'network_bytes_sent' in monitoring.metrics
        assert 'network_bytes_recv' in monitoring.metrics

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_cpu_tag(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试CPU指标包含unit tag"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        collect_system_metrics()
        
        cpu_metric = monitoring.metrics['cpu_usage'][0]
        assert cpu_metric['tags']['unit'] == 'percent'

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_memory_tag(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试内存指标包含unit tag"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        collect_system_metrics()
        
        memory_metric = monitoring.metrics['memory_usage'][0]
        assert memory_metric['tags']['unit'] == 'percent'

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_disk_tag(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试磁盘指标包含unit tag"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        collect_system_metrics()
        
        disk_metric = monitoring.metrics['disk_usage'][0]
        assert disk_metric['tags']['unit'] == 'percent'

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_network_tags(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试网络指标包含unit tag"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        collect_system_metrics()
        
        sent_metric = monitoring.metrics['network_bytes_sent'][0]
        recv_metric = monitoring.metrics['network_bytes_recv'][0]
        assert sent_metric['tags']['unit'] == 'bytes'
        assert recv_metric['tags']['unit'] == 'bytes'

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_network_bytes_values(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试网络字节数记录"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=12345, bytes_recv=67890)
        
        collect_system_metrics()
        
        sent_metric = monitoring.metrics['network_bytes_sent'][0]
        recv_metric = monitoring.metrics['network_bytes_recv'][0]
        assert sent_metric['value'] == 12345
        assert recv_metric['value'] == 67890

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_network_none(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试网络指标为None时不记录网络指标"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = None
        
        collect_system_metrics()
        
        # 应该仍然记录其他指标
        assert 'cpu_usage' in monitoring.metrics
        assert 'memory_usage' in monitoring.metrics
        assert 'disk_usage' in monitoring.metrics
        # 网络指标不应该被记录
        assert 'network_bytes_sent' not in monitoring.metrics
        assert 'network_bytes_recv' not in monitoring.metrics

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_return_structure_complete(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试返回值结构完整性"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        # 验证所有必需字段存在且为数值类型
        assert isinstance(result['cpu_percent'], (int, float))
        assert isinstance(result['memory_percent'], (int, float))
        assert isinstance(result['disk_percent'], (int, float))

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_metrics_return_only_three_keys(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试返回值只包含三个键"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        
        result = collect_system_metrics()
        
        # 返回值应该只包含三个键
        assert len(result) == 3
        assert set(result.keys()) == {'cpu_percent', 'memory_percent', 'disk_percent'}

