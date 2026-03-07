#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig收集系统指标网络部分测试
补充collect_system_metrics中网络指标为空的场景
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

class TestCollectSystemMetricsNetwork:
    """测试collect_system_metrics的网络指标处理"""

    def test_collect_system_metrics_no_network(self):
        """测试网络指标为空的情况（行179 if network:）"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk_obj = MagicMock()
                    mock_disk_obj.percent = 70.0
                    mock_disk.return_value = mock_disk_obj
                    
                    with patch('psutil.net_io_counters', return_value=None):
                        monitoring.metrics = {}
                        
                        result = collect_system_metrics()
                        
                        # 应该正常返回基本指标
                        assert 'cpu_percent' in result
                        assert 'memory_percent' in result
                        assert 'disk_percent' in result
                        
                        # 网络指标不应该被记录
                        assert 'network_bytes_sent' not in monitoring.metrics
                        assert 'network_bytes_recv' not in monitoring.metrics

    def test_collect_system_metrics_with_network(self):
        """测试有网络指标的情况"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk_obj = MagicMock()
                    mock_disk_obj.percent = 70.0
                    mock_disk.return_value = mock_disk_obj
                    
                    with patch('psutil.net_io_counters') as mock_network:
                        mock_network_obj = MagicMock()
                        mock_network_obj.bytes_sent = 1000
                        mock_network_obj.bytes_recv = 2000
                        mock_network.return_value = mock_network_obj
                        
                        monitoring.metrics = {}
                        
                        result = collect_system_metrics()
                        
                        # 验证网络指标被记录
                        assert 'network_bytes_sent' in monitoring.metrics
                        assert 'network_bytes_recv' in monitoring.metrics
                        assert monitoring.metrics['network_bytes_sent'][0]['value'] == 1000
                        assert monitoring.metrics['network_bytes_recv'][0]['value'] == 2000



