#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig收集系统指标完整测试
补充collect_system_metrics的所有分支和场景
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

class TestCollectSystemMetricsComplete:
    """测试collect_system_metrics的完整场景"""

    def test_collect_system_metrics_full_flow_with_network(self):
        """测试完整的系统指标收集流程（包含网络）"""
        try:
            with patch('psutil.cpu_percent', return_value=55.5) as mock_cpu:
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory_obj = MagicMock()
                    mock_memory_obj.percent = 65.5
                    mock_memory.return_value = mock_memory_obj
                    
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk_obj = MagicMock()
                        mock_disk_obj.percent = 75.5
                        mock_disk.return_value = mock_disk_obj
                        
                        with patch('psutil.net_io_counters') as mock_network:
                            mock_network_obj = MagicMock()
                            mock_network_obj.bytes_sent = 1234567
                            mock_network_obj.bytes_recv = 2345678
                            mock_network.return_value = mock_network_obj
                            
                            # 重置监控实例
                            monitoring.metrics = {}
                            
                            result = collect_system_metrics()
                            
                            # 验证返回值
                            assert isinstance(result, dict)
                            assert 'cpu_percent' in result
                            assert 'memory_percent' in result
                            assert 'disk_percent' in result
                            assert result['cpu_percent'] == 55.5
                            assert result['memory_percent'] == 65.5
                            assert result['disk_percent'] == 75.5
                            
                            # 验证指标被记录
                            assert 'cpu_usage' in monitoring.metrics
                            assert 'memory_usage' in monitoring.metrics
                            assert 'disk_usage' in monitoring.metrics
                            assert 'network_bytes_sent' in monitoring.metrics
                            assert 'network_bytes_recv' in monitoring.metrics
        except ImportError:
            pytest.skip("psutil not available")

    def test_collect_system_metrics_without_network(self):
        """测试系统指标收集（无网络数据）"""
        try:
            with patch('psutil.cpu_percent', return_value=45.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory_obj = MagicMock()
                    mock_memory_obj.percent = 55.0
                    mock_memory.return_value = mock_memory_obj
                    
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk_obj = MagicMock()
                        mock_disk_obj.percent = 65.0
                        mock_disk.return_value = mock_disk_obj
                        
                        with patch('psutil.net_io_counters', return_value=None):
                            monitoring.metrics = {}
                            
                            result = collect_system_metrics()
                            
                            assert isinstance(result, dict)
                            assert 'cpu_percent' in result
                            assert 'memory_percent' in result
                            assert 'disk_percent' in result
                            
                            # 验证网络指标未被记录
                            assert 'network_bytes_sent' not in monitoring.metrics
                            assert 'network_bytes_recv' not in monitoring.metrics
        except ImportError:
            pytest.skip("psutil not available")

    def test_collect_system_metrics_network_is_falsy(self):
        """测试网络数据为falsy值的情况"""
        try:
            with patch('psutil.cpu_percent', return_value=50.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory_obj = MagicMock()
                    mock_memory_obj.percent = 60.0
                    mock_memory.return_value = mock_memory_obj
                    
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk_obj = MagicMock()
                        mock_disk_obj.percent = 70.0
                        mock_disk.return_value = mock_disk_obj
                        
                        # 测试网络为False的情况
                        with patch('psutil.net_io_counters', return_value=False):
                            monitoring.metrics = {}
                            
                            result = collect_system_metrics()
                            
                            assert isinstance(result, dict)
                            # 网络指标不应该被记录（因为if network:条件）
                            assert 'network_bytes_sent' not in monitoring.metrics or len(monitoring.metrics.get('network_bytes_sent', [])) == 0
        except ImportError:
            pytest.skip("psutil not available")

    def test_collect_system_metrics_edge_cases(self):
        """测试边界情况"""
        try:
            # 测试极端值
            with patch('psutil.cpu_percent', return_value=0.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory_obj = MagicMock()
                    mock_memory_obj.percent = 0.0
                    mock_memory.return_value = mock_memory_obj
                    
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk_obj = MagicMock()
                        mock_disk_obj.percent = 0.0
                        mock_disk.return_value = mock_disk_obj
                        
                        with patch('psutil.net_io_counters', return_value=None):
                            monitoring.metrics = {}
                            
                            result = collect_system_metrics()
                            
                            assert result['cpu_percent'] == 0.0
                            assert result['memory_percent'] == 0.0
                            assert result['disk_percent'] == 0.0
        except ImportError:
            pytest.skip("psutil not available")

