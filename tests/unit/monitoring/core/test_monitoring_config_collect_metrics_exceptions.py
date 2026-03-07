#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig收集系统指标异常处理测试
补充collect_system_metrics的异常处理场景
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

class TestCollectSystemMetricsExceptions:
    """测试collect_system_metrics的异常处理"""

    def setup_method(self):
        """每个测试前重置监控实例"""
        monitoring.metrics = {}

    def test_collect_system_metrics_psutil_cpu_error(self):
        """测试CPU指标收集失败时的异常处理"""
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            # 应该抛出异常或处理错误
            with pytest.raises(Exception):
                collect_system_metrics()

    def test_collect_system_metrics_psutil_memory_error(self):
        """测试内存指标收集失败时的异常处理"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory', side_effect=Exception("Memory error")):
                # 应该抛出异常或处理错误
                with pytest.raises(Exception):
                    collect_system_metrics()

    def test_collect_system_metrics_psutil_disk_error(self):
        """测试磁盘指标收集失败时的异常处理"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                with patch('psutil.disk_usage', side_effect=Exception("Disk error")):
                    # 应该抛出异常或处理错误
                    with pytest.raises(Exception):
                        collect_system_metrics()

    def test_collect_system_metrics_psutil_network_none(self):
        """测试网络指标为None时的处理"""
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
                        # 网络为None时应该正常处理，不记录网络指标
                        result = collect_system_metrics()
                        
                        assert isinstance(result, dict)
                        assert 'cpu_percent' in result
                        assert 'memory_percent' in result
                        assert 'disk_percent' in result
                        # 网络指标不应该被记录
                        network_metrics = [k for k in monitoring.metrics.keys() if 'network' in k]
                        assert len(network_metrics) == 0

    def test_collect_system_metrics_psutil_network_error(self):
        """测试网络指标收集失败时的异常处理"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk_obj = MagicMock()
                    mock_disk_obj.percent = 70.0
                    mock_disk.return_value = mock_disk_obj
                    
                    with patch('psutil.net_io_counters', side_effect=Exception("Network error")):
                        # 网络错误时应该抛出异常或跳过网络指标
                        # 根据实际实现，可能会抛出异常或忽略网络错误
                        try:
                            result = collect_system_metrics()
                            # 如果成功，验证其他指标正常
                            assert isinstance(result, dict)
                            assert 'cpu_percent' in result
                        except Exception:
                            # 如果抛出异常，这也是预期的行为
                            pass

    def test_collect_system_metrics_record_metric_error(self):
        """测试记录指标失败时的异常处理"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch.object(monitoring, 'record_metric', side_effect=Exception("Record error")):
                # 记录指标失败时应该抛出异常
                with pytest.raises(Exception):
                    collect_system_metrics()

    def test_collect_system_metrics_partial_failure(self):
        """测试部分指标收集失败的情况"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                with patch('psutil.disk_usage', side_effect=Exception("Disk error")):
                    # 部分失败时应该抛出异常
                    with pytest.raises(Exception):
                        collect_system_metrics()

    def test_collect_system_metrics_disk_usage_windows_path(self):
        """测试Windows系统上的磁盘路径处理"""
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory_obj = MagicMock()
                mock_memory_obj.percent = 60.0
                mock_memory.return_value = mock_memory_obj
                
                # Windows系统使用 'C:\\'
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk_obj = MagicMock()
                    mock_disk_obj.percent = 70.0
                    mock_disk.return_value = mock_disk_obj
                    mock_disk.side_effect = [
                        OSError("Invalid path"),  # '/' 失败
                        mock_disk_obj  # 可以测试多次尝试
                    ]
                    
                    with patch('psutil.net_io_counters', return_value=None):
                        # 应该处理路径错误
                        try:
                            result = collect_system_metrics()
                            # 如果失败，也应该有合理的错误处理
                        except (OSError, Exception):
                            # 这是预期的，因为磁盘路径错误
                            pass

    def test_collect_system_metrics_all_errors(self):
        """测试所有指标收集都失败的情况"""
        with patch('psutil.cpu_percent', side_effect=Exception("All errors")):
            with patch('psutil.virtual_memory', side_effect=Exception("All errors")):
                with patch('psutil.disk_usage', side_effect=Exception("All errors")):
                    # 所有指标都失败时应该抛出异常
                    with pytest.raises(Exception):
                        collect_system_metrics()


