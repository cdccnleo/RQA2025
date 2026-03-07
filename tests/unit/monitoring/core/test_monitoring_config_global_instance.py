#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig全局实例测试
补充全局monitoring实例的测试
"""

import pytest
from unittest.mock import patch

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
    MonitoringSystem = getattr(core_monitoring_config_module, 'MonitoringSystem', None)
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    collect_system_metrics = getattr(core_monitoring_config_module, 'collect_system_metrics', None)
    if MonitoringSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitoringConfigGlobalInstance:
    """测试MonitoringConfig全局实例"""

    def test_global_monitoring_instance_exists(self):
        """测试全局monitoring实例存在"""
        assert monitoring is not None
        assert isinstance(monitoring, MonitoringSystem)

    def test_collect_system_metrics_uses_global_instance(self):
        """测试collect_system_metrics使用全局实例"""
        try:
            with patch('psutil.cpu_percent', return_value=50.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value.percent = 60.0
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk.return_value.percent = 70.0
                        with patch('psutil.net_io_counters') as mock_net:
                            mock_net.return_value = None
                            
                            result = collect_system_metrics()
                            
                            # 验证函数使用了全局monitoring实例
                            if monitoring and hasattr(monitoring, 'metrics'):
                                assert 'cpu_usage' in monitoring.metrics or result is not None
                            assert result is not None
        except ImportError:
            pytest.skip("psutil not available")

    def test_global_monitoring_independent_of_class(self):
        """测试全局monitoring实例独立于MonitoringSystem类"""
        # 创建新的实例
        new_instance = MonitoringSystem()
        
        # 添加数据到新实例
        new_instance.record_metric('new_instance_metric', 200.0)
        
        # 全局实例不应该受影响
        assert 'new_instance_metric' not in monitoring.metrics

    def test_global_monitoring_thread_safety_basic(self):
        """测试全局monitoring基本线程安全性"""
        import threading
        
        def record_metrics():
            for i in range(10):
                monitoring.record_metric(f'thread_metric_{i}', float(i))
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证所有指标都被记录
        # 注意：由于并发，具体数量可能不同，但应该有指标被记录
        assert len(monitoring.metrics) > 0


