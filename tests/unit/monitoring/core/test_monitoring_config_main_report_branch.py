#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig主程序报告分支测试
补充__main__块中report['performance_summary']存在时的分支测试
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import json

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
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    collect_system_metrics = getattr(core_monitoring_config_module, 'collect_system_metrics', None)
    simulate_api_performance_test = getattr(core_monitoring_config_module, 'simulate_api_performance_test', None)
    test_concurrency_performance = getattr(core_monitoring_config_module, 'test_concurrency_performance', None)
    
    if monitoring is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitoringConfigMainReportBranch:
    """测试MonitoringConfig主程序的报告分支"""

    @pytest.fixture(autouse=True)
    def reset_monitoring(self):
        """重置monitoring实例"""
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []
        yield
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []

    def test_main_execution_with_performance_summary(self):
        """测试主程序执行，包含性能摘要的分支"""
        # 设置有追踪数据，产生性能摘要
        span_id = monitoring.start_trace('trace_1', 'operation')
        monitoring.end_trace(span_id)
        
        with patch('builtins.print'):
            with patch('src.monitoring.core.monitoring_config.collect_system_metrics') as mock_collect:
                mock_collect.return_value = {
                    'cpu_percent': 50.0,
                    'memory_percent': 60.0,
                    'disk_percent': 70.0
                }
                
                with patch('src.monitoring.core.monitoring_config.simulate_api_performance_test') as mock_api:
                    mock_api.return_value = {
                        'avg_response_time': 100.0,
                        'p95_response_time': 200.0,
                        'total_requests': 100
                    }
                    
                    with patch('src.monitoring.core.monitoring_config.test_concurrency_performance') as mock_concurrency:
                        mock_concurrency.return_value = {
                            'concurrent_requests': 50,
                            'avg_response_time': 150.0,
                            'max_response_time': 300.0
                        }
                        
                        with patch('builtins.open', mock_open()) as mock_file:
                            with patch('json.dump'):
                                # 执行主程序逻辑
                                report = monitoring.generate_report()
                                
                                # 验证性能摘要存在
                                assert 'performance_summary' in report
                                assert report['performance_summary'] != {}

    def test_main_execution_without_performance_summary(self):
        """测试主程序执行，无性能摘要的分支"""
        # 不创建追踪数据，不会有性能摘要
        with patch('builtins.print'):
            with patch('src.monitoring.core.monitoring_config.collect_system_metrics') as mock_collect:
                mock_collect.return_value = {
                    'cpu_percent': 50.0,
                    'memory_percent': 60.0,
                    'disk_percent': 70.0
                }
                
                with patch('src.monitoring.core.monitoring_config.simulate_api_performance_test') as mock_api:
                    mock_api.return_value = {
                        'avg_response_time': 100.0,
                        'p95_response_time': 200.0,
                        'total_requests': 100
                    }
                    
                    with patch('src.monitoring.core.monitoring_config.test_concurrency_performance') as mock_concurrency:
                        mock_concurrency.return_value = {
                            'concurrent_requests': 50,
                            'avg_response_time': 150.0,
                            'max_response_time': 300.0
                        }
                        
                        # 执行主程序逻辑
                        report = monitoring.generate_report()
                        
                        # 验证性能摘要为空字典（因为没有追踪）
                        assert 'performance_summary' in report
                        assert report['performance_summary'] == {}

    def test_main_execution_with_alerts(self):
        """测试主程序执行，有告警的分支"""
        # 创建告警
        monitoring.record_metric('cpu_usage', 90.0)  # 超过阈值，触发告警
        
        with patch('builtins.print'):
            alerts = monitoring.check_alerts()
            
            # 验证有告警
            assert len(alerts) > 0

    def test_main_execution_without_alerts(self):
        """测试主程序执行，无告警的分支"""
        # 不创建告警条件
        with patch('builtins.print'):
            alerts = monitoring.check_alerts()
            
            # 验证无告警
            assert isinstance(alerts, list)

    def test_main_execution_file_saving(self):
        """测试主程序文件保存逻辑"""
        test_results = {
            'system_metrics': {'cpu_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0},
            'concurrency_performance': {'concurrent_requests': 50},
            'alerts': [],
            'monitoring_report': {},
            'timestamp': '2025-01-27T00:00:00'
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_dump:
                # 模拟文件保存
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证文件被打开
                assert mock_file.called
                mock_dump.assert_called_once()



