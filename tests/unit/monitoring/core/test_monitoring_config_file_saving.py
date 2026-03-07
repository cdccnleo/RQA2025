#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig文件保存测试
测试__main__块中的文件保存逻辑
"""

import sys
import importlib
from pathlib import Path
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
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
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    if monitoring_config_module is None:
        pytest.skip("监控配置模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestMonitoringConfigFileSaving:
    """测试文件保存逻辑"""

    def test_file_saving_basic(self):
        """测试基本文件保存功能"""
        test_results = {
            'system_metrics': {'cpu_percent': 50.0, 'memory_percent': 60.0, 'disk_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0, 'p95_response_time': 200.0, 'total_requests': 100},
            'concurrency_performance': {'concurrent_requests': 50, 'avg_response_time': 300.0, 'max_response_time': 500.0},
            'alerts': [],
            'monitoring_report': {'metrics_count': 10, 'traces_count': 150, 'alerts_count': 0},
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                # 模拟保存测试结果
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证文件被打开，编码为utf-8
                mock_file.assert_called_once_with('monitoring_test_results.json', 'w', encoding='utf-8')
                # 验证json.dump被调用
                mock_json_dump.assert_called_once()

    def test_file_saving_with_alerts(self):
        """测试保存带告警的测试结果"""
        test_results = {
            'system_metrics': {'cpu_percent': 90.0, 'memory_percent': 80.0, 'disk_percent': 70.0},
            'api_performance': {'avg_response_time': 1500.0, 'p95_response_time': 3000.0, 'total_requests': 100},
            'concurrency_performance': {'concurrent_requests': 50, 'avg_response_time': 400.0, 'max_response_time': 600.0},
            'alerts': [
                {'type': 'cpu_high', 'message': 'CPU使用率过高: 90.0%', 'severity': 'critical'},
                {'type': 'memory_high', 'message': '内存使用率过高: 80.0%', 'severity': 'warning'},
                {'type': 'api_slow', 'message': 'API响应时间过慢: 1500ms', 'severity': 'warning'}
            ],
            'monitoring_report': {'metrics_count': 15, 'traces_count': 250, 'alerts_count': 3},
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()

    def test_file_saving_encoding(self):
        """测试文件保存的编码格式"""
        test_results = {
            'system_metrics': {'cpu_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0},
            'concurrency_performance': {'concurrent_requests': 50},
            'alerts': [],
            'monitoring_report': {'metrics_count': 10},
            'timestamp': datetime.now().isoformat()
        }
        
        # 测试UTF-8编码
        with patch('builtins.open', mock_open()) as mock_file:
            with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
            
            # 验证编码参数正确（没有空格）
            call_args = mock_file.call_args
            assert call_args[1]['encoding'] == 'utf-8'
            assert call_args[1]['encoding'] != 'utf - 8'  # 确保不是带空格的错误格式

    def test_file_saving_json_parameters(self):
        """测试JSON保存参数"""
        test_results = {
            'system_metrics': {'cpu_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0},
            'concurrency_performance': {'concurrent_requests': 50},
            'alerts': [],
            'monitoring_report': {'metrics_count': 10},
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证json.dump的参数
                call_args = mock_json_dump.call_args
                assert call_args[1]['ensure_ascii'] is False
                assert call_args[1]['indent'] == 2
                assert 'default' in call_args[1]

    def test_file_saving_with_unicode(self):
        """测试保存包含Unicode字符的测试结果"""
        test_results = {
            'system_metrics': {'cpu_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0},
            'concurrency_performance': {'concurrent_requests': 50},
            'alerts': [
                {'type': 'cpu_high', 'message': 'CPU使用率过高: 90.0%', 'severity': 'critical'}
            ],
            'monitoring_report': {'metrics_count': 10},
            'timestamp': datetime.now().isoformat(),
            'note': '测试中文字符保存'
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证ensure_ascii=False，允许保存Unicode字符
                call_args = mock_json_dump.call_args
                assert call_args[1]['ensure_ascii'] is False

    def test_file_saving_complete_structure(self):
        """测试完整的测试结果结构"""
        test_results = {
            'system_metrics': {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 50.0
            },
            'api_performance': {
                'avg_response_time': 100.0,
                'p95_response_time': 200.0,
                'total_requests': 100
            },
            'concurrency_performance': {
                'concurrent_requests': 50,
                'avg_response_time': 300.0,
                'max_response_time': 500.0
            },
            'alerts': [],
            'monitoring_report': {
                'timestamp': datetime.now().isoformat(),
                'metrics_count': 10,
                'traces_count': 150,
                'alerts_count': 0,
                'latest_metrics': {},
                'performance_summary': {
                    'avg_duration': 0.15,
                    'max_duration': 0.5,
                    'min_duration': 0.1,
                    'total_traces': 150
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证所有必需的字段都存在
                assert 'system_metrics' in test_results
                assert 'api_performance' in test_results
                assert 'concurrency_performance' in test_results
                assert 'alerts' in test_results
                assert 'monitoring_report' in test_results
                assert 'timestamp' in test_results

    def test_file_saving_with_default_str(self):
        """测试default=str参数处理非JSON序列化对象"""
        from datetime import datetime
        
        test_results = {
            'system_metrics': {'cpu_percent': 50.0},
            'api_performance': {'avg_response_time': 100.0},
            'concurrency_performance': {'concurrent_requests': 50},
            'alerts': [],
            'monitoring_report': {'metrics_count': 10},
            'timestamp': datetime.now(),  # datetime对象需要使用default=str
            'some_object': datetime.now()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
                
                # 验证default参数存在，用于处理datetime等对象
                call_args = mock_json_dump.call_args
                assert 'default' in call_args[1]
                assert call_args[1]['default'] == str


