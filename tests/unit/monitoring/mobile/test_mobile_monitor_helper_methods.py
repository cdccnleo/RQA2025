#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
from pathlib import Path
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / 'src')

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    mobile_mobile_monitor_module = importlib.import_module('monitoring.mobile.mobile_monitor')
    MobileMonitor = getattr(mobile_mobile_monitor_module, 'MobileMonitor', None)
    get_mobile_monitor = getattr(mobile_mobile_monitor_module, 'get_mobile_monitor', None)
    start_mobile_monitoring = getattr(mobile_mobile_monitor_module, 'start_mobile_monitoring', None)
    stop_mobile_monitoring = getattr(mobile_mobile_monitor_module, 'stop_mobile_monitoring', None)

    if MobileMonitor is None:
        pytest.skip('监控模块导入失败', allow_module_level=True)
except ImportError:
    pytest.skip('监控模块导入失败', allow_module_level=True)


class TestMobileMonitorHelperMethods:
    '''测试MobileMonitor辅助方法'''

    @pytest.fixture
    def monitor(self):
        '''创建monitor实例'''
        config = {
            'host': '127.0.0.1',
            'port': 8082,
            'debug': False
        }
        return MobileMonitor(config)

    def test_generate_mock_system_data_structure(self, monitor):
        '''测试生成模拟系统数据结构'''
        data = monitor._generate_mock_system_data()
        
        assert isinstance(data, dict)
        assert 'status' in data
        assert 'uptime' in data
        assert 'active_nodes' in data
        assert 'total_trades' in data
        assert 'pnl' in data
        assert 'cpu_usage' in data
        assert 'memory_usage' in data
        assert 'disk_usage' in data
        assert 'network_sent' in data
        assert 'network_recv' in data

    def test_generate_mock_system_data_values(self, monitor):
        '''测试生成模拟系统数据的值'''
        data = monitor._generate_mock_system_data()
        
        # 检查数值类型
        assert isinstance(data['cpu_usage'], (int, float))
        assert isinstance(data['memory_usage'], (int, float))
        assert isinstance(data['disk_usage'], (int, float))
        assert isinstance(data['total_trades'], int)
        assert isinstance(data['pnl'], (int, float))

    def test_monitor_initialization_with_config(self):
        '''测试带配置的monitor初始化'''
        config = {
            'host': 'localhost',
            'port': 8083,
            'debug': True
        }
        monitor = MobileMonitor(config)
        
        assert monitor is not None
        assert hasattr(monitor, 'config')

    def test_monitor_initialization_default_config(self):
        '''测试默认配置的monitor初始化'''
        monitor = MobileMonitor()
        
        assert monitor is not None
        assert hasattr(monitor, 'config')
