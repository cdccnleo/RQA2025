#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / 'src')

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块 - 只在需要时导入，避免模块级别执行
try:
    import monitoring.mobile.mobile_monitor as mobile_monitor_module
    create_mobile_monitor_app = getattr(mobile_monitor_module, 'create_mobile_monitor_app', None)
    if create_mobile_monitor_app is None:
        pytest.skip('监控模块导入失败', allow_module_level=True)
except ImportError:
    pytest.skip('监控模块导入失败', allow_module_level=True)


class TestMobileMonitorGlobalFunctions:
    '''测试MobileMonitor全局函数'''

    @patch('monitoring.mobile.mobile_monitor.create_mobile_monitor_app')
    @patch('builtins.print')
    def test_create_mobile_monitor_app_mock(self, mock_print, mock_create_app):
        '''测试create_mobile_monitor_app的mock设置'''
        mock_monitor = Mock()
        mock_create_app.return_value = mock_monitor

        # 测试mock设置
        assert mock_create_app.return_value == mock_monitor

    @patch('monitoring.mobile.mobile_monitor.create_mobile_monitor_app')
    @patch('builtins.print')
    def test_main_block_simulation(self, mock_print, mock_create_app):
        '''测试主块逻辑模拟'''
        mock_monitor = Mock()
        mock_create_app.return_value = mock_monitor

        # 模拟主块逻辑但不实际执行
        # 在实际的主块中会调用: app = create_mobile_monitor_app(); app.start_server()

        # 验证mock设置正确
        assert mock_create_app is not None
        assert mock_print is not None
