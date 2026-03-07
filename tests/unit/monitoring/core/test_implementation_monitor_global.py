#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImplementationMonitor全局函数测试
补充implementation_monitor.py中全局函数的测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

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
    core_implementation_monitor_module = importlib.import_module('src.monitoring.core.implementation_monitor')
    get_implementation_monitor = getattr(core_implementation_monitor_module, 'get_implementation_monitor', None)
    ImplementationMonitor = getattr(core_implementation_monitor_module, 'ImplementationMonitor', None)
    _implementation_monitor = getattr(core_implementation_monitor_module, '_implementation_monitor', None)
    
    if ImplementationMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestGlobalFunctions:
    """测试全局函数"""

    def test_get_implementation_monitor_creates_instance(self):
        """测试get_implementation_monitor创建实例"""
        # 重置全局实例
        import src.monitoring.core.implementation_monitor as im_module
        im_module._implementation_monitor = None
        
        monitor = get_implementation_monitor()
        
        assert isinstance(monitor, ImplementationMonitor)
        assert im_module._implementation_monitor is not None
        assert im_module._implementation_monitor is monitor

    def test_get_implementation_monitor_returns_same_instance(self):
        """测试get_implementation_monitor返回相同实例"""
        import src.monitoring.core.implementation_monitor as im_module
        
        # 确保有实例
        if im_module._implementation_monitor is None:
            im_module._implementation_monitor = ImplementationMonitor()
        
        monitor1 = get_implementation_monitor()
        monitor2 = get_implementation_monitor()
        
        assert monitor1 is monitor2
        assert monitor1 is im_module._implementation_monitor

    def test_get_implementation_monitor_singleton(self):
        """测试get_implementation_monitor单例模式"""
        import src.monitoring.core.implementation_monitor as im_module
        
        # 重置全局实例
        im_module._implementation_monitor = None
        
        monitor1 = get_implementation_monitor()
        monitor2 = get_implementation_monitor()
        
        assert monitor1 is monitor2

    def test_get_implementation_monitor_multiple_calls(self):
        """测试多次调用get_implementation_monitor"""
        import src.monitoring.core.implementation_monitor as im_module
        
        # 重置全局实例
        im_module._implementation_monitor = None
        
        monitors = [get_implementation_monitor() for _ in range(5)]
        
        # 所有调用应该返回同一个实例
        assert all(mon is monitors[0] for mon in monitors)

    def test_get_implementation_monitor_with_operations(self):
        """测试get_implementation_monitor配合操作使用"""
        import src.monitoring.core.implementation_monitor as im_module
        
        # 重置全局实例
        im_module._implementation_monitor = None
        
        monitor = get_implementation_monitor()
        
        # 验证可以使用返回的实例进行操作
        dashboard_id = "test_dashboard_global"
        
        # 创建dashboard
        dashboard = monitor.create_dashboard(dashboard_id, "Test Dashboard", "Test Description")
        assert dashboard is not None
        assert dashboard.dashboard_id == dashboard_id
        
        # 再次获取应该是同一个实例
        monitor2 = get_implementation_monitor()
        assert monitor is monitor2
        
        # 应该能够访问之前创建的dashboard
        summary = monitor2.get_dashboard_summary(dashboard_id)
        assert summary is not None
        assert summary['dashboard_id'] == dashboard_id


class TestGlobalFunctionsIntegration:
    """测试全局函数集成"""

    def test_global_functions_work_together(self):
        """测试全局函数协同工作"""
        import src.monitoring.core.implementation_monitor as im_module
        
        # 重置全局实例
        im_module._implementation_monitor = None
        
        # 获取监控器
        monitor1 = get_implementation_monitor()
        assert isinstance(monitor1, ImplementationMonitor)
        
        # 执行操作
        dashboard_id = "integration_test_dashboard"
        dashboard = monitor1.create_dashboard(
            dashboard_id,
            "Integration Test Dashboard",
            "Testing global functions integration"
        )
        assert dashboard is not None
        
        # 再次获取应该是同一个实例
        monitor2 = get_implementation_monitor()
        assert monitor1 is monitor2
        
        # 验证数据一致性
        summary = monitor2.get_dashboard_summary(dashboard_id)
        assert summary is not None
        assert summary['dashboard_id'] == dashboard_id



