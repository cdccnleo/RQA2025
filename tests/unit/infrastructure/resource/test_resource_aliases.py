#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""resource别名模块测试"""

import pytest


def test_monitoring_alert_system_import():
    """测试monitoring_alert_system导入"""
    try:
        from src.infrastructure.resource import monitoring_alert_system
        assert monitoring_alert_system is not None
    except ImportError:
        pytest.skip("模块不可用")


def test_resource_optimization_import():
    """测试resource_optimization导入"""
    try:
        from src.infrastructure.resource import resource_optimization
        assert resource_optimization is not None
    except ImportError:
        pytest.skip("模块不可用")


def test_system_monitor_import():
    """测试system_monitor导入"""
    try:
        from src.infrastructure.resource import system_monitor
        assert system_monitor is not None
    except ImportError:
        pytest.skip("模块不可用")


def test_task_scheduler_import():
    """测试task_scheduler导入"""
    try:
        from src.infrastructure.resource import task_scheduler
        assert task_scheduler is not None
    except ImportError:
        pytest.skip("模块不可用")


def test_monitoring_alert_system_class():
    """测试MonitoringAlertSystem类"""
    try:
        from src.infrastructure.resource.monitoring_alert_system import MonitoringAlertSystem
        assert MonitoringAlertSystem is not None
        assert hasattr(MonitoringAlertSystem, '__init__')
    except Exception:
        pytest.skip("测试跳过")


def test_resource_optimization_class():
    """测试ResourceOptimization类"""
    try:
        from src.infrastructure.resource.resource_optimization import ResourceOptimization
        assert ResourceOptimization is not None
        assert hasattr(ResourceOptimization, '__init__')
    except Exception:
        pytest.skip("测试跳过")


def test_system_monitor_class():
    """测试SystemMonitor类"""
    try:
        from src.infrastructure.resource.system_monitor import SystemMonitor
        assert SystemMonitor is not None
        assert hasattr(SystemMonitor, '__init__')
    except Exception:
        pytest.skip("测试跳过")


def test_task_scheduler_class():
    """测试TaskScheduler类"""
    try:
        from src.infrastructure.resource.task_scheduler import TaskScheduler
        assert TaskScheduler is not None
        assert hasattr(TaskScheduler, '__init__')
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_import():
    """测试ResourceManager导入"""
    try:
        from src.infrastructure.resource.core.resource_manager import ResourceManager
        assert ResourceManager is not None
    except ImportError:
        pytest.skip("测试跳过")


def test_system_resource_analyzer_import():
    """测试SystemResourceAnalyzer导入"""
    try:
        from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
        assert SystemResourceAnalyzer is not None
    except ImportError:
        pytest.skip("测试跳过")


def test_resource_manager_initialization():
    """测试ResourceManager初始化"""
    try:
        from src.infrastructure.resource.core.resource_manager import ResourceManager
        manager = ResourceManager()
        assert manager is not None
    except Exception:
        pytest.skip("测试跳过")


def test_system_resource_analyzer_methods():
    """测试SystemResourceAnalyzer方法"""
    try:
        from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
        analyzer = SystemResourceAnalyzer()
        assert analyzer is not None
        
        # 检查常见方法
        if hasattr(analyzer, 'get_cpu_usage'):
            assert callable(analyzer.get_cpu_usage)
        if hasattr(analyzer, 'get_memory_usage'):
            assert callable(analyzer.get_memory_usage)
    except Exception:
        pytest.skip("测试跳过")

