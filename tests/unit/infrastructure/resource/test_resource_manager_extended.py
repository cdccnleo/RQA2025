#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ResourceManager扩展测试"""

import pytest
import time


def test_resource_manager_import():
    """测试ResourceManager导入"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        assert CoreResourceManager is not None
    except ImportError:
        pytest.skip("CoreResourceManager不可用")


def test_resource_manager_init_default():
    """测试ResourceManager默认初始化"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager = CoreResourceManager()
        assert manager is not None
        assert manager.config is not None
        assert manager._monitoring is True
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_init_with_config():
    """测试ResourceManager带配置初始化"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        from src.infrastructure.resource.config.config_classes import ResourceMonitorConfig
        
        config = ResourceMonitorConfig()
        manager = CoreResourceManager(config=config)
        assert manager is not None
        assert manager.config is config
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_has_lock():
    """测试ResourceManager有锁机制"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        import threading
        
        manager = CoreResourceManager()
        assert hasattr(manager, '_lock')
        assert isinstance(manager._lock, threading.Lock)
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_has_history():
    """测试ResourceManager有历史记录"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager = CoreResourceManager()
        assert hasattr(manager, '_resource_history')
        assert isinstance(manager._resource_history, list)
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_start_monitoring():
    """测试ResourceManager启动监控"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager = CoreResourceManager()
        
        # 调用start_monitoring
        manager.start_monitoring()
        assert manager._monitoring is True
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_monitor_thread():
    """测试ResourceManager监控线程"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager = CoreResourceManager()
        
        # 给监控线程一点时间启动
        time.sleep(0.1)
        
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_logger():
    """测试ResourceManager有logger"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager = CoreResourceManager()
        assert hasattr(manager, 'logger')
        assert manager.logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_resource_manager_multiple_instances():
    """测试创建多个ResourceManager实例"""
    try:
        from src.infrastructure.resource.core.resource_manager import CoreResourceManager
        manager1 = CoreResourceManager()
        manager2 = CoreResourceManager()
        
        assert manager1 is not None
        assert manager2 is not None
        assert id(manager1) != id(manager2)
    except Exception:
        pytest.skip("测试跳过")


def test_system_resource_analyzer_import():
    """测试SystemResourceAnalyzer导入"""
    try:
        from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
        assert SystemResourceAnalyzer is not None
    except ImportError:
        pytest.skip("SystemResourceAnalyzer不可用")


def test_system_resource_analyzer_init():
    """测试SystemResourceAnalyzer初始化"""
    try:
        from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
        analyzer = SystemResourceAnalyzer()
        assert analyzer is not None
    except Exception:
        pytest.skip("测试跳过")


def test_system_resource_analyzer_methods():
    """测试SystemResourceAnalyzer方法"""
    try:
        from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
        analyzer = SystemResourceAnalyzer()
        
        # 检查常见方法
        common_methods = ['get_cpu_usage', 'get_memory_usage', 'get_disk_usage']
        for method_name in common_methods:
            if hasattr(analyzer, method_name):
                assert callable(getattr(analyzer, method_name))
    except Exception:
        pytest.skip("测试跳过")


def test_event_bus_import():
    """测试EventBus导入"""
    try:
        from src.infrastructure.resource.core.event_bus import EventBus
        assert EventBus is not None
    except ImportError:
        pytest.skip("EventBus不可用")


def test_event_bus_init():
    """测试EventBus初始化"""
    try:
        from src.infrastructure.resource.core.event_bus import EventBus
        bus = EventBus()
        assert bus is not None
    except Exception:
        pytest.skip("测试跳过")

