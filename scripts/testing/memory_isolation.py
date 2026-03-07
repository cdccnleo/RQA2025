#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存隔离脚本

在测试环境中创建完全隔离的内存环境，避免内存泄漏
"""

import os
import sys
import gc
import time


def create_isolated_environment():
    """创建隔离的测试环境"""
    print("🔒 创建内存隔离环境...")

    # 设置环境变量
    os.environ['PYTEST_CURRENT_TEST'] = 'isolated_memory_test'
    os.environ['DISABLE_HEAVY_IMPORTS'] = 'true'
    os.environ['ENABLE_MEMORY_OPTIMIZATION'] = 'true'
    os.environ['PROMETHEUS_ISOLATED'] = 'true'

    # 创建隔离的Prometheus注册表
    try:
        from prometheus_client import CollectorRegistry
        isolated_registry = CollectorRegistry()
        print("✅ 创建隔离的Prometheus注册表")
        return isolated_registry
    except Exception as e:
        print(f"❌ 创建隔离注册表失败: {e}")
        return None


def aggressive_cleanup():
    """激进的内存清理"""
    print("🧹 开始激进内存清理...")

    # 步骤1: 强制清理单例
    force_cleanup_singletons()
    time.sleep(0.1)

    # 步骤2: 强制清理全局变量
    force_cleanup_global_variables()
    time.sleep(0.1)

    # 步骤3: 强制清理Prometheus注册表
    force_cleanup_prometheus_registry()
    time.sleep(0.1)

    # 步骤4: 强制清理缓存
    force_cleanup_caches()
    time.sleep(0.1)

    # 步骤5: 强制停止线程
    force_stop_threads()
    time.sleep(0.1)

    # 步骤6: 强制清理模块缓存
    force_cleanup_module_cache()
    time.sleep(0.1)

    # 步骤7: 强制垃圾回收
    force_garbage_collection()

    print("✅ 激进内存清理完成")


def force_cleanup_singletons():
    """强制清理单例"""
    singleton_classes = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
        ('src.infrastructure.logging.log_manager', 'LogManager'),
        ('src.infrastructure.error.error_handler', 'ErrorHandler'),
    ]

    for module_path, class_name in singleton_classes:
        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]
                cls = getattr(module, class_name, None)
                if cls is not None:
                    if hasattr(cls, '_instance'):
                        cls._instance = None
                    if hasattr(cls, '_instances'):
                        cls._instances.clear()
        except Exception:
            pass


def force_cleanup_global_variables():
    """强制清理全局变量"""
    global_vars = [
        ('src.infrastructure.config.unified_manager', '_unified_manager_instance'),
        ('src.infrastructure.monitoring.metrics_collector', '_metrics_collector_instance'),
    ]

    for module_path, var_name in global_vars:
        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]
                if hasattr(module, var_name):
                    setattr(module, var_name, None)
        except Exception:
            pass


def force_cleanup_prometheus_registry():
    """强制清理Prometheus注册表"""
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            REGISTRY._names_to_collectors.clear()
    except Exception:
        pass


def force_cleanup_caches():
    """强制清理缓存"""
    cache_modules = [
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
    ]

    for module_path, class_name in cache_modules:
        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]
                cls = getattr(module, class_name, None)
                if cls is not None:
                    cache_attrs = ['_cache', '_metrics', '_instances', '_data']
                    for attr_name in cache_attrs:
                        if hasattr(cls, attr_name):
                            cache_obj = getattr(cls, attr_name)
                            if hasattr(cache_obj, 'clear'):
                                cache_obj.clear()
        except Exception:
            pass


def force_stop_threads():
    """强制停止线程"""
    try:
        import threading
        for thread in threading.enumerate():
            if thread.name.lower().find('monitor') != -1:
                thread.join(timeout=0.5)
    except Exception:
        pass


def force_cleanup_module_cache():
    """强制清理模块缓存"""
    infrastructure_modules = [
        'src.infrastructure.config',
        'src.infrastructure.monitoring',
        'src.infrastructure.logging',
        'src.infrastructure.error',
    ]

    for module_name in infrastructure_modules:
        if module_name in sys.modules:
            try:
                del sys.modules[module_name]
            except Exception:
                pass


def force_garbage_collection():
    """强制垃圾回收"""
    for _ in range(10):
        gc.collect()
        time.sleep(0.1)


if __name__ == "__main__":
    isolated_registry = create_isolated_environment()
    aggressive_cleanup()
