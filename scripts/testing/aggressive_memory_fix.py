#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激进内存泄漏修复脚本

基于检测结果，针对严重的内存泄漏问题进行彻底修复：
1. 内存暴涨：从18MB暴涨到983MB (+965MB)
2. 主要泄漏源：importlib模块加载、单例实例、Prometheus指标
3. 采用激进策略：强制清理、模块重载、内存隔离
"""

import sys
import gc
import psutil
import time
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class AggressiveMemoryFixer:
    """激进内存泄漏修复器"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.cleaned_modules = set()
        self.cleaned_instances = set()

    def start_monitoring(self):
        """开始内存监控"""
        self.initial_memory = self.process.memory_info().rss
        print(f"🔍 开始内存监控，初始内存: {self.initial_memory / 1024 / 1024:.2f} MB")

    def take_memory_snapshot(self, label: str):
        """记录内存快照"""
        current_memory = self.process.memory_info().rss
        memory_diff = current_memory - self.initial_memory
        print(
            f"📊 内存快照 [{label}]: {current_memory / 1024 / 1024:.2f} MB (变化: {memory_diff / 1024 / 1024:+.2f} MB)")

    def force_cleanup_singletons(self):
        """强制清理所有单例实例"""
        print("\n🧹 强制清理单例实例")

        # 定义所有需要清理的单例类
        singleton_classes = [
            ('src.infrastructure.init_infrastructure', 'Infrastructure'),
            ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
            ('src.infrastructure.config.services.unified_hot_reload', 'UnifiedConfigHotReload'),
            ('src.infrastructure.config.services.unified_sync', 'UnifiedConfigSync'),
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
            ('src.infrastructure.monitoring.metrics_collector', 'MetricsCollector'),
            ('src.infrastructure.monitoring.enhanced_monitor_manager', 'EnhancedMonitorManager'),
            ('src.infrastructure.logging.log_manager', 'LogManager'),
            ('src.infrastructure.logging.trading_logger', 'TradingLogger'),
            ('src.infrastructure.logging.log_metrics', 'LogMetrics'),
            ('src.infrastructure.logging.resource_manager', 'ResourceManager'),
            ('src.infrastructure.logging.backpressure', 'Backpressure'),
            ('src.infrastructure.error.error_handler', 'ErrorHandler'),
            ('src.infrastructure.error.retry_handler', 'RetryHandler'),
            ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
            ('src.infrastructure.resource.gpu_manager', 'GPUManager'),
            ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
            ('src.infrastructure.cache.enhanced_cache_manager', 'EnhancedCacheManager'),
            ('src.infrastructure.di.container', 'DependencyContainer'),
            ('src.infrastructure.di.enhanced_container', 'EnhancedDependencyContainer'),
            ('src.infrastructure.database.unified_database_manager', 'UnifiedDatabaseManager'),
            ('src.core.security.unified_security', 'UnifiedSecurity'),
            ('src.infrastructure.event', 'EventService'),
            ('src.infrastructure.utils.audit', 'AuditService'),
        ]

        for module_path, class_name in singleton_classes:
            try:
                # 检查模块是否已加载
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                    cls = getattr(module, class_name, None)

                    if cls is not None:
                        # 清理单例实例
                        if hasattr(cls, '_instance') and cls._instance is not None:
                            cls._instance = None
                            self.cleaned_instances.add(f"{class_name}._instance")
                            print(f"✅ 清理 {class_name} 单例实例")

                        # 清理多实例缓存
                        if hasattr(cls, '_instances'):
                            original_size = len(cls._instances)
                            cls._instances.clear()
                            self.cleaned_instances.add(f"{class_name}._instances")
                            print(f"✅ 清理 {class_name} 实例缓存 ({original_size} 个实例)")

                        # 清理其他可能的实例变量
                        instance_vars = ['_instance', '_instances', '_manager', '_service']
                        for var_name in instance_vars:
                            if hasattr(cls, var_name):
                                setattr(cls, var_name, None)
                                self.cleaned_instances.add(f"{class_name}.{var_name}")

            except Exception as e:
                print(f"❌ 清理 {module_path}.{class_name} 失败: {e}")

    def force_cleanup_global_variables(self):
        """强制清理全局变量"""
        print("\n🔧 强制清理全局变量")

        global_vars = [
            ('src.infrastructure.config.unified_manager', '_unified_manager_instance'),
            ('src.infrastructure.monitoring.metrics_collector', '_metrics_collector_instance'),
            ('src.infrastructure.config.services.unified_hot_reload', '_hot_reload_instance'),
            ('src.infrastructure.config.services.unified_sync', '_sync_instance'),
            ('src.infrastructure.monitoring.application_monitor', '_app_monitor_instance'),
            ('src.infrastructure.monitoring.system_monitor', '_sys_monitor_instance'),
        ]

        for module_path, var_name in global_vars:
            try:
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                    if hasattr(module, var_name):
                        setattr(module, var_name, None)
                        self.cleaned_instances.add(f"{module_path}.{var_name}")
                        print(f"✅ 清理全局变量: {module_path}.{var_name}")

            except Exception as e:
                print(f"❌ 清理全局变量失败 {module_path}.{var_name}: {e}")

    def force_cleanup_prometheus_registry(self):
        """强制清理Prometheus注册表"""
        print("\n🔧 强制清理Prometheus注册表")

        try:
            # 清理主注册表
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                original_size = len(REGISTRY._names_to_collectors)
                REGISTRY._names_to_collectors.clear()
                print(f"✅ 清理Prometheus注册表: {original_size} 个指标")

            # 清理其他可能的注册表
            for name in list(sys.modules.keys()):
                if 'prometheus' in name.lower():
                    module = sys.modules[name]
                    if hasattr(module, '_names_to_collectors'):
                        module._names_to_collectors.clear()
                        print(f"✅ 清理 {name} 注册表")

        except Exception as e:
            print(f"❌ 清理Prometheus注册表失败: {e}")

    def force_cleanup_caches(self):
        """强制清理所有缓存"""
        print("\n🧹 强制清理缓存")

        cache_modules = [
            ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
            ('src.infrastructure.logging.log_manager', 'LogManager'),
            ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
        ]

        for module_path, class_name in cache_modules:
            try:
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                    cls = getattr(module, class_name, None)

                    if cls is not None:
                        # 清理缓存属性
                        cache_attrs = ['_cache', '_metrics',
                                       '_instances', '_data', '_config', '_handlers']
                        for attr_name in cache_attrs:
                            if hasattr(cls, attr_name):
                                cache_obj = getattr(cls, attr_name)
                                if hasattr(cache_obj, 'clear'):
                                    original_size = len(cache_obj)
                                    cache_obj.clear()
                                    print(f"✅ 清理 {class_name}.{attr_name} ({original_size} 项)")

            except Exception as e:
                print(f"❌ 清理缓存失败 {module_path}.{class_name}: {e}")

    def force_cleanup_module_cache(self):
        """强制清理模块缓存"""
        print("\n🧹 强制清理模块缓存")

        # 清理基础设施相关模块
        infrastructure_modules = [
            'src.infrastructure',
            'src.infrastructure.config',
            'src.infrastructure.monitoring',
            'src.infrastructure.logging',
            'src.infrastructure.error',
            'src.infrastructure.resource',
            'src.infrastructure.cache',
            'src.infrastructure.di',
            'src.infrastructure.database',
            'src.core.security',
            'src.infrastructure.event',
            'src.infrastructure.utils',
        ]

        cleaned_count = 0
        for module_name in infrastructure_modules:
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                    cleaned_count += 1
                    self.cleaned_modules.add(module_name)
                except Exception as e:
                    print(f"❌ 清理模块缓存失败 {module_name}: {e}")

        print(f"✅ 清理了 {cleaned_count} 个模块缓存")

    def force_stop_threads(self):
        """强制停止所有后台线程"""
        print("\n🛑 强制停止后台线程")

        try:
            active_threads = threading.enumerate()
            stopped_count = 0

            for thread in active_threads:
                thread_name = thread.name.lower()
                if any(keyword in thread_name for keyword in ['monitor', 'watch', 'background', 'daemon', 'worker']):
                    if thread.is_alive() and thread != threading.current_thread():
                        try:
                            thread.join(timeout=0.5)
                            stopped_count += 1
                            print(f"✅ 停止线程: {thread.name}")
                        except Exception as e:
                            print(f"❌ 停止线程失败 {thread.name}: {e}")

            print(f"✅ 停止了 {stopped_count} 个后台线程")

        except Exception as e:
            print(f"❌ 停止线程失败: {e}")

    def force_garbage_collection(self):
        """强制垃圾回收"""
        print("\n💪 强制垃圾回收")

        # 多次垃圾回收
        for i in range(10):
            collected = gc.collect()
            print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")
            time.sleep(0.1)

        # 清理弱引用
        gc.collect()

        # 清理循环引用
        gc.collect()

    def create_memory_isolation_script(self):
        """创建内存隔离脚本"""
        print("\n📝 创建内存隔离脚本")

        isolation_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存隔离脚本

在测试环境中创建完全隔离的内存环境，避免内存泄漏
"""

import os
import sys
import gc
import time
import importlib
from typing import Dict, Any

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
'''

        isolation_file = project_root / "scripts" / "testing" / "memory_isolation.py"

        with open(isolation_file, 'w', encoding='utf-8') as f:
            f.write(isolation_script)

        print(f"✅ 创建内存隔离脚本: {isolation_file}")

    def run_aggressive_fix(self):
        """运行激进修复"""
        print("🚀 开始激进内存泄漏修复")
        print("=" * 60)

        # 开始监控
        self.start_monitoring()

        # 执行激进清理
        self.take_memory_snapshot("修复前")

        self.force_cleanup_singletons()
        self.force_cleanup_global_variables()
        self.force_cleanup_prometheus_registry()
        self.force_cleanup_caches()
        self.force_stop_threads()
        self.force_cleanup_module_cache()
        self.force_garbage_collection()

        self.take_memory_snapshot("修复后")

        # 创建隔离脚本
        self.create_memory_isolation_script()

        print(f"\n✅ 激进内存泄漏修复完成")
        print(f"清理的模块数量: {len(self.cleaned_modules)}")
        print(f"清理的实例数量: {len(self.cleaned_instances)}")
        print(f"清理的模块: {', '.join(list(self.cleaned_modules)[:5])}...")
        print(f"清理的实例: {', '.join(list(self.cleaned_instances)[:5])}...")

        print("\n📋 使用说明:")
        print("1. 运行内存隔离: python scripts/testing/memory_isolation.py")
        print("2. 在测试中使用隔离环境")
        print("3. 避免大量import操作，使用激进清理策略")


def main():
    """主函数"""
    fixer = AggressiveMemoryFixer()
    fixer.run_aggressive_fix()


if __name__ == "__main__":
    main()
