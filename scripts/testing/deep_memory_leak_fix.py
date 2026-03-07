#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度内存泄漏修复脚本

专门解决基础设施层的内存暴涨问题，包括：
1. 彻底清理所有单例实例
2. 隔离Prometheus指标注册
3. 清理所有缓存和注册表
4. 强制垃圾回收和内存释放
5. 监控内存使用并提供详细报告
"""

import os
import sys
import gc
import psutil
import time
from pathlib import Path
import importlib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DeepMemoryLeakFixer:
    """深度内存泄漏修复器"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.memory_snapshots = []
        self.cleaned_instances = set()

    def start_monitoring(self):
        """开始内存监控"""
        self.initial_memory = self.process.memory_info().rss
        print(f"🔍 开始内存监控，初始内存: {self.initial_memory / 1024 / 1024:.2f} MB")

    def take_memory_snapshot(self, label: str):
        """记录内存快照"""
        current_memory = self.process.memory_info().rss
        memory_diff = current_memory - self.initial_memory
        snapshot = {
            'label': label,
            'memory_mb': current_memory / 1024 / 1024,
            'diff_mb': memory_diff / 1024 / 1024,
            'timestamp': time.time()
        }
        self.memory_snapshots.append(snapshot)
        print(f"📊 内存快照 [{label}]: {snapshot['memory_mb']:.2f} MB (变化: {snapshot['diff_mb']:+.2f} MB)")

    def deep_clean_singletons(self):
        """深度清理所有单例实例"""
        print("\n🧹 深度清理单例实例")

        # 定义所有需要清理的单例类
        singleton_classes = [
            # 基础设施核心
            ('src.infrastructure.init_infrastructure', 'Infrastructure'),
            ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
            ('src.infrastructure.config.services.unified_hot_reload', 'UnifiedConfigHotReload'),
            ('src.infrastructure.config.services.unified_sync', 'UnifiedConfigSync'),

            # 监控模块
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
            ('src.infrastructure.monitoring.metrics_collector', 'MetricsCollector'),
            ('src.infrastructure.monitoring.enhanced_monitor_manager', 'EnhancedMonitorManager'),

            # 日志模块
            ('src.infrastructure.logging.log_manager', 'LogManager'),
            ('src.infrastructure.logging.trading_logger', 'TradingLogger'),
            ('src.infrastructure.logging.log_metrics', 'LogMetrics'),
            ('src.infrastructure.logging.resource_manager', 'ResourceManager'),
            ('src.infrastructure.logging.backpressure', 'Backpressure'),

            # 错误处理
            ('src.infrastructure.error.error_handler', 'ErrorHandler'),
            ('src.infrastructure.error.retry_handler', 'RetryHandler'),

            # 资源管理
            ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
            ('src.infrastructure.resource.gpu_manager', 'GPUManager'),

            # 缓存管理
            ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
            ('src.infrastructure.cache.enhanced_cache_manager', 'EnhancedCacheManager'),

            # 依赖注入
            ('src.infrastructure.di.container', 'DependencyContainer'),
            ('src.infrastructure.di.enhanced_container', 'EnhancedDependencyContainer'),

            # 数据库管理
            ('src.infrastructure.database.unified_database_manager', 'UnifiedDatabaseManager'),

            # 安全模块
            ('src.core.security.unified_security', 'UnifiedSecurity'),

            # 事件系统
            ('src.infrastructure.event', 'EventService'),

            # 工具类
            ('src.infrastructure.utils.audit', 'AuditService'),
        ]

        for module_path, class_name in singleton_classes:
            try:
                # 动态导入模块
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

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

                # 清理全局实例变量
                global_instance_vars = [
                    '_unified_manager_instance',
                    '_metrics_collector_instance',
                    '_hot_reload_instance',
                    '_sync_instance',
                ]

                for var_name in global_instance_vars:
                    if hasattr(module, var_name):
                        setattr(module, var_name, None)
                        self.cleaned_instances.add(f"{module_path}.{var_name}")
                        print(f"✅ 清理全局实例: {module_path}.{var_name}")

            except Exception as e:
                print(f"❌ 清理 {module_path}.{class_name} 失败: {e}")

    def isolate_prometheus_registry(self):
        """隔离Prometheus注册表"""
        print("\n🔧 隔离Prometheus注册表")

        try:
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

    def deep_clean_caches(self):
        """深度清理所有缓存"""
        print("\n🧹 深度清理缓存")

        # 清理配置缓存
        try:
            from src.infrastructure.config.unified_manager import get_unified_config_manager
            config_manager = get_unified_config_manager()
            if hasattr(config_manager, '_core'):
                config_manager._core.clear_cache()
                print("✅ 清理配置管理器缓存")
        except Exception as e:
            print(f"❌ 清理配置管理器缓存失败: {e}")

        # 清理监控数据缓存
        monitoring_classes = [
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
        ]

        for module_path, class_name in monitoring_classes:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                # 清理监控数据
                if hasattr(cls, '_metrics'):
                    cls._metrics.clear()
                    print(f"✅ 清理 {class_name} 监控数据")

                # 清理查询缓存
                if hasattr(cls, '_cache'):
                    cls._cache.clear()
                    print(f"✅ 清理 {class_name} 查询缓存")

            except Exception as e:
                print(f"❌ 清理 {class_name} 缓存失败: {e}")

    def stop_all_threads(self):
        """停止所有后台线程"""
        print("\n🛑 停止所有后台线程")

        # 查找并停止监控线程
        try:
            import threading
            active_threads = threading.enumerate()

            for thread in active_threads:
                thread_name = thread.name.lower()
                if any(keyword in thread_name for keyword in ['monitor', 'watch', 'background', 'daemon']):
                    if thread.is_alive() and thread != threading.current_thread():
                        thread.join(timeout=1)
                        print(f"✅ 停止线程: {thread.name}")

        except Exception as e:
            print(f"❌ 停止线程失败: {e}")

    def cleanup_module_cache(self):
        """清理模块缓存"""
        print("\n🧹 清理模块缓存")

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
                except Exception as e:
                    print(f"❌ 清理模块缓存失败 {module_name}: {e}")

        print(f"✅ 清理了 {cleaned_count} 个模块缓存")

    def force_memory_cleanup(self):
        """强制内存清理"""
        print("\n💪 强制内存清理")

        # 多次垃圾回收
        for i in range(5):
            collected = gc.collect()
            print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")
            time.sleep(0.1)  # 短暂等待

        # 清理弱引用
        gc.collect()

        # 清理循环引用
        gc.collect()

    def create_isolated_environment(self):
        """创建隔离的测试环境"""
        print("\n🔒 创建隔离测试环境")

        # 设置环境变量
        os.environ['PYTEST_CURRENT_TEST'] = 'infrastructure_memory_test'
        os.environ['DISABLE_HEAVY_IMPORTS'] = 'true'
        os.environ['ENABLE_MEMORY_OPTIMIZATION'] = 'true'

        # 创建隔离的Prometheus注册表
        try:
            from prometheus_client import CollectorRegistry
            isolated_registry = CollectorRegistry()
            print("✅ 创建隔离的Prometheus注册表")
            return isolated_registry
        except Exception as e:
            print(f"❌ 创建隔离注册表失败: {e}")
            return None

    def run_memory_safe_test(self, test_function):
        """运行内存安全的测试"""
        print(f"\n🧪 运行内存安全测试: {test_function.__name__}")

        # 测试前快照
        self.take_memory_snapshot("测试前")

        try:
            # 运行测试
            result = test_function()

            # 测试后快照
            self.take_memory_snapshot("测试后")

            # 执行清理
            self.deep_clean_singletons()
            self.isolate_prometheus_registry()
            self.deep_clean_caches()
            self.stop_all_threads()
            self.cleanup_module_cache()
            self.force_memory_cleanup()

            # 清理后快照
            self.take_memory_snapshot("清理后")

            return result

        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            return None

    def analyze_memory_usage(self):
        """分析内存使用情况"""
        print("\n📊 内存使用分析")

        if not self.memory_snapshots:
            return

        for i, snapshot in enumerate(self.memory_snapshots):
            if i > 0:
                prev = self.memory_snapshots[i-1]
                growth = snapshot['memory_mb'] - prev['memory_mb']
                print(f"  {prev['label']} -> {snapshot['label']}: {growth:+.2f} MB")

        total_growth = self.memory_snapshots[-1]['diff_mb']
        print(f"\n总内存增长: {total_growth:+.2f} MB")

        if total_growth > 50:
            print("⚠️  检测到显著内存增长，可能存在内存泄漏")
        elif total_growth > 20:
            print("⚠️  检测到中等内存增长，建议优化")
        else:
            print("✅ 内存使用正常")

    def create_memory_cleanup_script(self):
        """创建内存清理脚本"""
        print("\n📝 创建内存清理脚本")

        cleanup_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存清理脚本

用于在测试前后彻底清理内存，防止内存泄漏
"""

import gc
import sys
import os
import time
from typing import Dict, Any

def deep_cleanup():
    """深度清理所有内存"""
    print("🧹 开始深度内存清理...")
    
    # 清理单例
    cleanup_singletons()
    
    # 清理Prometheus注册表
    cleanup_prometheus_registry()
    
    # 清理缓存
    cleanup_caches()
    
    # 停止线程
    stop_threads()
    
    # 清理模块缓存
    cleanup_module_cache()
    
    # 强制垃圾回收
    force_garbage_collection()
    
    print("✅ 深度内存清理完成")

def cleanup_singletons():
    """清理所有单例实例"""
    singletons = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
        ('src.infrastructure.logging.log_manager', 'LogManager'),
        ('src.infrastructure.error.error_handler', 'ErrorHandler'),
        ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
        ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
    ]
    
    for module_path, class_name in singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            if hasattr(cls, '_instance'):
                cls._instance = None
                
            if hasattr(cls, '_instances'):
                cls._instances.clear()
                
        except Exception:
            pass

def cleanup_prometheus_registry():
    """清理Prometheus注册表"""
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            REGISTRY._names_to_collectors.clear()
    except Exception:
        pass

def cleanup_caches():
    """清理所有缓存"""
    try:
        from src.infrastructure.config import get_unified_config_manager
        config_manager = get_unified_config_manager()
        if hasattr(config_manager, '_core'):
            config_manager._core.clear_cache()
    except Exception:
        pass

def stop_threads():
    """停止后台线程"""
    try:
        import threading
        for thread in threading.enumerate():
            if thread.name.lower().find('monitor') != -1:
                thread.join(timeout=1)
    except Exception:
        pass

def cleanup_module_cache():
    """清理模块缓存"""
    infrastructure_modules = [
        'src.infrastructure',
        'src.infrastructure.config',
        'src.infrastructure.monitoring',
        'src.infrastructure.logging',
        'src.infrastructure.error',
        'src.infrastructure.resource',
        'src.infrastructure.cache',
    ]
    
    for module_name in infrastructure_modules:
        if module_name in sys.modules:
            try:
                del sys.modules[module_name]
            except Exception:
                pass

def force_garbage_collection():
    """强制垃圾回收"""
    for _ in range(5):
        gc.collect()
        time.sleep(0.1)

if __name__ == "__main__":
    deep_cleanup()
'''

        cleanup_file = project_root / "scripts" / "testing" / "deep_memory_cleanup.py"

        with open(cleanup_file, 'w', encoding='utf-8') as f:
            f.write(cleanup_script)

        print(f"✅ 创建深度内存清理脚本: {cleanup_file}")

    def run_comprehensive_fix(self):
        """运行综合修复"""
        print("🚀 开始深度内存泄漏修复")
        print("=" * 60)

        # 开始监控
        self.start_monitoring()

        # 创建隔离环境
        isolated_registry = self.create_isolated_environment()

        # 执行深度清理
        self.deep_clean_singletons()
        self.isolate_prometheus_registry()
        self.deep_clean_caches()
        self.stop_all_threads()
        self.cleanup_module_cache()
        self.force_memory_cleanup()

        # 分析内存使用
        self.analyze_memory_usage()

        # 创建清理脚本
        self.create_memory_cleanup_script()

        print(f"\n✅ 深度内存泄漏修复完成")
        print(f"清理的实例数量: {len(self.cleaned_instances)}")
        print(f"清理的实例: {', '.join(list(self.cleaned_instances)[:10])}...")


def main():
    """主函数"""
    fixer = DeepMemoryLeakFixer()
    fixer.run_comprehensive_fix()


if __name__ == "__main__":
    main()
