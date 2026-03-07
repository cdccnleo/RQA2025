#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面内存泄漏检测器

基于基础设施层内存泄漏分析报告，创建全面的内存泄漏检测和修复工具：
1. 检测单例实例泄漏
2. 检测Prometheus指标泄漏
3. 检测模块缓存泄漏
4. 检测线程泄漏
5. 提供自动修复功能
"""

import sys
import psutil
import threading
import inspect
from pathlib import Path
from typing import List
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MemoryLeakInfo:
    """内存泄漏信息"""
    leak_type: str
    module_path: str
    class_name: str
    instance_count: int
    memory_size: float  # MB
    description: str


class ComprehensiveMemoryLeakDetector:
    """全面内存泄漏检测器"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.detected_leaks: List[MemoryLeakInfo] = []
        self.singleton_modules = [
            'src.infrastructure.init_infrastructure',
            'src.infrastructure.config.unified_manager',
            'src.infrastructure.config.services.unified_hot_reload',
            'src.infrastructure.config.services.unified_sync',
            'src.infrastructure.monitoring.application_monitor',
            'src.infrastructure.monitoring.system_monitor',
            'src.infrastructure.monitoring.metrics_collector',
            'src.infrastructure.monitoring.enhanced_monitor_manager',
            'src.infrastructure.logging.log_manager',
            'src.infrastructure.logging.trading_logger',
            'src.infrastructure.logging.log_metrics',
            'src.infrastructure.logging.resource_manager',
            'src.infrastructure.logging.backpressure',
            'src.infrastructure.error.error_handler',
            'src.infrastructure.error.retry_handler',
            'src.infrastructure.resource.resource_manager',
            'src.infrastructure.resource.gpu_manager',
            'src.infrastructure.cache.memory_cache_manager',
            'src.infrastructure.cache.enhanced_cache_manager',
            'src.infrastructure.di.container',
            'src.infrastructure.di.enhanced_container',
            'src.infrastructure.database.unified_database_manager',
            'src.core.security.unified_security',
            'src.infrastructure.event',
            'src.infrastructure.utils.audit',
        ]

    def start_monitoring(self):
        """开始内存监控"""
        self.initial_memory = self.process.memory_info().rss
        print(f"🔍 开始内存监控，初始内存: {self.initial_memory / 1024 / 1024:.2f} MB")

    def detect_singleton_leaks(self):
        """检测单例内存泄漏"""
        print("\n🔍 检测单例内存泄漏...")

        for module_name in self.singleton_modules:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr):
                        # 检测单例实例
                        if hasattr(attr, '_instance') and attr._instance is not None:
                            instance_size = sys.getsizeof(attr._instance)
                            leak_info = MemoryLeakInfo(
                                leak_type="singleton_instance",
                                module_path=module_name,
                                class_name=attr.__name__,
                                instance_count=1,
                                memory_size=instance_size / 1024 / 1024,
                                description=f"单例实例 {attr.__name__} 未清理"
                            )
                            self.detected_leaks.append(leak_info)
                            print(
                                f"⚠️  发现单例泄漏: {module_name}.{attr.__name__} ({instance_size / 1024 / 1024:.2f} MB)")

                        # 检测多实例缓存
                        if hasattr(attr, '_instances') and len(attr._instances) > 0:
                            total_size = sum(sys.getsizeof(instance)
                                             for instance in attr._instances.values())
                            leak_info = MemoryLeakInfo(
                                leak_type="singleton_cache",
                                module_path=module_name,
                                class_name=attr.__name__,
                                instance_count=len(attr._instances),
                                memory_size=total_size / 1024 / 1024,
                                description=f"单例缓存 {attr.__name__} 包含 {len(attr._instances)} 个实例"
                            )
                            self.detected_leaks.append(leak_info)
                            print(
                                f"⚠️  发现缓存泄漏: {module_name}.{attr.__name__} ({len(attr._instances)} 个实例, {total_size / 1024 / 1024:.2f} MB)")

    def detect_prometheus_leaks(self):
        """检测Prometheus指标泄漏"""
        print("\n🔍 检测Prometheus指标泄漏...")

        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                collector_count = len(REGISTRY._names_to_collectors)
                if collector_count > 0:
                    leak_info = MemoryLeakInfo(
                        leak_type="prometheus_registry",
                        module_path="prometheus_client",
                        class_name="REGISTRY",
                        instance_count=collector_count,
                        memory_size=collector_count * 0.1,  # 估算每个指标0.1MB
                        description=f"Prometheus注册表包含 {collector_count} 个指标"
                    )
                    self.detected_leaks.append(leak_info)
                    print(f"⚠️  发现Prometheus泄漏: {collector_count} 个指标")

        except Exception as e:
            print(f"❌ 检测Prometheus泄漏失败: {e}")

    def detect_module_cache_leaks(self):
        """检测模块缓存泄漏"""
        print("\n🔍 检测模块缓存泄漏...")

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

        cached_count = 0
        for module_name in infrastructure_modules:
            if module_name in sys.modules:
                cached_count += 1

        if cached_count > 0:
            leak_info = MemoryLeakInfo(
                leak_type="module_cache",
                module_path="sys.modules",
                class_name="ModuleCache",
                instance_count=cached_count,
                memory_size=cached_count * 0.5,  # 估算每个模块0.5MB
                description=f"模块缓存包含 {cached_count} 个基础设施模块"
            )
            self.detected_leaks.append(leak_info)
            print(f"⚠️  发现模块缓存泄漏: {cached_count} 个模块")

    def detect_thread_leaks(self):
        """检测线程泄漏"""
        print("\n🔍 检测线程泄漏...")

        active_threads = threading.enumerate()
        monitor_threads = []

        for thread in active_threads:
            thread_name = thread.name.lower()
            if any(keyword in thread_name for keyword in ['monitor', 'watch', 'background', 'daemon', 'worker']):
                if thread.is_alive() and thread != threading.current_thread():
                    monitor_threads.append(thread)

        if monitor_threads:
            leak_info = MemoryLeakInfo(
                leak_type="thread_leak",
                module_path="threading",
                class_name="Thread",
                instance_count=len(monitor_threads),
                memory_size=len(monitor_threads) * 0.1,  # 估算每个线程0.1MB
                description=f"后台线程泄漏: {len(monitor_threads)} 个线程"
            )
            self.detected_leaks.append(leak_info)
            print(f"⚠️  发现线程泄漏: {len(monitor_threads)} 个后台线程")

    def detect_cache_leaks(self):
        """检测缓存泄漏"""
        print("\n🔍 检测缓存泄漏...")

        cache_modules = [
            ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
            ('src.infrastructure.logging.log_manager', 'LogManager'),
            ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
        ]

        for module_path, class_name in cache_modules:
            if module_path in sys.modules:
                module = sys.modules[module_path]
                cls = getattr(module, class_name, None)

                if cls is not None:
                    cache_attrs = ['_cache', '_metrics',
                                   '_instances', '_data', '_config', '_handlers']
                    for attr_name in cache_attrs:
                        if hasattr(cls, attr_name):
                            cache_obj = getattr(cls, attr_name)
                            if hasattr(cache_obj, '__len__') and len(cache_obj) > 0:
                                cache_size = len(cache_obj)
                                leak_info = MemoryLeakInfo(
                                    leak_type="cache_leak",
                                    module_path=module_path,
                                    class_name=f"{class_name}.{attr_name}",
                                    instance_count=cache_size,
                                    memory_size=cache_size * 0.01,  # 估算每个缓存项0.01MB
                                    description=f"缓存泄漏: {class_name}.{attr_name} 包含 {cache_size} 项"
                                )
                                self.detected_leaks.append(leak_info)
                                print(
                                    f"⚠️  发现缓存泄漏: {module_path}.{class_name}.{attr_name} ({cache_size} 项)")

    def run_comprehensive_detection(self):
        """运行全面检测"""
        print("🚀 开始全面内存泄漏检测")
        print("=" * 60)

        # 开始监控
        self.start_monitoring()

        # 执行各项检测
        self.detect_singleton_leaks()
        self.detect_prometheus_leaks()
        self.detect_module_cache_leaks()
        self.detect_thread_leaks()
        self.detect_cache_leaks()

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成检测报告"""
        print("\n📊 内存泄漏检测报告")
        print("=" * 60)

        if not self.detected_leaks:
            print("✅ 未检测到内存泄漏")
            return

        # 按类型分组
        leak_types = {}
        total_memory = 0

        for leak in self.detected_leaks:
            if leak.leak_type not in leak_types:
                leak_types[leak.leak_type] = []
            leak_types[leak.leak_type].append(leak)
            total_memory += leak.memory_size

        # 打印分组报告
        for leak_type, leaks in leak_types.items():
            print(f"\n🔍 {leak_type.upper()} 泄漏:")
            type_total = sum(leak.memory_size for leak in leaks)
            print(f"   总内存: {type_total:.2f} MB")
            print(f"   泄漏数量: {len(leaks)}")

            for leak in leaks:
                print(
                    f"   - {leak.module_path}.{leak.class_name}: {leak.memory_size:.2f} MB ({leak.description})")

        print(f"\n📈 总计:")
        print(f"   总泄漏数量: {len(self.detected_leaks)}")
        print(f"   总内存泄漏: {total_memory:.2f} MB")

        # 检查当前内存使用
        current_memory = self.process.memory_info().rss
        memory_diff = current_memory - self.initial_memory
        print(f"   当前内存增长: {memory_diff / 1024 / 1024:.2f} MB")

        # 提供修复建议
        self.provide_fix_suggestions()

    def provide_fix_suggestions(self):
        """提供修复建议"""
        print("\n🔧 修复建议:")
        print("=" * 60)

        if any(leak.leak_type == "singleton_instance" for leak in self.detected_leaks):
            print("1. 单例实例泄漏:")
            print("   - 在测试后清理单例实例")
            print("   - 使用 isolated_environment fixture")
            print("   - 手动设置 _instance = None")

        if any(leak.leak_type == "prometheus_registry" for leak in self.detected_leaks):
            print("2. Prometheus指标泄漏:")
            print("   - 使用隔离的CollectorRegistry")
            print("   - 清理REGISTRY._names_to_collectors")
            print("   - 设置PROMETHEUS_ISOLATED环境变量")

        if any(leak.leak_type == "module_cache" for leak in self.detected_leaks):
            print("3. 模块缓存泄漏:")
            print("   - 清理sys.modules中的基础设施模块")
            print("   - 使用importlib.reload重新加载模块")
            print("   - 避免大量import操作")

        if any(leak.leak_type == "thread_leak" for leak in self.detected_leaks):
            print("4. 线程泄漏:")
            print("   - 正确停止后台线程")
            print("   - 使用thread.join(timeout=0.5)")
            print("   - 设置daemon=True")

        if any(leak.leak_type == "cache_leak" for leak in self.detected_leaks):
            print("5. 缓存泄漏:")
            print("   - 清理所有缓存对象")
            print("   - 使用cache.clear()方法")
            print("   - 定期清理缓存")

        print("\n💡 推荐使用:")
        print("   - 激进内存清理策略")
        print("   - 内存隔离环境")
        print("   - 自动清理fixtures")

    def auto_fix_leaks(self):
        """自动修复检测到的泄漏"""
        print("\n🔧 开始自动修复内存泄漏...")

        from scripts.testing.aggressive_memory_fix import AggressiveMemoryFixer

        fixer = AggressiveMemoryFixer()
        fixer.run_aggressive_fix()

        # 重新检测
        print("\n🔍 修复后重新检测...")
        self.detected_leaks.clear()
        self.run_comprehensive_detection()

        if not self.detected_leaks:
            print("✅ 内存泄漏修复成功!")
        else:
            print("⚠️  仍有部分内存泄漏，建议手动检查")


def main():
    """主函数"""
    detector = ComprehensiveMemoryLeakDetector()

    # 运行全面检测
    detector.run_comprehensive_detection()

    # 询问是否自动修复
    if detector.detected_leaks:
        print("\n❓ 是否自动修复检测到的内存泄漏? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice in ['y', 'yes', '是']:
                detector.auto_fix_leaks()
        except KeyboardInterrupt:
            print("\n❌ 用户取消操作")
        except Exception as e:
            print(f"\n❌ 自动修复失败: {e}")


if __name__ == "__main__":
    main()
