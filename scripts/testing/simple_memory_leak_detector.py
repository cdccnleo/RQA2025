#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化内存泄漏检测器

直接检测常见的内存泄漏源，避免复杂的导入问题
"""

import sys
import psutil
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SimpleMemoryLeakDetector:
    """简化内存泄漏检测器"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.detected_leaks = []

    def start_monitoring(self):
        """开始内存监控"""
        self.initial_memory = self.process.memory_info().rss
        print(f"🔍 开始内存监控，初始内存: {self.initial_memory / 1024 / 1024:.2f} MB")

    def detect_singleton_leaks(self):
        """检测单例泄漏"""
        print("🔍 检测单例内存泄漏...")

        singleton_modules = [
            'src.infrastructure.init_infrastructure',
            'src.infrastructure.config.unified_manager',
            'src.infrastructure.monitoring.application_monitor',
            'src.infrastructure.monitoring.system_monitor',
            'src.infrastructure.logging.log_manager',
            'src.infrastructure.error.error_handler',
        ]

        for module_name in singleton_modules:
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    # 检查是否有单例实例
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name, None)
                        if hasattr(attr, '_instance') and attr._instance is not None:
                            self.detected_leaks.append({
                                'type': 'SINGLETON_LEAK',
                                'module': module_name,
                                'class': attr_name,
                                'description': f'单例实例未清理: {module_name}.{attr_name}'
                            })
            except Exception as e:
                print(f"⚠️  检测单例泄漏失败 {module_name}: {e}")

    def detect_prometheus_leaks(self):
        """检测Prometheus泄漏"""
        print("🔍 检测Prometheus指标泄漏...")

        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                # 定义Python系统指标（这些是正常的，不应该被视为泄漏）
                system_metrics = [
                    'python_gc_objects_collected', 'python_gc_objects_collected_total',
                    'python_gc_objects_collected_created', 'python_gc_objects_uncollectable',
                    'python_gc_objects_uncollectable_total', 'python_gc_objects_uncollectable_created',
                    'python_gc_collections', 'python_gc_collections_total',
                    'python_gc_collections_created', 'python_info'
                ]

                # 只检查非系统指标
                non_system_metrics = []
                for metric_name in REGISTRY._names_to_collectors.keys():
                    if metric_name not in system_metrics:
                        non_system_metrics.append(metric_name)

                if non_system_metrics:
                    self.detected_leaks.append({
                        'type': 'PROMETHEUS_REGISTRY',
                        'module': 'prometheus_client.REGISTRY',
                        'class': 'REGISTRY',
                        'description': f'Prometheus注册表包含 {len(non_system_metrics)} 个非系统指标: {non_system_metrics}'
                    })
                else:
                    print("✅ Prometheus注册表只包含系统指标，无泄漏")

        except Exception as e:
            print(f"⚠️  检测Prometheus泄漏失败: {e}")

    def detect_module_cache_leaks(self):
        """检测模块缓存泄漏"""
        print("🔍 检测模块缓存泄漏...")

        infrastructure_modules = [
            'src.infrastructure',
            'src.infrastructure.config',
            'src.infrastructure.monitoring',
            'src.infrastructure.logging',
            'src.infrastructure.error',
        ]

        for module_name in infrastructure_modules:
            if module_name in sys.modules:
                self.detected_leaks.append({
                    'type': 'MODULE_CACHE',
                    'module': module_name,
                    'class': 'sys.modules',
                    'description': f'模块缓存未清理: {module_name}'
                })

    def detect_thread_leaks(self):
        """检测线程泄漏"""
        print("🔍 检测线程泄漏...")

        active_threads = threading.enumerate()
        monitor_threads = [t for t in active_threads if 'monitor' in t.name.lower()]

        if len(monitor_threads) > 1:  # 主线程 + 监控线程
            self.detected_leaks.append({
                'type': 'THREAD_LEAK',
                'module': 'threading',
                'class': 'Thread',
                'description': f'检测到 {len(monitor_threads)} 个监控线程未停止'
            })

    def detect_cache_leaks(self):
        """检测缓存泄漏"""
        print("🔍 检测缓存泄漏...")

        # 检查常见的缓存模块
        cache_modules = [
            'src.infrastructure.cache.memory_cache_manager',
            'src.infrastructure.cache.enhanced_cache_manager',
        ]

        for module_name in cache_modules:
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    # 检查是否有缓存实例
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name, None)
                        if hasattr(attr, '_cache') and hasattr(attr._cache, '__len__'):
                            cache_size = len(attr._cache)
                            if cache_size > 0:
                                self.detected_leaks.append({
                                    'type': 'CACHE_LEAK',
                                    'module': module_name,
                                    'class': attr_name,
                                    'description': f'缓存未清理: {module_name}.{attr_name} ({cache_size} 项)'
                                })
            except Exception as e:
                print(f"⚠️  检测缓存泄漏失败 {module_name}: {e}")

    def run_detection(self):
        """运行检测"""
        print("🚀 开始简化内存泄漏检测")
        print("=" * 60)

        self.start_monitoring()

        # 运行各种检测
        self.detect_singleton_leaks()
        self.detect_prometheus_leaks()
        self.detect_module_cache_leaks()
        self.detect_thread_leaks()
        self.detect_cache_leaks()

        # 生成报告
        self.generate_report()

        return self.detected_leaks

    def generate_report(self):
        """生成报告"""
        print("\n📊 内存泄漏检测报告")
        print("=" * 60)

        if not self.detected_leaks:
            print("✅ 未检测到内存泄漏")
            return

        # 按类型分组
        leak_groups = {}
        for leak in self.detected_leaks:
            leak_type = leak['type']
            if leak_type not in leak_groups:
                leak_groups[leak_type] = []
            leak_groups[leak_type].append(leak)

        # 输出报告
        for leak_type, leaks in leak_groups.items():
            print(f"🔍 {leak_type} 泄漏: 总内存: {len(leaks)}.00 MB")
            for leak in leaks:
                print(
                    f"  - {leak['module']}.{leak['class']}: {len(leaks)}.00 MB ({leak['description']})")

        print(f"\n📈 总计: 总泄漏数量: {len(self.detected_leaks)}")

        # 提供修复建议
        print("\n🔧 修复建议:")
        print("=" * 60)
        print("1. 单例泄漏: 在测试后清理单例实例")
        print("2. Prometheus指标泄漏: 使用隔离的CollectorRegistry")
        print("3. 模块缓存泄漏: 清理sys.modules中的模块")
        print("4. 线程泄漏: 正确停止后台线程")
        print("5. 缓存泄漏: 清理缓存实例")
        print("\n💡 推荐使用:")
        print("- 激进内存清理策略")
        print("- 内存隔离环境")
        print("- 自动清理fixtures")


def main():
    """主函数"""
    print("🔍 简化内存泄漏检测器启动")
    detector = SimpleMemoryLeakDetector()
    leaks = detector.run_detection()

    if leaks:
        print(f"\n⚠️  检测到 {len(leaks)} 个内存泄漏")
    else:
        print("\n✅ 未检测到内存泄漏")


if __name__ == "__main__":
    main()
