#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存泄漏检测和修复脚本

检测和修复基础设施层测试用例中的内存泄漏问题，包括：
1. import时的全局注册
2. 单例模式的缓存问题
3. 监控模块的Prometheus指标注册
4. 线程池和连接池未正确清理
5. 配置管理器的缓存问题
"""

import gc
import sys
import psutil
import time
from pathlib import Path
import importlib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MemoryLeakDetector:
    """内存泄漏检测器"""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.memory_snapshots = []
        self.singleton_instances = {}
        self.global_registries = {}

    def start_monitoring(self):
        """开始监控内存使用"""
        self.baseline_memory = self.process.memory_info().rss
        print(f"开始监控内存使用，基准内存: {self.baseline_memory / 1024 / 1024:.2f} MB")

    def take_snapshot(self, label: str):
        """记录内存快照"""
        current_memory = self.process.memory_info().rss
        memory_diff = current_memory - self.baseline_memory
        snapshot = {
            'label': label,
            'memory_mb': current_memory / 1024 / 1024,
            'diff_mb': memory_diff / 1024 / 1024,
            'timestamp': time.time()
        }
        self.memory_snapshots.append(snapshot)
        print(f"内存快照 [{label}]: {snapshot['memory_mb']:.2f} MB (变化: {snapshot['diff_mb']:+.2f} MB)")

    def analyze_memory_growth(self):
        """分析内存增长情况"""
        if not self.memory_snapshots:
            return

        print("\n=== 内存增长分析 ===")
        for i, snapshot in enumerate(self.memory_snapshots):
            if i > 0:
                prev = self.memory_snapshots[i-1]
                growth = snapshot['memory_mb'] - prev['memory_mb']
                print(f"{prev['label']} -> {snapshot['label']}: {growth:+.2f} MB")

        total_growth = self.memory_snapshots[-1]['diff_mb']
        print(f"\n总内存增长: {total_growth:+.2f} MB")

        if total_growth > 50:  # 超过50MB认为有泄漏
            print("⚠️  检测到显著内存增长，可能存在内存泄漏")


class SingletonCleaner:
    """单例清理器"""

    def __init__(self):
        self.singleton_classes = [
            'src.infrastructure.init_infrastructure.Infrastructure',
            'src.infrastructure.config.unified_manager.UnifiedConfigManager',
            'src.infrastructure.monitoring.application_monitor.ApplicationMonitor',
            'src.infrastructure.monitoring.system_monitor.SystemMonitor',
            'src.infrastructure.logging.log_manager.LogManager',
            'src.infrastructure.error.error_handler.ErrorHandler',
            'src.infrastructure.resource.resource_manager.ResourceManager',
            'src.infrastructure.cache.memory_cache_manager.MemoryCacheManager',
        ]

    def cleanup_singletons(self):
        """清理所有单例实例"""
        print("\n=== 清理单例实例 ===")

        for class_path in self.singleton_classes:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                if hasattr(cls, '_instance') and cls._instance is not None:
                    print(f"清理 {class_name} 单例实例")
                    cls._instance = None

                if hasattr(cls, '_instances'):
                    print(f"清理 {class_name} 多实例缓存")
                    cls._instances.clear()

            except Exception as e:
                print(f"清理 {class_path} 失败: {e}")

    def cleanup_global_registries(self):
        """清理全局注册表"""
        print("\n=== 清理全局注册表 ===")

        # 清理Prometheus注册表
        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                original_size = len(REGISTRY._names_to_collectors)
                REGISTRY._names_to_collectors.clear()
                print(f"清理Prometheus注册表: {original_size} 个指标")
        except Exception as e:
            print(f"清理Prometheus注册表失败: {e}")

        # 清理依赖注入容器
        try:
            from src.infrastructure.di.container import DependencyContainer
            if hasattr(DependencyContainer, '_instance'):
                DependencyContainer._instance = None
                print("清理依赖注入容器")
        except Exception as e:
            print(f"清理依赖注入容器失败: {e}")

        # 清理数据注册表
        try:
            from src.data.registry import DataRegistry
            if hasattr(DataRegistry, '_instance'):
                DataRegistry._instance = None
                print("清理数据注册表")
        except Exception as e:
            print(f"清理数据注册表失败: {e}")


class ConfigManagerCleaner:
    """配置管理器清理器"""

    def __init__(self):
        self.config_managers = [
            'src.infrastructure.config.unified_manager.UnifiedConfigManager',
            'src.infrastructure.config.services.cache_service.CacheService',
        ]

    def cleanup_config_cache(self):
        """清理配置缓存"""
        print("\n=== 清理配置缓存 ===")

        for class_path in self.config_managers:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                # 清理全局实例
                if hasattr(module, '_unified_manager_instance'):
                    module._unified_manager_instance = None
                    print(f"清理 {class_name} 全局实例")

                # 清理缓存
                if hasattr(cls, '_cache'):
                    cls._cache.clear()
                    print(f"清理 {class_name} 缓存")

            except Exception as e:
                print(f"清理 {class_path} 失败: {e}")


class MonitoringCleaner:
    """监控模块清理器"""

    def __init__(self):
        self.monitoring_modules = [
            'src.infrastructure.monitoring.application_monitor.ApplicationMonitor',
            'src.infrastructure.monitoring.system_monitor.SystemMonitor',
            'src.infrastructure.monitoring.metrics_collector.MetricsCollector',
        ]

    def cleanup_monitoring(self):
        """清理监控模块"""
        print("\n=== 清理监控模块 ===")

        for class_path in self.monitoring_modules:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                # 清理全局实例
                if hasattr(module, '_metrics_collector_instance'):
                    module._metrics_collector_instance = None
                    print(f"清理 {class_name} 全局实例")

                # 清理监控数据
                if hasattr(cls, '_metrics'):
                    cls._metrics.clear()
                    print(f"清理 {class_name} 监控数据")

                # 停止监控线程
                if hasattr(cls, '_monitor_thread') and cls._monitor_thread:
                    cls._monitor_thread.join(timeout=1)
                    print(f"停止 {class_name} 监控线程")

            except Exception as e:
                print(f"清理 {class_path} 失败: {e}")


class ThreadPoolCleaner:
    """线程池清理器"""

    def __init__(self):
        self.thread_pools = []

    def register_thread_pool(self, pool):
        """注册线程池"""
        self.thread_pools.append(pool)

    def cleanup_thread_pools(self):
        """清理所有线程池"""
        print("\n=== 清理线程池 ===")

        for pool in self.thread_pools:
            try:
                if hasattr(pool, 'shutdown'):
                    pool.shutdown(wait=False)
                    print("关闭线程池")
                elif hasattr(pool, 'close'):
                    pool.close()
                    print("关闭线程池")
            except Exception as e:
                print(f"关闭线程池失败: {e}")

        self.thread_pools.clear()


class InfrastructureTestOptimizer:
    """基础设施测试优化器"""

    def __init__(self):
        self.detector = MemoryLeakDetector()
        self.singleton_cleaner = SingletonCleaner()
        self.config_cleaner = ConfigManagerCleaner()
        self.monitoring_cleaner = MonitoringCleaner()
        self.thread_cleaner = ThreadPoolCleaner()

    def run_memory_leak_detection(self):
        """运行内存泄漏检测"""
        print("🔍 开始基础设施层内存泄漏检测")

        # 开始监控
        self.detector.start_monitoring()

        # 测试前快照
        self.detector.take_snapshot("测试前")

        # 模拟测试场景
        self._simulate_test_scenarios()

        # 测试后快照
        self.detector.take_snapshot("测试后")

        # 清理资源
        self._cleanup_resources()

        # 清理后快照
        self.detector.take_snapshot("清理后")

        # 分析结果
        self.detector.analyze_memory_growth()

    def _simulate_test_scenarios(self):
        """模拟测试场景"""
        print("\n=== 模拟测试场景 ===")

        # 模拟导入基础设施模块
        print("1. 导入基础设施模块")
        try:
            self.detector.take_snapshot("导入基础设施")
        except Exception as e:
            print(f"导入基础设施失败: {e}")

        # 模拟配置管理器
        print("2. 测试配置管理器")
        try:
            from src.infrastructure.config import get_unified_config_manager
            config_manager = get_unified_config_manager()
            self.detector.take_snapshot("配置管理器")
        except Exception as e:
            print(f"配置管理器测试失败: {e}")

        # 模拟监控模块
        print("3. 测试监控模块")
        try:
            from src.infrastructure.monitoring import ApplicationMonitor
            monitor = ApplicationMonitor(skip_thread=True)
            self.detector.take_snapshot("监控模块")
        except Exception as e:
            print(f"监控模块测试失败: {e}")

        # 模拟日志模块
        print("4. 测试日志模块")
        try:
            from src.infrastructure.logging import get_log_manager
            log_manager = get_log_manager()
            self.detector.take_snapshot("日志模块")
        except Exception as e:
            print(f"日志模块测试失败: {e}")

        # 模拟错误处理模块
        print("5. 测试错误处理模块")
        try:
            from src.infrastructure.error import get_error_handler
            error_handler = get_error_handler()
            self.detector.take_snapshot("错误处理模块")
        except Exception as e:
            print(f"错误处理模块测试失败: {e}")

    def _cleanup_resources(self):
        """清理资源"""
        print("\n=== 清理资源 ===")

        # 强制垃圾回收
        gc.collect()

        # 清理单例
        self.singleton_cleaner.cleanup_singletons()
        self.singleton_cleaner.cleanup_global_registries()

        # 清理配置缓存
        self.config_cleaner.cleanup_config_cache()

        # 清理监控模块
        self.monitoring_cleaner.cleanup_monitoring()

        # 清理线程池
        self.thread_cleaner.cleanup_thread_pools()

        # 再次强制垃圾回收
        gc.collect()

    def generate_fix_recommendations(self):
        """生成修复建议"""
        print("\n=== 修复建议 ===")

        recommendations = [
            "1. 在测试用例的teardown方法中添加单例清理",
            "2. 使用隔离的Prometheus注册表避免指标重复注册",
            "3. 在测试前清理配置缓存",
            "4. 确保监控线程在测试后正确停止",
            "5. 使用weakref避免循环引用",
            "6. 在import时避免执行全局注册操作",
            "7. 使用工厂模式替代单例模式",
            "8. 添加内存使用监控和告警",
        ]

        for rec in recommendations:
            print(rec)

    def create_test_fixtures(self):
        """创建测试fixtures"""
        print("\n=== 创建测试Fixtures ===")

        fixture_code = '''
import pytest
import gc
from typing import Generator

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """自动清理单例实例"""
    yield
    # 清理单例
    try:
        from src.infrastructure.init_infrastructure import Infrastructure
        Infrastructure._instance = None
    except:
        pass
    
    try:
        from src.infrastructure.config.unified_manager import UnifiedConfigManager
        UnifiedConfigManager._instance = None
    except:
        pass
    
    # 清理Prometheus注册表
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            REGISTRY._names_to_collectors.clear()
    except:
        pass
    
    # 强制垃圾回收
    gc.collect()

@pytest.fixture
def isolated_registry():
    """提供隔离的Prometheus注册表"""
    from prometheus_client import CollectorRegistry
    return CollectorRegistry()

@pytest.fixture
def clean_config_manager():
    """提供清理过的配置管理器"""
    from src.infrastructure.config import get_unified_config_manager
    manager = get_unified_config_manager()
    yield manager
    # 清理配置缓存
    if hasattr(manager, '_core'):
        manager._core.clear_cache()
'''

        fixture_file = project_root / "tests" / "unit" / "infrastructure" / "conftest.py"

        with open(fixture_file, 'w', encoding='utf-8') as f:
            f.write(fixture_code)

        print(f"创建测试fixtures文件: {fixture_file}")


def main():
    """主函数"""
    print("🚀 基础设施层内存泄漏检测和修复工具")
    print("=" * 50)

    optimizer = InfrastructureTestOptimizer()

    # 运行内存泄漏检测
    optimizer.run_memory_leak_detection()

    # 生成修复建议
    optimizer.generate_fix_recommendations()

    # 创建测试fixtures
    optimizer.create_test_fixtures()

    print("\n✅ 内存泄漏检测完成")


if __name__ == "__main__":
    main()
