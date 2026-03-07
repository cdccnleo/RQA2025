#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级内存泄漏修复脚本

避免在修复过程中产生内存暴涨，采用渐进式清理策略：
1. 避免大量import操作
2. 使用弱引用和延迟清理
3. 分步骤执行清理
4. 监控内存使用
5. 提供详细的修复报告
"""

import sys
import gc
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class LightweightMemoryFixer:
    """轻量级内存泄漏修复器"""

    def __init__(self):
        self.fixes_applied = []
        self.cleaned_instances = set()

    def safe_import_and_clean(self, module_path: str, class_name: str, instance_var: str = '_instance'):
        """安全导入并清理单例"""
        try:
            # 使用__import__避免大量import
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            if hasattr(cls, instance_var):
                setattr(cls, instance_var, None)
                self.cleaned_instances.add(f"{module_path}.{class_name}.{instance_var}")
                print(f"✅ 清理 {class_name}.{instance_var}")

        except Exception as e:
            print(f"❌ 清理 {module_path}.{class_name} 失败: {e}")

    def cleanup_singletons_step_by_step(self):
        """分步骤清理单例实例"""
        print("🧹 分步骤清理单例实例")

        # 第一步：清理基础设施核心
        print("\n步骤1: 清理基础设施核心")
        core_singletons = [
            ('src.infrastructure.init_infrastructure', 'Infrastructure'),
            ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ]

        for module_path, class_name in core_singletons:
            self.safe_import_and_clean(module_path, class_name)
            time.sleep(0.1)  # 短暂等待

        # 第二步：清理监控模块
        print("\n步骤2: 清理监控模块")
        monitoring_singletons = [
            ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
            ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
        ]

        for module_path, class_name in monitoring_singletons:
            self.safe_import_and_clean(module_path, class_name, '_instances')
            time.sleep(0.1)

        # 第三步：清理日志模块
        print("\n步骤3: 清理日志模块")
        logging_singletons = [
            ('src.infrastructure.logging.log_manager', 'LogManager'),
            ('src.infrastructure.logging.trading_logger', 'TradingLogger'),
        ]

        for module_path, class_name in logging_singletons:
            self.safe_import_and_clean(module_path, class_name)
            time.sleep(0.1)

        # 第四步：清理其他模块
        print("\n步骤4: 清理其他模块")
        other_singletons = [
            ('src.infrastructure.error.error_handler', 'ErrorHandler'),
            ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
            ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
        ]

        for module_path, class_name in other_singletons:
            self.safe_import_and_clean(module_path, class_name)
            time.sleep(0.1)

    def cleanup_global_instances(self):
        """清理全局实例变量"""
        print("\n🔧 清理全局实例变量")

        global_instances = [
            ('src.infrastructure.config.unified_manager', '_unified_manager_instance'),
            ('src.infrastructure.monitoring.metrics_collector', '_metrics_collector_instance'),
            ('src.infrastructure.config.services.unified_hot_reload', '_hot_reload_instance'),
            ('src.infrastructure.config.services.unified_sync', '_sync_instance'),
        ]

        for module_path, var_name in global_instances:
            try:
                module = __import__(module_path, fromlist=[var_name])
                if hasattr(module, var_name):
                    setattr(module, var_name, None)
                    self.cleaned_instances.add(f"{module_path}.{var_name}")
                    print(f"✅ 清理全局实例: {module_path}.{var_name}")
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ 清理全局实例失败 {module_path}.{var_name}: {e}")

    def safe_prometheus_cleanup(self):
        """安全清理Prometheus注册表"""
        print("\n🔧 安全清理Prometheus注册表")

        try:
            # 只导入必要的模块
            prometheus_module = __import__('prometheus_client', fromlist=['REGISTRY'])
            registry = getattr(prometheus_module, 'REGISTRY')

            if hasattr(registry, '_names_to_collectors'):
                original_size = len(registry._names_to_collectors)
                registry._names_to_collectors.clear()
                print(f"✅ 清理Prometheus注册表: {original_size} 个指标")

        except Exception as e:
            print(f"❌ 清理Prometheus注册表失败: {e}")

    def safe_cache_cleanup(self):
        """安全清理缓存"""
        print("\n🧹 安全清理缓存")

        try:
            # 清理配置缓存
            config_module = __import__('src.infrastructure.config.unified_manager', fromlist=[
                                       'get_unified_config_manager'])
            get_config_manager = getattr(config_module, 'get_unified_config_manager')
            config_manager = get_config_manager()

            if hasattr(config_manager, '_core'):
                config_manager._core.clear_cache()
                print("✅ 清理配置管理器缓存")

        except Exception as e:
            print(f"❌ 清理配置缓存失败: {e}")

    def safe_module_cache_cleanup(self):
        """安全清理模块缓存"""
        print("\n🧹 安全清理模块缓存")

        # 只清理关键模块
        key_modules = [
            'src.infrastructure.config.unified_manager',
            'src.infrastructure.monitoring.application_monitor',
            'src.infrastructure.monitoring.system_monitor',
        ]

        cleaned_count = 0
        for module_name in key_modules:
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                    cleaned_count += 1
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ 清理模块缓存失败 {module_name}: {e}")

        print(f"✅ 清理了 {cleaned_count} 个关键模块缓存")

    def gentle_garbage_collection(self):
        """温和的垃圾回收"""
        print("\n💪 温和的垃圾回收")

        # 分多次进行垃圾回收
        for i in range(3):
            collected = gc.collect()
            print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")
            time.sleep(0.2)  # 给系统时间释放内存

    def create_lightweight_cleanup_script(self):
        """创建轻量级清理脚本"""
        print("\n📝 创建轻量级清理脚本")

        cleanup_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级内存清理脚本

避免内存暴涨的渐进式清理策略
"""

import gc
import sys
import time
from typing import Dict, Any

def lightweight_cleanup():
    """轻量级内存清理"""
    print("🧹 开始轻量级内存清理...")
    
    # 步骤1: 清理核心单例
    cleanup_core_singletons()
    time.sleep(0.1)
    
    # 步骤2: 清理监控模块
    cleanup_monitoring_singletons()
    time.sleep(0.1)
    
    # 步骤3: 清理全局实例
    cleanup_global_instances()
    time.sleep(0.1)
    
    # 步骤4: 清理Prometheus注册表
    cleanup_prometheus_registry()
    time.sleep(0.1)
    
    # 步骤5: 温和垃圾回收
    gentle_garbage_collection()
    
    print("✅ 轻量级内存清理完成")

def cleanup_core_singletons():
    """清理核心单例"""
    core_singletons = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
    ]
    
    for module_path, class_name in core_singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instance'):
                cls._instance = None
                print(f"✅ 清理 {class_name} 单例")
        except Exception:
            pass

def cleanup_monitoring_singletons():
    """清理监控模块单例"""
    monitoring_singletons = [
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
    ]
    
    for module_path, class_name in monitoring_singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instances'):
                cls._instances.clear()
                print(f"✅ 清理 {class_name} 实例缓存")
        except Exception:
            pass

def cleanup_global_instances():
    """清理全局实例"""
    global_instances = [
        ('src.infrastructure.config.unified_manager', '_unified_manager_instance'),
        ('src.infrastructure.monitoring.metrics_collector', '_metrics_collector_instance'),
    ]
    
    for module_path, var_name in global_instances:
        try:
            module = __import__(module_path, fromlist=[var_name])
            if hasattr(module, var_name):
                setattr(module, var_name, None)
                print(f"✅ 清理全局实例: {var_name}")
        except Exception:
            pass

def cleanup_prometheus_registry():
    """清理Prometheus注册表"""
    try:
        prometheus_module = __import__('prometheus_client', fromlist=['REGISTRY'])
        registry = getattr(prometheus_module, 'REGISTRY')
        if hasattr(registry, '_names_to_collectors'):
            registry._names_to_collectors.clear()
            print("✅ 清理Prometheus注册表")
    except Exception:
        pass

def gentle_garbage_collection():
    """温和的垃圾回收"""
    for i in range(3):
        collected = gc.collect()
        print(f"✅ 第{i+1}次垃圾回收: {collected} 个对象")
        time.sleep(0.1)

if __name__ == "__main__":
    lightweight_cleanup()
'''

        cleanup_file = project_root / "scripts" / "testing" / "lightweight_memory_cleanup.py"

        with open(cleanup_file, 'w', encoding='utf-8') as f:
            f.write(cleanup_script)

        print(f"✅ 创建轻量级内存清理脚本: {cleanup_file}")

    def create_test_fixtures(self):
        """创建测试fixtures"""
        print("\n📝 创建测试fixtures")

        fixture_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级测试fixtures

避免内存暴涨的测试环境配置
"""

import pytest
import gc
import time
from typing import Generator

@pytest.fixture(autouse=True)
def lightweight_cleanup():
    """轻量级自动清理"""
    yield
    # 测试后清理
    cleanup_core_singletons()
    cleanup_prometheus_registry()
    gentle_garbage_collection()

def cleanup_core_singletons():
    """清理核心单例"""
    core_singletons = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
    ]
    
    for module_path, class_name in core_singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instance'):
                cls._instance = None
        except Exception:
            pass

def cleanup_prometheus_registry():
    """清理Prometheus注册表"""
    try:
        prometheus_module = __import__('prometheus_client', fromlist=['REGISTRY'])
        registry = getattr(prometheus_module, 'REGISTRY')
        if hasattr(registry, '_names_to_collectors'):
            registry._names_to_collectors.clear()
    except Exception:
        pass

def gentle_garbage_collection():
    """温和的垃圾回收"""
    for _ in range(2):
        gc.collect()
        time.sleep(0.1)

@pytest.fixture
def isolated_registry():
    """提供隔离的Prometheus注册表"""
    try:
        prometheus_module = __import__('prometheus_client', fromlist=['CollectorRegistry'])
        CollectorRegistry = getattr(prometheus_module, 'CollectorRegistry')
        return CollectorRegistry()
    except Exception:
        return None

@pytest.fixture
def clean_config_manager():
    """提供清理过的配置管理器"""
    try:
        config_module = __import__('src.infrastructure.config.unified_manager', fromlist=['get_unified_config_manager'])
        get_config_manager = getattr(config_module, 'get_unified_config_manager')
        manager = get_config_manager()
        yield manager
        # 清理配置缓存
        if hasattr(manager, '_core'):
            manager._core.clear_cache()
    except Exception:
        yield None
'''

        fixture_file = project_root / "tests" / "unit" / "infrastructure" / "lightweight_fixtures.py"

        with open(fixture_file, 'w', encoding='utf-8') as f:
            f.write(fixture_code)

        print(f"✅ 创建轻量级测试fixtures: {fixture_file}")

    def run_lightweight_fix(self):
        """运行轻量级修复"""
        print("🚀 开始轻量级内存泄漏修复")
        print("=" * 50)

        # 分步骤执行清理
        self.cleanup_singletons_step_by_step()
        self.cleanup_global_instances()
        self.safe_prometheus_cleanup()
        self.safe_cache_cleanup()
        self.safe_module_cache_cleanup()
        self.gentle_garbage_collection()

        # 创建辅助脚本
        self.create_lightweight_cleanup_script()
        self.create_test_fixtures()

        print(f"\n✅ 轻量级内存泄漏修复完成")
        print(f"清理的实例数量: {len(self.cleaned_instances)}")
        print(f"清理的实例: {', '.join(list(self.cleaned_instances)[:10])}...")

        print("\n📋 使用说明:")
        print("1. 运行轻量级清理: python scripts/testing/lightweight_memory_cleanup.py")
        print("2. 在测试中使用fixtures: from tests.unit.infrastructure.lightweight_fixtures import lightweight_cleanup")
        print("3. 避免大量import操作，使用渐进式清理策略")


def main():
    """主函数"""
    fixer = LightweightMemoryFixer()
    fixer.run_lightweight_fix()


if __name__ == "__main__":
    main()
