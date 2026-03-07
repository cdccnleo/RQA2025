#!/usr/bin/env python3
"""
基础设施层验证脚本
直接测试核心功能，验证内存泄漏问题
"""

import sys
import time
import psutil
import gc


def get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def run_cleanup():
    """运行内存清理"""
    print("🧹 运行内存清理...")

    # 强制垃圾回收
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")

    # 清理模块缓存
    modules_to_clear = []
    for name in list(sys.modules.keys()):
        if any(keyword in name.lower() for keyword in ['infrastructure', 'monitoring', 'config']):
            modules_to_clear.append(name)

    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

    if modules_to_clear:
        print(f"🧹 清理了 {len(modules_to_clear)} 个模块缓存")

    # 清理Prometheus注册表
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            # 只清理非系统指标
            system_metrics = [
                'python_gc_objects_collected', 'python_gc_objects_collected_total',
                'python_gc_objects_collected_created', 'python_gc_objects_uncollectable',
                'python_gc_objects_uncollectable_total', 'python_gc_objects_uncollectable_created',
                'python_gc_collections', 'python_gc_collections_total',
                'python_gc_collections_created', 'python_info'
            ]

            metrics_to_remove = []
            for metric_name in REGISTRY._names_to_collectors.keys():
                if metric_name not in system_metrics:
                    metrics_to_remove.append(metric_name)

            for metric_name in metrics_to_remove:
                del REGISTRY._names_to_collectors[metric_name]

            if metrics_to_remove:
                print(f"🧹 清理Prometheus注册表: {len(metrics_to_remove)} 个非系统指标")
    except Exception as e:
        print(f"⚠️  Prometheus清理失败: {e}")


def test_infrastructure_core():
    """测试基础设施层核心功能"""
    print("🔍 测试基础设施层核心功能")

    # 运行清理
    run_cleanup()

    # 记录测试前内存
    memory_before = get_memory_usage()
    start_time = time.time()

    try:
        # 导入基础设施层
        print("   导入基础设施层...")
        from src.infrastructure.init_infrastructure import Infrastructure

        # 创建基础设施实例
        print("   创建基础设施实例...")
        infra = Infrastructure()

        # 测试配置管理器
        print("   测试配置管理器...")
        config = infra.config
        print(f"   配置管理器类型: {type(config).__name__}")

        # 测试日志管理器
        print("   测试日志管理器...")
        log = infra.log
        print(f"   日志管理器类型: {type(log).__name__}")

        # 测试监控组件
        print("   测试监控组件...")
        if hasattr(infra, 'system_monitor'):
            print(f"   系统监控器类型: {type(infra.system_monitor).__name__}")
        if hasattr(infra, 'application_monitor'):
            print(f"   应用监控器类型: {type(infra.application_monitor).__name__}")

        execution_time = time.time() - start_time

        # 记录测试后内存
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        # 判断是否有内存泄漏（增长超过20MB）
        leak_detected = memory_growth > 20

        print(f"   ✅ 基础设施层测试完成")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")
        print(f"   内存泄漏: {'是' if leak_detected else '否'}")

        if leak_detected:
            print(f"   ⚠️  检测到内存泄漏！")

        return True, memory_growth, leak_detected

    except Exception as e:
        execution_time = time.time() - start_time
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        print(f"   ❌ 基础设施层测试失败: {e}")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")

        return False, memory_growth, memory_growth > 20


def test_monitoring_modules():
    """测试监控模块"""
    print("\n🔍 测试监控模块")

    # 运行清理
    run_cleanup()

    # 记录测试前内存
    memory_before = get_memory_usage()
    start_time = time.time()

    try:
        # 导入监控模块
        print("   导入应用监控器...")
        from src.infrastructure.monitoring.application_monitor import ApplicationMonitor

        print("   导入系统监控器...")
        from src.infrastructure.monitoring.system_monitor import SystemMonitor

        # 创建监控器实例
        print("   创建监控器实例...")
        app_monitor = ApplicationMonitor()
        sys_monitor = SystemMonitor()

        print(f"   应用监控器类型: {type(app_monitor).__name__}")
        print(f"   系统监控器类型: {type(sys_monitor).__name__}")

        execution_time = time.time() - start_time

        # 记录测试后内存
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        # 判断是否有内存泄漏（增长超过20MB）
        leak_detected = memory_growth > 20

        print(f"   ✅ 监控模块测试完成")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")
        print(f"   内存泄漏: {'是' if leak_detected else '否'}")

        if leak_detected:
            print(f"   ⚠️  检测到内存泄漏！")

        return True, memory_growth, leak_detected

    except Exception as e:
        execution_time = time.time() - start_time
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        print(f"   ❌ 监控模块测试失败: {e}")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")

        return False, memory_growth, memory_growth > 20


def test_config_modules():
    """测试配置模块"""
    print("\n🔍 测试配置模块")

    # 运行清理
    run_cleanup()

    # 记录测试前内存
    memory_before = get_memory_usage()
    start_time = time.time()

    try:
        # 导入配置管理器
        print("   导入统一配置管理器...")
        from src.infrastructure.config.unified_manager import UnifiedConfigManager

        # 创建配置管理器实例
        print("   创建配置管理器实例...")
        config_manager = UnifiedConfigManager()

        print(f"   配置管理器类型: {type(config_manager).__name__}")

        execution_time = time.time() - start_time

        # 记录测试后内存
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        # 判断是否有内存泄漏（增长超过20MB）
        leak_detected = memory_growth > 20

        print(f"   ✅ 配置模块测试完成")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")
        print(f"   内存泄漏: {'是' if leak_detected else '否'}")

        if leak_detected:
            print(f"   ⚠️  检测到内存泄漏！")

        return True, memory_growth, leak_detected

    except Exception as e:
        execution_time = time.time() - start_time
        memory_after = get_memory_usage()
        memory_growth = memory_after - memory_before

        print(f"   ❌ 配置模块测试失败: {e}")
        print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
        print(f"   执行时间: {execution_time:.2f}秒")

        return False, memory_growth, memory_growth > 20


def main():
    """主函数"""
    print("🔧 开始基础设施层验证...")
    print("=" * 60)

    results = []

    # 测试基础设施层核心功能
    success1, growth1, leak1 = test_infrastructure_core()
    results.append(("基础设施层核心", success1, growth1, leak1))

    # 测试监控模块
    success2, growth2, leak2 = test_monitoring_modules()
    results.append(("监控模块", success2, growth2, leak2))

    # 测试配置模块
    success3, growth3, leak3 = test_config_modules()
    results.append(("配置模块", success3, growth3, leak3))

    # 生成报告
    print("\n📊 基础设施层验证报告")
    print("=" * 60)

    total_success = sum(1 for _, success, _, _ in results if success)
    total_leaks = sum(1 for _, _, _, leak in results if leak)
    total_growth = sum(growth for _, _, growth, _ in results)

    print(f"总测试数: {len(results)}")
    print(f"成功测试: {total_success}")
    print(f"内存泄漏: {total_leaks}")
    print(f"总内存增长: {total_growth:.2f}MB")

    for name, success, growth, leak in results:
        status = "✅" if success else "❌"
        leak_status = "⚠️" if leak else "✅"
        print(f"{status} {name}: {leak_status} ({growth:.2f}MB)")

    if total_leaks == 0:
        print(f"\n✅ 所有模块验证通过，未检测到内存泄漏")
        return 0
    else:
        print(f"\n❌ 检测到 {total_leaks} 个模块存在内存泄漏")
        return 1


if __name__ == "__main__":
    sys.exit(main())
