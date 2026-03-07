#!/usr/bin/env python3
"""
详细内存分析脚本
分析EnhancedDataIntegrationManager各个组件的内存使用情况
"""

from src.data.quality.monitor import DataQualityMonitor
from src.data.cache.cache_manager import CacheManager, CacheConfig
from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager, DataStreamConfig
import sys
import gc
import time
import psutil
import tracemalloc
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_memory_info():
    """获取当前进程的内存信息"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }


def analyze_component_memory():
    """分析各个组件的内存使用情况"""
    print("🔍 开始详细内存分析...")

    # 启动内存跟踪
    tracemalloc.start()

    # 记录初始状态
    initial_memory = get_memory_info()
    initial_snapshot = tracemalloc.take_snapshot()

    print(f"📊 初始内存状态:")
    print(f"  - RSS: {initial_memory['rss']:.2f} MB")
    print(f"  - VMS: {initial_memory['vms']:.2f} MB")
    print(f"  - 内存占比: {initial_memory['percent']:.2f}%")

    # 测试各个组件
    components = {}

    # 1. 测试CacheManager
    print("\n🔧 测试CacheManager...")
    cache_start = get_memory_info()
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    cache_end = get_memory_info()
    components['CacheManager'] = {
        'start': cache_start,
        'end': cache_end,
        'increase': cache_end['rss'] - cache_start['rss']
    }
    print(f"  CacheManager内存增加: {components['CacheManager']['increase']:.2f} MB")

    # 2. 测试DataQualityMonitor
    print("🔧 测试DataQualityMonitor...")
    monitor_start = get_memory_info()
    quality_monitor = DataQualityMonitor()
    monitor_end = get_memory_info()
    components['DataQualityMonitor'] = {
        'start': monitor_start,
        'end': monitor_end,
        'increase': monitor_end['rss'] - monitor_start['rss']
    }
    print(f"  DataQualityMonitor内存增加: {components['DataQualityMonitor']['increase']:.2f} MB")

    # 3. 测试EnhancedDataIntegrationManager
    print("🔧 测试EnhancedDataIntegrationManager...")
    manager_start = get_memory_info()
    manager = EnhancedDataIntegrationManager()
    manager_end = get_memory_info()
    components['EnhancedDataIntegrationManager'] = {
        'start': manager_start,
        'end': manager_end,
        'increase': manager_end['rss'] - manager_start['rss']
    }
    print(
        f"  EnhancedDataIntegrationManager内存增加: {components['EnhancedDataIntegrationManager']['increase']:.2f} MB")

    # 4. 测试管理器操作
    print("🔧 测试管理器操作...")
    operations_start = get_memory_info()

    # 添加节点
    manager.register_node("node1", "192.168.1.100", 8080)
    manager.register_node("node2", "192.168.1.101", 8081)

    # 创建数据流
    stream_config1 = DataStreamConfig("stream1", "stock_data")
    stream_config2 = DataStreamConfig("stream2", "market_data")
    stream1_id = manager.create_data_stream(stream_config1)
    stream2_id = manager.create_data_stream(stream_config2)

    # 添加性能指标
    manager.performance_monitor.record_metric("load_time", 0.5)
    manager.performance_monitor.record_metric("cache_hit_rate", 0.85)

    # 触发告警
    manager.alert_manager.trigger_alert("high_memory_usage", "Memory usage exceeded 80%")

    operations_end = get_memory_info()
    components['Operations'] = {
        'start': operations_start,
        'end': operations_end,
        'increase': operations_end['rss'] - operations_start['rss']
    }
    print(f"  操作内存增加: {components['Operations']['increase']:.2f} MB")

    # 5. 测试关闭和清理
    print("🔧 测试关闭和清理...")
    cleanup_start = get_memory_info()

    # 停止数据流
    manager.stop_data_stream(stream1_id)
    manager.stop_data_stream(stream2_id)

    # 关闭管理器
    manager.shutdown()

    cleanup_end = get_memory_info()
    components['Cleanup'] = {
        'start': cleanup_start,
        'end': cleanup_end,
        'increase': cleanup_end['rss'] - cleanup_start['rss']
    }
    print(f"  清理内存变化: {components['Cleanup']['increase']:.2f} MB")

    # 6. 强制垃圾回收
    print("🔧 强制垃圾回收...")
    gc_start = get_memory_info()
    gc.collect()
    gc_end = get_memory_info()
    components['GarbageCollection'] = {
        'start': gc_start,
        'end': gc_end,
        'increase': gc_end['rss'] - gc_start['rss']
    }
    print(f"  垃圾回收内存变化: {components['GarbageCollection']['increase']:.2f} MB")

    # 最终状态
    final_memory = get_memory_info()
    final_snapshot = tracemalloc.take_snapshot()

    # 分析内存差异
    top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')

    print("\n📈 详细内存分析报告:")
    print("=" * 60)

    # 组件内存使用统计
    print("\n🔧 各组件内存使用情况:")
    total_increase = 0
    for component, data in components.items():
        increase = data['increase']
        total_increase += increase
        status = "✅" if increase < 10 else "⚠️" if increase < 50 else "❌"
        print(f"  {status} {component}: {increase:+.2f} MB")

    print(f"\n📊 总内存增加: {total_increase:.2f} MB")

    # 最终内存状态
    final_increase = final_memory['rss'] - initial_memory['rss']
    print(f"\n📊 最终内存状态:")
    print(f"  - 初始RSS: {initial_memory['rss']:.2f} MB")
    print(f"  - 最终RSS: {final_memory['rss']:.2f} MB")
    print(f"  - 净增加: {final_increase:+.2f} MB")

    # 内存泄漏评估
    if final_increase < 5:
        print("\n✅ 内存使用正常 - 无明显内存泄漏")
    elif final_increase < 20:
        print("\n⚠️ 内存使用偏高 - 建议优化")
    else:
        print("\n❌ 内存使用异常 - 可能存在内存泄漏")

    # 显示内存分配最多的代码行
    print("\n🔍 内存分配最多的代码行:")
    for stat in top_stats[:5]:
        print(f"  {stat.count_diff:+d} 块: {stat.traceback.format()}")

    # 停止内存跟踪
    tracemalloc.stop()

    return {
        'components': components,
        'total_increase': total_increase,
        'final_increase': final_increase,
        'initial_memory': initial_memory,
        'final_memory': final_memory
    }


def analyze_multiple_iterations():
    """分析多次迭代的内存使用情况"""
    print("\n🔄 开始多次迭代内存分析...")

    results = []
    for i in range(3):
        print(f"\n📊 第 {i+1} 次迭代:")
        result = analyze_component_memory()
        results.append(result)

        # 等待一下再进行下一次迭代
        time.sleep(2)

    # 分析迭代结果
    print("\n📈 多次迭代分析结果:")
    print("=" * 60)

    final_increases = [r['final_increase'] for r in results]
    avg_increase = sum(final_increases) / len(final_increases)
    max_increase = max(final_increases)
    min_increase = min(final_increases)

    print(f"  平均内存增加: {avg_increase:.2f} MB")
    print(f"  最大内存增加: {max_increase:.2f} MB")
    print(f"  最小内存增加: {min_increase:.2f} MB")
    print(f"  内存增加范围: {max_increase - min_increase:.2f} MB")

    # 评估稳定性
    if max_increase - min_increase < 5:
        print("✅ 内存使用稳定")
    elif max_increase - min_increase < 20:
        print("⚠️ 内存使用略有波动")
    else:
        print("❌ 内存使用不稳定，可能存在内存泄漏")


def main():
    """主函数"""
    print("🔍 EnhancedDataIntegrationManager 详细内存分析")
    print("=" * 60)

    try:
        # 单次详细分析
        analyze_component_memory()

        # 多次迭代分析
        analyze_multiple_iterations()

        print("\n✅ 详细内存分析完成")

    except Exception as e:
        print(f"\n❌ 内存分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
