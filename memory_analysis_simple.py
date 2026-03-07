#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存分析和基线建立脚本
用于检测潜在的内存泄漏问题
"""

import sys
import gc
from collections import defaultdict
import psutil
import os


def analyze_memory_baseline():
    """建立内存基线"""
    print("📊 建立内存基线...")

    # 强制垃圾回收
    gc.collect()

    # 获取当前对象统计
    total_objects = len(gc.get_objects())
    print("   Total objects: {}".format(total_objects))

    # 获取系统内存信息
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print("   Memory usage: {:.1f} MB".format(memory_info.rss / 1024 / 1024))
        print("   Virtual memory: {:.1f} MB".format(memory_info.vms / 1024 / 1024))
    except ImportError:
        print("   psutil not available, skipping memory stats")

    return total_objects


def detect_memory_growth(baseline_objects, iterations=3):
    """检测内存增长"""
    print("\n🔍 开始内存泄漏检测 ({} 次迭代)...".format(iterations))

    for i in range(iterations):
        print("\nIteration {}:".format(i+1))

        # 模拟一些操作
        temp_data = []
        for j in range(1000):
            temp_data.append({"id": j, "data": "x" * 100})

        # 清理数据
        del temp_data

        # 强制垃圾回收
        collected = gc.collect()

        # 分析对象增长
        current_objects = len(gc.get_objects())
        growth = current_objects - baseline_objects

        print("   Current objects: {}".format(current_objects))
        print("   Object growth: {}".format(growth))
        print("   GC collected: {} objects".format(collected))

        if growth > 100:
            print("   WARNING: Significant object growth detected")


def analyze_object_types():
    """分析对象类型分布"""
    print("\n🔗 分析对象类型分布...")

    all_objects = gc.get_objects()
    type_counts = defaultdict(int)

    for obj in all_objects:
        type_counts[type(obj).__name__] += 1

    # 显示前10个最常见的类型
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    print("   Top 10 object types:")
    for i, (obj_type, count) in enumerate(sorted_types[:10]):
        print("      {}. {}: {}".format(i+1, obj_type, count))


def find_large_objects():
    """查找大对象"""
    print("\n📏 查找大对象...")

    all_objects = gc.get_objects()
    large_objects = []

    for obj in all_objects:
        try:
            size = sys.getsizeof(obj)
            if size > 10000:  # 10KB以上
                large_objects.append((type(obj).__name__, size))
        except:
            pass

    if large_objects:
        print("   Large objects found:")
        for obj_type, size in sorted(large_objects, key=lambda x: x[1], reverse=True)[:5]:
            print("      {}: {:.1f} KB".format(obj_type, size / 1024))
    else:
        print("   No large objects found")


def check_circular_references():
    """检查循环引用"""
    print("\n🔄 检查循环引用...")

    # 保存当前调试标志
    old_flags = gc.get_debug()

    # 启用调试
    gc.set_debug(gc.DEBUG_SAVEALL)

    # 强制垃圾回收
    collected = gc.collect()

    # 检查垃圾对象
    if gc.garbage:
        print("   WARNING: {} circular reference objects found".format(len(gc.garbage)))
        for i, obj in enumerate(gc.garbage[:3]):
            print("      {}. {}: {}".format(i+1, type(obj).__name__, str(obj)[:50]))
    else:
        print("   No circular references detected")

    # 清理并恢复调试设置
    gc.garbage.clear()
    gc.set_debug(old_flags)


def main():
    """主函数"""
    print("🧠 RQA2025 内存分析工具")
    print("=" * 40)

    # 建立基线
    baseline = analyze_memory_baseline()

    # 检测内存增长
    detect_memory_growth(baseline, iterations=3)

    # 分析对象类型
    analyze_object_types()

    # 查找大对象
    find_large_objects()

    # 检查循环引用
    check_circular_references()

    print("\n📋 内存分析完成!")
    print("💡 建议:")
    print("   • 如果发现对象持续增长，检查是否有未清理的引用")
    print("   • 大对象过多可能影响性能，考虑优化数据结构")
    print("   • 循环引用会导致内存泄漏，需要特别注意")


if __name__ == "__main__":
    main()
