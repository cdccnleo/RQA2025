#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存分析和基线建立脚本
用于检测潜在的内存泄漏问题
"""

import sys
import gc
import weakref
import threading
from collections import defaultdict
import time
import psutil
import os


class MemoryAnalyzer:
    """内存分析器"""

    def __init__(self):
        self.baseline_objects = {}
        self.object_refs = weakref.WeakValueDictionary()
        self.leak_candidates = []
        self.memory_stats = {}

    def take_baseline(self):
        """建立内存基线"""
        print("📊 建立内存基线...")

        # 强制垃圾回收
        gc.collect()

        # 获取当前对象统计
        self.baseline_objects = {
            'total_objects': len(gc.get_objects()),
            'gc_stats': gc.get_stats(),
            'referrers': self._analyze_referrers(),
            'large_objects': self._find_large_objects()
        }

        # 获取系统内存信息
        process = psutil.Process(os.getpid())
        self.memory_stats['baseline'] = {
            'rss': process.memory_info().rss,
            'vms': process.memory_info().vms,
            'cpu_percent': process.cpu_percent(interval=0.1)
        }

        print(f"   ✅ 基线建立完成: {self.baseline_objects['total_objects']} 个对象")
        print(".1f"
        print(".1f"
    def detect_leaks(self, iterations=5):
        """检测内存泄漏"""
        print(f"\n🔍 开始内存泄漏检测 ({iterations} 次迭代)...")

        for i in range(iterations):
            print(f"\n📈 第 {i+1} 次迭代:")

            # 模拟一些操作
            self._simulate_operations()

            # 强制垃圾回收
            collected=gc.collect()

            # 分析对象增长
            current_objects=len(gc.get_objects())
            growth=current_objects - self.baseline_objects['total_objects']

            print(f"   📊 当前对象数: {current_objects}")
            print(f"   📈 对象增长: {growth}")
            print(f"   🗑️ 垃圾回收: {collected} 个对象")

            # 检查是否有显著增长
            if growth > 100:  # 阈值
                print("   ⚠️ 检测到对象数量显著增长")
                self._analyze_growth()

            time.sleep(0.1)

    def _simulate_operations(self):
        """模拟系统操作以检测内存泄漏"""
        # 创建一些临时对象
        temp_list=[i for i in range(1000)]
        temp_dict={f"key_{i}": f"value_{i}" for i in range(500)}

        # 模拟数据库查询结果
        mock_results=[{"id": i, "data": "x" * 100} for i in range(100)]

        # 清理局部变量（但可能会有循环引用）
        del temp_list
        del temp_dict
        del mock_results

    def _analyze_referrers(self):
        """分析对象引用关系"""
        print("   🔗 分析对象引用关系...")

        # 获取所有对象
        all_objects=gc.get_objects()

        # 按类型统计
        type_counts=defaultdict(int)
        for obj in all_objects:
            type_counts[type(obj).__name__] += 1

        # 返回前10个最常见的类型
        sorted_types=sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_types[:10])

    def _find_large_objects(self):
        """查找大对象"""
        print("   📏 查找大对象...")

        all_objects=gc.get_objects()
        large_objects=[]

        for obj in all_objects:
            try:
                size=sys.getsizeof(obj)
                if size > 10000:  # 10KB以上
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size': size,
                        'id': id(obj)
                    })
            except:
                pass

        # 按大小排序
        large_objects.sort(key=lambda x: x['size'], reverse=True)
        return large_objects[:5]  # 返回最大的5个

    def _analyze_growth(self):
        """分析对象增长原因"""
        print("   🔍 分析对象增长原因...")

        # 获取垃圾回收器统计
        gc_stats=gc.get_stats()
        print(f"   📊 GC统计: {gc_stats}")

        # 检查未回收的对象
        unreachable=gc.garbage
        if unreachable:
            print(f"   ⚠️ 发现 {len(unreachable)} 个不可达对象")
            for i, obj in enumerate(unreachable[:3]):
                print(f"      {i+1}. {type(obj).__name__}")

    def check_circular_refs(self):
        """检查循环引用"""
        print("\n🔄 检查循环引用...")

        # 启用垃圾回收调试
        gc.set_debug(gc.DEBUG_SAVEALL)

        # 强制垃圾回收
        collected=gc.collect()

        # 检查垃圾对象
        if gc.garbage:
            print(f"   ⚠️ 发现 {len(gc.garbage)} 个循环引用对象")
            for i, obj in enumerate(gc.garbage[:5]):
                print(f"      {i+1}. {type(obj).__name__}: {obj}")
        else:
            print("   ✅ 未发现明显的循环引用")

        # 清理垃圾
        gc.garbage.clear()
        gc.set_debug(0)

    def generate_report(self):
        """生成内存分析报告"""
        print("\n📋 内存分析报告")
        print("=" * 50)

        print("\n📊 对象统计:")
        print(f"   总对象数: {self.baseline_objects.get('total_objects', 'N/A')}")

        print("\n🔗 主要对象类型:")
        referrers=self.baseline_objects.get('referrers', {})
        for obj_type, count in referrers.items():
            print(f"   {obj_type}: {count}")

        print("\n📏 大对象:")
        large_objects=self.baseline_objects.get('large_objects', [])
        for obj in large_objects:
            print(".1f"

        print("\n💡 建议:")
        if large_objects:
            print("   • 考虑优化大对象的内存使用")
        if len(referrers) > 5:
            print("   • 对象类型较多，建议检查是否有多余的对象")
        print("   • 建议定期运行内存分析")

def main():
    """主函数"""
    print("🧠 RQA2025 内存分析工具")
    print("=" * 40)

    analyzer=MemoryAnalyzer()

    # 建立基线
    analyzer.take_baseline()

    # 检测泄漏
    analyzer.detect_leaks(iterations=3)

    # 检查循环引用
    analyzer.check_circular_refs()

    # 生成报告
    analyzer.generate_report()

    print("\n🎉 内存分析完成!")

if __name__ == "__main__":
    main()
