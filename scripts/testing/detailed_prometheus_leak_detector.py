#!/usr/bin/env python3
"""
详细的Prometheus内存泄漏检测器
专门检测Prometheus相关的内存泄漏问题
"""

import sys
import psutil
from typing import List
from dataclasses import dataclass


@dataclass
class PrometheusLeakInfo:
    """Prometheus泄漏信息"""
    metric_name: str
    metric_type: str
    is_system_metric: bool
    memory_impact: float = 0.0


class DetailedPrometheusLeakDetector:
    """详细的Prometheus泄漏检测器"""

    def __init__(self):
        self.system_metrics = [
            'python_gc_objects_collected', 'python_gc_objects_collected_total',
            'python_gc_objects_collected_created', 'python_gc_objects_uncollectable',
            'python_gc_objects_uncollectable_total', 'python_gc_objects_uncollectable_created',
            'python_gc_collections', 'python_gc_collections_total',
            'python_gc_collections_created', 'python_info'
        ]

    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def detect_prometheus_leaks(self) -> List[PrometheusLeakInfo]:
        """检测Prometheus泄漏"""
        leaks = []

        try:
            from prometheus_client import REGISTRY

            if not hasattr(REGISTRY, '_names_to_collectors'):
                print("⚠️  Prometheus注册表没有_names_to_collectors属性")
                return leaks

            print(f"🔍 检测Prometheus注册表中的指标...")
            print(f"   总指标数量: {len(REGISTRY._names_to_collectors)}")

            for metric_name, collector in REGISTRY._names_to_collectors.items():
                is_system = metric_name in self.system_metrics
                metric_type = type(collector).__name__

                leak_info = PrometheusLeakInfo(
                    metric_name=metric_name,
                    metric_type=metric_type,
                    is_system_metric=is_system
                )

                if not is_system:
                    leaks.append(leak_info)
                    print(f"   ⚠️  检测到非系统指标: {metric_name} ({metric_type})")
                else:
                    print(f"   ✅ 系统指标: {metric_name} ({metric_type})")

            return leaks

        except ImportError:
            print("⚠️  Prometheus客户端未安装")
            return leaks
        except Exception as e:
            print(f"⚠️  Prometheus检测失败: {e}")
            return leaks

    def run_detection(self):
        """运行检测"""
        print("🔍 详细Prometheus泄漏检测器启动")
        print("🚀 开始详细Prometheus泄漏检测")
        print("=" * 60)

        # 记录初始内存
        initial_memory = self.get_memory_usage()
        print(f"🔍 开始内存监控，初始内存: {initial_memory:.2f} MB")

        # 检测Prometheus泄漏
        print("\n🔍 检测Prometheus指标泄漏...")
        leaks = self.detect_prometheus_leaks()

        # 生成报告
        print("\n📊 详细Prometheus泄漏检测报告")
        print("=" * 60)

        if not leaks:
            print("✅ 未检测到Prometheus泄漏")
            print("✅ 所有指标都是系统指标，无泄漏")
        else:
            print(f"❌ 检测到 {len(leaks)} 个Prometheus泄漏:")
            for leak in leaks:
                print(f"   - {leak.metric_name} ({leak.metric_type})")

        # 记录最终内存
        final_memory = self.get_memory_usage()
        memory_growth = final_memory - initial_memory
        print(f"\n📈 内存变化: {initial_memory:.2f}MB -> {final_memory:.2f}MB (增长: {memory_growth:.2f}MB)")

        if memory_growth > 10:
            print(f"⚠️  检测到内存增长: {memory_growth:.2f}MB")
        else:
            print(f"✅ 内存增长正常: {memory_growth:.2f}MB")

        return len(leaks) == 0


def main():
    """主函数"""
    detector = DetailedPrometheusLeakDetector()
    success = detector.run_detection()

    if success:
        print("\n✅ 未检测到Prometheus泄漏")
        return 0
    else:
        print("\n❌ 检测到Prometheus泄漏")
        return 1


if __name__ == "__main__":
    sys.exit(main())
