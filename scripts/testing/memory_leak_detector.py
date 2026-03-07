#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存泄漏检测器

专门用于检测和分析基础设施层测试的内存泄漏问题。
"""

import os
import sys
import time
import psutil
import gc
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import tracemalloc

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    memory_mb: float
    objects_count: int
    peak_memory_mb: float
    gc_stats: Dict[str, Any]


class MemoryLeakDetector:
    """内存泄漏检测器"""

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.tracemalloc_started = False

    def start_tracemalloc(self):
        """启动内存追踪"""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True

    def stop_tracemalloc(self):
        """停止内存追踪"""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False

    def take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        current_memory = self.process.memory_info().rss / 1024 / 1024

        # 获取GC统计信息
        gc.collect()
        gc_stats = {
            'collections': gc.get_stats(),
            'counts': gc.get_count(),
            'objects': len(gc.get_objects())
        }

        # 获取tracemalloc统计信息
        if self.tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / 1024 / 1024
        else:
            peak_memory_mb = current_memory

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            memory_mb=current_memory,
            objects_count=gc_stats['objects'],
            peak_memory_mb=peak_memory_mb,
            gc_stats=gc_stats
        )

        self.snapshots.append(snapshot)
        return snapshot

    def analyze_memory_growth(self) -> Dict[str, Any]:
        """分析内存增长"""
        if len(self.snapshots) < 2:
            return {"error": "需要至少两个快照"}

        first = self.snapshots[0]
        last = self.snapshots[-1]

        memory_growth = last.memory_mb - first.memory_mb
        objects_growth = last.objects_count - first.objects_count
        time_span = last.timestamp - first.timestamp

        return {
            "memory_growth_mb": memory_growth,
            "objects_growth": objects_growth,
            "time_span_seconds": time_span,
            "growth_rate_mb_per_second": memory_growth / time_span if time_span > 0 else 0,
            "peak_memory_mb": max(s.peak_memory_mb for s in self.snapshots),
            "snapshots_count": len(self.snapshots)
        }

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        leaks = []

        # 分析内存增长
        growth_analysis = self.analyze_memory_growth()
        if "error" not in growth_analysis:
            # 检查内存增长是否异常
            if growth_analysis["memory_growth_mb"] > 100:  # 增长超过100MB
                leaks.append({
                    "type": "MEMORY_GROWTH",
                    "severity": "HIGH",
                    "description": f"内存增长异常: {growth_analysis['memory_growth_mb']:.1f}MB",
                    "details": growth_analysis
                })

            # 检查对象增长是否异常
            if growth_analysis["objects_growth"] > 10000:  # 对象增长超过10000个
                leaks.append({
                    "type": "OBJECT_GROWTH",
                    "severity": "MEDIUM",
                    "description": f"对象增长异常: {growth_analysis['objects_growth']}个",
                    "details": growth_analysis
                })

        # 检查峰值内存
        max_peak = max(s.peak_memory_mb for s in self.snapshots)
        if max_peak > self.max_memory_mb:
            leaks.append({
                "type": "PEAK_MEMORY",
                "severity": "HIGH",
                "description": f"峰值内存过高: {max_peak:.1f}MB > {self.max_memory_mb}MB",
                "details": {"peak_memory_mb": max_peak}
            })

        return leaks

    def get_top_memory_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取内存分配最多的对象"""
        if not self.tracemalloc_started:
            return []

        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            allocations = []
            for stat in top_stats[:limit]:
                allocations.append({
                    "file": stat.traceback.format()[-1],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })

            return allocations
        except Exception as e:
            return [{"error": str(e)}]


def run_test_with_memory_monitoring(
    test_path: str,
    pytest_args: List[str] = None,
    snapshot_interval: float = 1.0,
    max_duration: int = 300
) -> Dict[str, Any]:
    """运行测试并监控内存使用"""

    if pytest_args is None:
        pytest_args = ["-v", "--tb=short"]

    detector = MemoryLeakDetector()
    detector.start_tracemalloc()

    # 构建pytest命令
    cmd = [sys.executable, "-m", "pytest", test_path] + pytest_args

    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )

        start_time = time.time()
        monitoring_thread = None

        def monitor_memory():
            """内存监控线程"""
            while process.poll() is None:
                detector.take_snapshot()
                time.sleep(snapshot_interval)

                # 检查超时
                if time.time() - start_time > max_duration:
                    process.terminate()
                    break

        # 启动监控线程
        monitoring_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitoring_thread.start()

        # 等待进程完成
        stdout, stderr = process.communicate()
        exit_code = process.returncode

        # 等待监控线程完成
        if monitoring_thread:
            monitoring_thread.join(timeout=5.0)

        # 分析结果
        leaks = detector.detect_leaks()
        top_allocations = detector.get_top_memory_allocations()

        result = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "duration": time.time() - start_time,
            "memory_leaks": leaks,
            "top_allocations": top_allocations,
            "snapshots": len(detector.snapshots),
            "final_memory_mb": detector.process.memory_info().rss / 1024 / 1024
        }

        return result

    finally:
        detector.stop_tracemalloc()


def analyze_infrastructure_tests():
    """分析基础设施层测试的内存使用"""

    # 测试文件列表
    test_files = [
        "tests/unit/infrastructure/test_unified_hot_reload.py",
        "tests/unit/infrastructure/test_deployment_validator.py",
        "tests/unit/infrastructure/test_coverage_improvement.py",
        "tests/unit/infrastructure/test_async_inference_engine_top20.py",
        "tests/unit/infrastructure/test_lock.py",
        "tests/unit/infrastructure/test_service_launcher.py"
    ]

    results = {}

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🔍 分析测试文件: {test_file}")

            try:
                result = run_test_with_memory_monitoring(
                    test_file,
                    pytest_args=["-v", "--tb=short", "--maxfail=1"],
                    snapshot_interval=0.5,
                    max_duration=60
                )

                results[test_file] = result

                print(f"  退出码: {result['exit_code']}")
                print(f"  最终内存: {result['final_memory_mb']:.1f}MB")
                print(f"  快照数量: {result['snapshots']}")

                if result['memory_leaks']:
                    print(f"  ⚠️  发现 {len(result['memory_leaks'])} 个内存泄漏:")
                    for leak in result['memory_leaks']:
                        print(f"    - {leak['severity']}: {leak['description']}")
                else:
                    print("  ✅ 未发现内存泄漏")

                if result['top_allocations']:
                    print("  📊 内存分配最多的对象:")
                    for alloc in result['top_allocations'][:3]:
                        if "error" not in alloc:
                            print(
                                f"    - {alloc['file']}: {alloc['size_mb']:.1f}MB ({alloc['count']}个)")

            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                results[test_file] = {"error": str(e)}

    return results


def generate_memory_report(results: Dict[str, Any]):
    """生成内存报告"""

    print("\n" + "="*60)
    print("📊 基础设施层测试内存分析报告")
    print("="*60)

    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if "error" not in r)
    failed_tests = total_tests - successful_tests

    print(f"\n📈 测试统计:")
    print(f"  总测试文件: {total_tests}")
    print(f"  成功分析: {successful_tests}")
    print(f"  分析失败: {failed_tests}")

    if successful_tests > 0:
        # 内存统计
        memory_usage = [r['final_memory_mb'] for r in results.values() if "error" not in r]
        avg_memory = sum(memory_usage) / len(memory_usage)
        max_memory = max(memory_usage)
        min_memory = min(memory_usage)

        print(f"\n💾 内存使用统计:")
        print(f"  平均内存: {avg_memory:.1f}MB")
        print(f"  最大内存: {max_memory:.1f}MB")
        print(f"  最小内存: {min_memory:.1f}MB")

        # 泄漏统计
        total_leaks = sum(len(r.get('memory_leaks', []))
                          for r in results.values() if "error" not in r)
        print(f"\n⚠️  内存泄漏统计:")
        print(f"  总泄漏数: {total_leaks}")

        if total_leaks > 0:
            print("\n🔍 详细泄漏信息:")
            for test_file, result in results.items():
                if "error" not in result and result.get('memory_leaks'):
                    print(f"\n  {test_file}:")
                    for leak in result['memory_leaks']:
                        print(f"    - {leak['severity']}: {leak['description']}")

    print("\n" + "="*60)


def main():
    """主函数"""
    print("🔍 开始基础设施层测试内存泄漏分析...")

    # 分析测试
    results = analyze_infrastructure_tests()

    # 生成报告
    generate_memory_report(results)

    print("\n✅ 内存泄漏分析完成！")


if __name__ == "__main__":
    main()
