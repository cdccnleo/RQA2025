#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI/CD内存泄漏检测器

集成到自动化测试流程中的内存泄漏检测：
1. 测试前内存基准
2. 测试后内存对比
3. 内存泄漏阈值检测
4. CI/CD集成报告
"""

import os
import sys
import gc
import psutil
import time
import json
import subprocess
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryTestResult:
    """内存测试结果"""
    test_name: str
    baseline_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    growth_rate_percent: float
    gc_collected: int
    test_duration_seconds: float
    passed: bool
    leak_detected: bool
    details: Dict[str, Any]


class CIMemoryLeakDetector:
    """CI内存泄漏检测器"""

    def __init__(self):
        self.process = psutil.Process()
        self.results: List[MemoryTestResult] = []
        self.baseline_memory = None
        self.test_start_time = None

        # CI配置
        self.memory_threshold_mb = 50  # 50MB增长阈值
        self.growth_threshold_percent = 25  # 25%增长阈值
        self.max_test_duration = 300  # 5分钟最大测试时间

    def start_test(self, test_name: str) -> None:
        """开始测试"""
        print(f"🧪 开始测试: {test_name}")

        # 清理内存
        self._cleanup_before_test()

        # 记录基准内存
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        self.test_start_time = time.time()

        print(f"📊 基准内存: {self.baseline_memory:.2f} MB")

    def end_test(self, test_name: str) -> MemoryTestResult:
        """结束测试"""
        if self.test_start_time is None:
            raise RuntimeError("Test not started")

        # 强制垃圾回收
        gc.collect()
        time.sleep(1)  # 等待内存释放

        # 记录最终内存
        final_memory = self.process.memory_info().rss / 1024 / 1024
        test_duration = time.time() - self.test_start_time

        # 计算内存增长
        memory_growth = final_memory - self.baseline_memory
        growth_rate = (memory_growth / self.baseline_memory) * \
            100 if self.baseline_memory > 0 else 0

        # 检测内存泄漏
        leak_detected = (memory_growth > self.memory_threshold_mb or
                         growth_rate > self.growth_threshold_percent)

        # 判断测试是否通过
        passed = not leak_detected and test_duration <= self.max_test_duration

        result = MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=self.baseline_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            growth_rate_percent=growth_rate,
            gc_collected=gc.collect(),
            test_duration_seconds=test_duration,
            passed=passed,
            leak_detected=leak_detected,
            details={
                'memory_threshold_mb': self.memory_threshold_mb,
                'growth_threshold_percent': self.growth_threshold_percent,
                'max_test_duration': self.max_test_duration
            }
        )

        self.results.append(result)

        # 输出结果
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        print(f"  内存增长: {memory_growth:+.2f} MB ({growth_rate:+.1f}%)")
        print(f"  测试时长: {test_duration:.2f} 秒")

        if leak_detected:
            print(f"  ⚠️  检测到内存泄漏!")

        return result

    def _cleanup_before_test(self) -> None:
        """测试前清理"""
        # 强制垃圾回收
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)

    def run_pytest_with_memory_monitoring(self, test_path: str, pytest_args: List[str] = None) -> Dict[str, Any]:
        """运行pytest并监控内存"""
        if pytest_args is None:
            pytest_args = []

        print(f"🚀 运行pytest: {test_path}")

        # 测试前内存基准
        self.start_test("pytest_memory_monitoring")

        try:
            # 构建pytest命令
            cmd = ["python", "-m", "pytest", test_path] + pytest_args

            # 运行pytest
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=self.max_test_duration)
            end_time = time.time()

            # 测试后内存对比
            test_result = self.end_test("pytest_memory_monitoring")

            return {
                'pytest_result': {
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'duration': end_time - start_time
                },
                'memory_result': test_result,
                'success': result.returncode == 0 and test_result.passed
            }

        except subprocess.TimeoutExpired:
            test_result = self.end_test("pytest_memory_monitoring")
            return {
                'pytest_result': {
                    'return_code': -1,
                    'stdout': '',
                    'stderr': 'Test timeout',
                    'duration': self.max_test_duration
                },
                'memory_result': test_result,
                'success': False
            }

    def run_infrastructure_tests(self) -> Dict[str, Any]:
        """运行基础设施测试"""
        test_paths = [
            "tests/unit/infrastructure",
            "tests/unit/config",
            "tests/unit/monitoring"
        ]

        all_results = {}

        for test_path in test_paths:
            if os.path.exists(test_path):
                print(f"\n📁 测试路径: {test_path}")
                result = self.run_pytest_with_memory_monitoring(test_path)
                all_results[test_path] = result
            else:
                print(f"⚠️  测试路径不存在: {test_path}")

        return all_results

    def generate_ci_report(self, output_file: str = None) -> Dict[str, Any]:
        """生成CI报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/ci_memory_report_{timestamp}.json"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 统计结果
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        leak_tests = sum(1 for r in self.results if r.leak_detected)

        # 计算平均内存增长
        if self.results:
            avg_growth = sum(r.memory_growth_mb for r in self.results) / len(self.results)
            max_growth = max(r.memory_growth_mb for r in self.results)
        else:
            avg_growth = 0
            max_growth = 0

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'leak_tests': leak_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'avg_memory_growth_mb': avg_growth,
                'max_memory_growth_mb': max_growth
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'baseline_memory_mb': r.baseline_memory_mb,
                    'final_memory_mb': r.final_memory_mb,
                    'memory_growth_mb': r.memory_growth_mb,
                    'growth_rate_percent': r.growth_rate_percent,
                    'test_duration_seconds': r.test_duration_seconds,
                    'passed': r.passed,
                    'leak_detected': r.leak_detected,
                    'details': r.details
                }
                for r in self.results
            ],
            'ci_config': {
                'memory_threshold_mb': self.memory_threshold_mb,
                'growth_threshold_percent': self.growth_threshold_percent,
                'max_test_duration': self.max_test_duration
            }
        }

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 CI报告已保存: {output_file}")

        # 输出摘要
        print(f"\n📋 CI测试摘要:")
        print(f"  总测试数: {total_tests}")
        print(f"  通过测试: {passed_tests}")
        print(f"  失败测试: {failed_tests}")
        print(f"  内存泄漏: {leak_tests}")
        print(f"  成功率: {report['summary']['success_rate']:.1f}%")
        print(f"  平均内存增长: {avg_growth:.2f} MB")
        print(f"  最大内存增长: {max_growth:.2f} MB")

        return report

    def check_ci_thresholds(self) -> bool:
        """检查CI阈值"""
        if not self.results:
            return True

        # 检查是否有内存泄漏
        leak_count = sum(1 for r in self.results if r.leak_detected)
        total_count = len(self.results)

        # 如果超过20%的测试检测到内存泄漏，则失败
        leak_rate = leak_count / total_count if total_count > 0 else 0

        if leak_rate > 0.2:
            print(f"❌ CI检查失败: {leak_rate:.1%} 的测试检测到内存泄漏")
            return False

        # 检查平均内存增长
        avg_growth = sum(r.memory_growth_mb for r in self.results) / len(self.results)

        if avg_growth > self.memory_threshold_mb:
            print(f"❌ CI检查失败: 平均内存增长 {avg_growth:.2f} MB 超过阈值")
            return False

        print("✅ CI检查通过")
        return True


def main():
    """主函数"""
    detector = CIMemoryLeakDetector()

    # 运行基础设施测试
    print("🚀 开始CI内存泄漏检测")
    print("=" * 50)

    results = detector.run_infrastructure_tests()

    # 生成报告
    report = detector.generate_ci_report()

    # 检查CI阈值
    ci_passed = detector.check_ci_thresholds()

    # 设置退出码
    sys.exit(0 if ci_passed else 1)


if __name__ == "__main__":
    main()
