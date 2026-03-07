#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存安全的基础设施层测试运行器

解决基础设施层测试的内存泄漏问题，提供内存监控和限制功能。
"""

import os
import sys
import psutil
import gc
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MemorySafeTestRunner:
    """内存安全的测试运行器"""

    def __init__(self, max_memory_mb: int = 2048, test_timeout_seconds: int = 300):
        """
        初始化测试运行器

        Args:
            max_memory_mb: 最大内存使用量(MB)
            test_timeout_seconds: 测试超时时间(秒)
        """
        self.max_memory_mb = max_memory_mb
        self.test_timeout_seconds = test_timeout_seconds
        self.process = psutil.Process()
        self.test_results = []

    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory_limit(self) -> bool:
        """检查内存是否超过限制"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_mb:
            print(f"⚠️  内存使用超过限制: {current_memory:.2f}MB > {self.max_memory_mb}MB")
            return False
        return True

    def force_garbage_collection(self):
        """强制垃圾回收"""
        gc.collect()
        time.sleep(0.1)  # 给GC一些时间

    def run_single_test_file(self, test_file: str) -> Dict[str, Any]:
        """运行单个测试文件"""
        print(f"\n🔍 运行测试文件: {test_file}")

        # 记录开始时的内存使用
        initial_memory = self.get_memory_usage()
        print(f"📊 初始内存使用: {initial_memory:.2f}MB")

        start_time = time.time()

        try:
            # 运行测试
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--maxfail=1",  # 限制失败数量
                "--disable-warnings"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_timeout_seconds
            )

            end_time = time.time()
            final_memory = self.get_memory_usage()

            # 记录结果
            test_result = {
                'file': test_file,
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'memory_increase': final_memory - initial_memory,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

            print(f"⏱️  测试耗时: {test_result['duration']:.2f}秒")
            print(f"📊 内存变化: {test_result['memory_increase']:+.2f}MB")

            if test_result['success']:
                print(f"✅ 测试通过: {test_file}")
            else:
                print(f"❌ 测试失败: {test_file}")
                print(f"错误信息: {result.stderr}")

            # 强制垃圾回收
            self.force_garbage_collection()

            return test_result

        except subprocess.TimeoutExpired:
            print(f"⏰ 测试超时: {test_file}")
            return {
                'file': test_file,
                'success': False,
                'duration': self.test_timeout_seconds,
                'error': 'timeout',
                'initial_memory': initial_memory,
                'final_memory': self.get_memory_usage()
            }
        except Exception as e:
            print(f"💥 测试异常: {test_file} - {e}")
            return {
                'file': test_file,
                'success': False,
                'duration': 0,
                'error': str(e),
                'initial_memory': initial_memory,
                'final_memory': self.get_memory_usage()
            }

    def run_infrastructure_tests(self, test_pattern: str = "tests/unit/infrastructure/") -> List[Dict[str, Any]]:
        """运行基础设施层测试"""
        print("🚀 开始运行基础设施层测试 (内存安全模式)")
        print(f"📊 内存限制: {self.max_memory_mb}MB")
        print(f"⏱️  超时限制: {self.test_timeout_seconds}秒")

        # 获取测试文件列表
        test_dir = Path(test_pattern)
        if test_dir.is_file():
            test_files = [test_pattern]
        else:
            test_files = list(Path("tests/unit/infrastructure").glob("test_*.py"))

        print(f"📁 发现 {len(test_files)} 个测试文件")

        # 按文件大小排序，先运行小文件
        test_files.sort(key=lambda f: f.stat().st_size if f.exists() else 0)

        for test_file in test_files:
            # 检查内存限制
            if not self.check_memory_limit():
                print(f"⚠️  内存使用过高，跳过剩余测试")
                break

            # 运行测试
            result = self.run_single_test_file(str(test_file))
            self.test_results.append(result)

            # 如果内存增长过快，强制垃圾回收
            if result.get('memory_increase', 0) > 100:  # 增长超过100MB
                print("🧹 检测到内存增长过快，执行强制垃圾回收")
                self.force_garbage_collection()

        return self.test_results

    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.test_results:
            return "没有测试结果"

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests

        total_duration = sum(r.get('duration', 0) for r in self.test_results)
        total_memory_increase = sum(r.get('memory_increase', 0) for r in self.test_results)

        report = f"""
📊 基础设施层测试报告
====================

📈 测试统计:
- 总测试文件: {total_tests}
- 通过测试: {passed_tests}
- 失败测试: {failed_tests}
- 成功率: {passed_tests/total_tests*100:.1f}%

⏱️  性能统计:
- 总耗时: {total_duration:.2f}秒
- 平均耗时: {total_duration/total_tests:.2f}秒
- 总内存增长: {total_memory_increase:.2f}MB
- 平均内存增长: {total_memory_increase/total_tests:.2f}MB

📋 详细结果:
"""

        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            report += f"{status} {result['file']} ({result.get('duration', 0):.2f}s, +{result.get('memory_increase', 0):.2f}MB)\n"

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="内存安全的基础设施层测试运行器")
    parser.add_argument("--max-memory", type=int, default=2048, help="最大内存使用量(MB)")
    parser.add_argument("--timeout", type=int, default=300, help="测试超时时间(秒)")
    parser.add_argument("--test-file", type=str, help="运行特定测试文件")

    args = parser.parse_args()

    # 创建测试运行器
    runner = MemorySafeTestRunner(
        max_memory_mb=args.max_memory,
        test_timeout_seconds=args.timeout
    )

    # 运行测试
    if args.test_file:
        results = [runner.run_single_test_file(args.test_file)]
    else:
        results = runner.run_infrastructure_tests()

    # 生成报告
    report = runner.generate_report()
    print(report)

    # 保存报告
    report_file = f"reports/infrastructure_test_report_{int(time.time())}.txt"
    os.makedirs("reports", exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 报告已保存到: {report_file}")

    # 返回适当的退出码
    failed_tests = sum(1 for r in results if not r['success'])
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())
