#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行测试运行器

实现基础设施层测试的并行执行，提高测试效率。
"""

import os
import sys
import psutil
import time
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ParallelTestRunner:
    """并行测试运行器"""

    def __init__(self, max_workers: int = None, max_memory_mb: int = 2048):
        """
        初始化并行测试运行器

        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
            max_memory_mb: 最大内存使用量(MB)
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 4)
        self.max_memory_mb = max_memory_mb
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

    def run_single_test_file(self, test_file: str) -> Dict[str, Any]:
        """运行单个测试文件"""
        print(f"🔍 运行测试文件: {test_file}")

        start_time = time.time()

        try:
            # 运行测试
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--maxfail=1",
                "--disable-warnings"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            end_time = time.time()

            # 记录结果
            test_result = {
                'file': test_file,
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

            print(
                f"⏱️  {test_file}: {test_result['duration']:.2f}s - {'✅' if test_result['success'] else '❌'}")

            return test_result

        except subprocess.TimeoutExpired:
            print(f"⏰  {test_file}: 超时")
            return {
                'file': test_file,
                'success': False,
                'duration': 300,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"💥  {test_file}: 异常 - {e}")
            return {
                'file': test_file,
                'success': False,
                'duration': 0,
                'error': str(e)
            }

    def run_parallel_tests(self, test_pattern: str = "tests/unit/infrastructure/") -> List[Dict[str, Any]]:
        """并行运行测试"""
        print(f"🚀 开始并行运行基础设施层测试")
        print(f"👥 工作进程数: {self.max_workers}")
        print(f"📊 内存限制: {self.max_memory_mb}MB")

        # 获取测试文件列表
        test_dir = Path(test_pattern)
        if test_dir.is_file():
            test_files = [test_pattern]
        else:
            test_files = list(Path("tests/unit/infrastructure").glob("test_*.py"))

        print(f"📁 发现 {len(test_files)} 个测试文件")

        # 按文件大小排序，先运行小文件
        test_files.sort(key=lambda f: f.stat().st_size if isinstance(f, Path) and f.exists() else 0)

        # 检查内存限制
        if not self.check_memory_limit():
            print(f"⚠️  内存使用过高，停止测试")
            return []

        # 并行执行测试
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有测试任务
            future_to_test = {
                executor.submit(self.run_single_test_file, str(test_file)): test_file
                for test_file in test_files
            }

            # 收集结果
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)

                    # 检查内存限制
                    if not self.check_memory_limit():
                        print(f"⚠️  内存使用过高，停止剩余测试")
                        break

                except Exception as e:
                    print(f"💥  {test_file}: 执行异常 - {e}")
                    results.append({
                        'file': str(test_file),
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    })

        self.test_results = results
        return results

    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.test_results:
            return "没有测试结果"

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests

        total_duration = sum(r.get('duration', 0) for r in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0

        # 计算并行效率
        sequential_time = sum(r.get('duration', 0) for r in self.test_results)
        parallel_time = max(r.get('duration', 0)
                            for r in self.test_results) if self.test_results else 0
        efficiency = (sequential_time / parallel_time * 100) if parallel_time > 0 else 0

        report = f"""
📊 并行测试报告
================

📈 测试统计:
- 总测试文件: {total_tests}
- 通过测试: {passed_tests}
- 失败测试: {failed_tests}
- 成功率: {passed_tests/total_tests*100:.1f}%

⏱️  性能统计:
- 总耗时: {total_duration:.2f}秒
- 平均耗时: {avg_duration:.2f}秒
- 并行效率: {efficiency:.1f}%
- 工作进程数: {self.max_workers}

📋 详细结果:
"""

        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            report += f"{status} {result['file']} ({result.get('duration', 0):.2f}s)\n"

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="并行测试运行器")
    parser.add_argument("--max-workers", type=int, default=None, help="最大工作进程数")
    parser.add_argument("--max-memory", type=int, default=2048, help="最大内存使用量(MB)")
    parser.add_argument("--test-pattern", type=str,
                        default="tests/unit/infrastructure/", help="测试文件模式")

    args = parser.parse_args()

    # 创建测试运行器
    runner = ParallelTestRunner(
        max_workers=args.max_workers,
        max_memory_mb=args.max_memory
    )

    # 运行测试
    results = runner.run_parallel_tests(args.test_pattern)

    # 生成报告
    report = runner.generate_report()
    print(report)

    # 保存报告
    report_file = f"reports/parallel_test_report_{int(time.time())}.txt"
    os.makedirs("reports", exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 报告已保存到: {report_file}")

    # 返回适当的退出码
    failed_tests = sum(1 for r in results if not r['success'])
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())
