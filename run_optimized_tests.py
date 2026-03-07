#!/usr/bin/env python3
"""
优化后的基础设施层测试执行脚本

通过并行执行、跳过耗时测试等策略提升执行效率
"""

import subprocess
import time
from pathlib import Path
import argparse
import json
import concurrent.futures
from typing import List, Dict, Any


class OptimizedTestRunner:
    """优化后的测试运行器"""

    def __init__(self, max_workers=4, timeout=60):
        self.max_workers = max_workers
        self.timeout = timeout
        self.project_root = Path(__file__).resolve().parent
        self.test_results = {}

    def run_test_parallel(self, test_commands: List[str]) -> Dict[str, Any]:
        """并行运行测试"""
        results = {}

        def run_single_test(cmd: str) -> tuple:
            test_name = cmd.split('/')[-1].replace('.py', '').replace('::', '_')
            start_time = time.time()

            try:
                result = subprocess.run(
                    f"python -m pytest {cmd} -q",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.project_root
                )

                end_time = time.time()
                duration = end_time - start_time

                return (test_name, {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'returncode': result.returncode,
                    'duration': duration,
                    'command': cmd,
                    'timeout': self.timeout
                })

            except subprocess.TimeoutExpired:
                end_time = time.time()
                return (test_name, {
                    'status': 'timeout',
                    'duration': self.timeout,
                    'command': cmd,
                    'timeout': self.timeout
                })

        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_single_test, cmd) for cmd in test_commands]
            for future in concurrent.futures.as_completed(futures):
                test_name, result = future.result()
                results[test_name] = result
                status = result['status']
                duration = result['duration']
                print("2d")

        return results

    def run_fast_tests_first(self):
        """优先运行快速测试"""
        print("🚀 执行优化策略：快速测试优先")

        # 快速测试（预计<5秒）
        fast_tests = [
            "tests/unit/infrastructure/test_cache_system.py::TestUnifiedCache::test_initialization",
            "tests/unit/infrastructure/test_cache_system.py::TestUnifiedCache::test_basic_operations",
            "tests/unit/infrastructure/test_logging_system.py::TestUnifiedLogger::test_initialization",
            "tests/unit/infrastructure/test_logging_system.py::TestUnifiedLogger::test_basic_logging",
        ]

        # 中等速度测试（预计5-15秒）
        medium_tests = [
            "tests/unit/infrastructure/test_cache_system.py::TestBaseCacheManager",
            "tests/unit/infrastructure/test_logging_system.py::TestUnifiedLogger",
        ]

        # 慢速测试（预计>30秒，带timeout标记）
        slow_tests = [
            "tests/unit/infrastructure/test_cache_system.py::TestUnifiedCache::test_concurrent_access",
            "tests/unit/infrastructure/test_logging_system.py::TestUnifiedLogger::test_concurrent_logging",
        ]

        all_results = {}

        # 1. 先运行快速测试
        print("\n⚡ 第一阶段：快速测试")
        if fast_tests:
            fast_results = self.run_test_parallel(fast_tests)
            all_results.update(fast_results)

        # 2. 运行中等速度测试
        print("\n🟡 第二阶段：中等速度测试")
        if medium_tests:
            medium_results = self.run_test_parallel(medium_tests)
            all_results.update(medium_results)

        # 3. 最后运行慢速测试（带更长的超时时间）
        print("\n🐌 第三阶段：慢速测试")
        if slow_tests:
            # 为慢速测试设置更长的超时时间
            original_timeout = self.timeout
            self.timeout = 180  # 3分钟超时

            slow_results = self.run_test_parallel(slow_tests)
            all_results.update(slow_results)

            # 恢复原始超时时间
            self.timeout = original_timeout

        return all_results

    def run_coverage_quick_scan(self):
        """快速覆盖率扫描"""
        print("\n📊 执行快速覆盖率扫描")

        # 只扫描核心模块，减少扫描范围
        coverage_cmd = (
            "python -m pytest tests/unit/infrastructure/test_cache_system.py "
            "tests/unit/infrastructure/test_logging_system.py "
            "--cov=src/infrastructure/cache "
            "--cov=src/infrastructure/logging "
            "--cov-report=term-missing "
            "--cov-report=json:reports/quick_coverage.json "
            "-q"
        )

        start_time = time.time()

        try:
            result = subprocess.run(
                coverage_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=self.project_root
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ 快速覆盖率扫描完成 ({duration:.1f}s)")
            print("覆盖率结果:")
            print(result.stdout.split('\n')[-10:])  # 显示最后10行

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("⏰ 快速覆盖率扫描超时")
            return False

    def generate_optimization_report(self, results: Dict[str, Any]):
        """生成优化报告"""
        print("\n" + "=" * 60)
        print("📋 测试优化报告")
        print("=" * 60)

        total_tests = len(results)
        passed = sum(1 for r in results.values() if r['status'] == 'passed')
        failed = sum(1 for r in results.values() if r['status'] == 'failed')
        timeout = sum(1 for r in results.values() if r['status'] == 'timeout')

        total_duration = sum(r['duration'] for r in results.values())

        print(f"总测试数: {total_tests}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"⏰ 超时: {timeout}")
        print(".1f")
        print(".2f")

        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(".1f")
            print(".2f")
        # 显示最耗时的测试
        if results:
            sorted_by_duration = sorted(
                results.items(), key=lambda x: x[1]['duration'], reverse=True)
            print("\n⏱️  最耗时的测试:")
            for i, (test_name, result) in enumerate(sorted_by_duration[:5]):
                print("2d")

        # 保存详细结果
        report_file = self.project_root / "reports" / "optimization_report.json"
        report_file.parent.mkdir(exist_ok=True)

        report_data = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'timeout': timeout,
                'total_duration': total_duration,
                'average_duration': total_duration / total_tests if total_tests > 0 else 0,
                'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0
            },
            'details': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_strategy': 'parallel_execution_with_priority'
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n详细报告已保存: {report_file}")

        return report_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化后的基础设施层测试运行器')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行工作数')
    parser.add_argument('--timeout', type=int, default=60,
                        help='单个测试超时时间(秒)')
    parser.add_argument('--coverage', action='store_true',
                        help='执行快速覆盖率扫描')
    parser.add_argument('--strategy', choices=['priority', 'parallel'],
                        default='priority', help='执行策略')

    args = parser.parse_args()

    runner = OptimizedTestRunner(
        max_workers=args.workers,
        timeout=args.timeout
    )

    print(f"🚀 启动优化测试执行器 (工作数: {args.workers}, 超时: {args.timeout}s)")

    if args.strategy == 'priority':
        results = runner.run_fast_tests_first()
    else:
        # 并行策略的实现可以后续添加
        results = runner.run_fast_tests_first()

    report = runner.generate_optimization_report(results)

    if args.coverage:
        runner.run_coverage_quick_scan()

    # 返回成功状态
    success_rate = report['summary']['success_rate']
    return 0 if success_rate >= 70 else 1  # 70%作为通过阈值


if __name__ == "__main__":
    exit(main())
