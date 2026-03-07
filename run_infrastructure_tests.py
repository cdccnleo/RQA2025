#!/usr/bin/env python3
"""
基础设施层测试优化运行脚本

提供带超时的基础设施层测试运行，支持分批执行和性能监控
"""

import subprocess
import time
from pathlib import Path
import argparse
import json


class InfrastructureTestRunner:
    """基础设施层测试运行器"""

    def __init__(self, timeout_per_test=120, max_workers=4):
        self.timeout_per_test = timeout_per_test
        self.max_workers = max_workers
        self.project_root = Path(__file__).resolve().parent
        self.test_results = {}

    def run_test_with_timeout(self, test_command, test_name, timeout=None):
        """运行单个测试用例，带超时控制"""
        if timeout is None:
            timeout = self.timeout_per_test

        print(f"🚀 启动测试: {test_name} (超时: {timeout}s)")

        try:
            start_time = time.time()

            # 使用subprocess运行测试，设置超时
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )

            end_time = time.time()
            duration = end_time - start_time

            self.test_results[test_name] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timeout': timeout
            }

            if result.returncode == 0:
                print(f"✅ 测试通过: {test_name} ({duration:.2f}s)")
            else:
                print(f"❌ 测试失败: {test_name} ({duration:.2f}s)")
                if result.stderr:
                    print(f"错误信息: {result.stderr[-500:]}")  # 只显示最后500字符

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"⏰ 测试超时: {test_name} ({timeout}s)")
            self.test_results[test_name] = {
                'status': 'timeout',
                'duration': timeout,
                'timeout': timeout
            }
            return False
        except Exception as e:
            print(f"💥 测试异常: {test_name} - {e}")
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'timeout': timeout
            }
            return False

    def run_infrastructure_tests(self, batch_mode=False):
        """运行基础设施层测试"""
        print("=" * 60)
        print("🏗️  基础设施层测试执行器")
        print("=" * 60)

        # 基础配置测试
        config_tests = [
            "tests/unit/infrastructure/config/test_config_manager.py -v",
            "tests/unit/infrastructure/config/test_configuration.py -v",
            "tests/unit/infrastructure/config/test_config_system_enhanced.py -v"
        ]

        # 缓存测试
        cache_tests = [
            "tests/unit/infrastructure/test_cache_system.py -v",
            "tests/unit/infrastructure/cache/test_unified_cache.py -v",
            "tests/unit/infrastructure/cache/test_cache_system.py -v"
        ]

        # 日志测试
        logging_tests = [
            "tests/unit/infrastructure/test_logging_system.py -v",
            "tests/unit/infrastructure/logging/test_unified_logger.py -v",
            "tests/unit/infrastructure/logging/test_logger.py -v"
        ]

        # 健康检查测试
        health_tests = [
            "tests/unit/infrastructure/test_health_system.py -v",
            "tests/unit/infrastructure/health/test_health_check_core.py -v",
            "tests/unit/infrastructure/health/test_enhanced_health_checker.py -v"
        ]

        # 错误处理测试
        error_tests = [
            "tests/unit/infrastructure/error/test_unified_error_handler.py -v",
            "tests/unit/infrastructure/error/test_circuit_breaker.py -v"
        ]

        # 监控测试
        monitoring_tests = [
            "tests/unit/infrastructure/monitoring/test_continuous_monitoring_system.py -v",
            "tests/unit/infrastructure/monitoring/test_monitoring_system.py -v"
        ]

        # 工具库测试
        utils_tests = [
            "tests/unit/infrastructure/utils/test_utils.py -v",
            "tests/unit/infrastructure/utils/test_data_utils.py -v"
        ]

        # 服务测试
        service_tests = [
            "tests/unit/infrastructure/service/test_microservice_manager.py -v",
            "tests/unit/infrastructure/service/test_service_launcher.py -v"
        ]

        all_test_batches = {
            'config': config_tests,
            'cache': cache_tests,
            'logging': logging_tests,
            'health': health_tests,
            'error': error_tests,
            'monitoring': monitoring_tests,
            'utils': utils_tests,
            'service': service_tests
        }

        total_passed = 0
        total_failed = 0
        total_timeout = 0

        if batch_mode:
            # 分批执行模式
            for batch_name, tests in all_test_batches.items():
                print(f"\n📦 执行测试批次: {batch_name}")
                print("-" * 40)

                batch_passed = 0
                batch_failed = 0
                batch_timeout = 0

                for test_cmd in tests:
                    test_name = f"{batch_name}_{test_cmd.split('/')[-1].replace('.py', '')}"
                    success = self.run_test_with_timeout(
                        f"python -m pytest {test_cmd}",
                        test_name,
                        timeout=self.timeout_per_test
                    )

                    if success:
                        batch_passed += 1
                    else:
                        batch_failed += 1

                print(f"批次 {batch_name} 结果: 通过 {batch_passed}, 失败 {batch_failed}")
                total_passed += batch_passed
                total_failed += batch_failed
        else:
            # 整体执行模式
            print("\n🎯 执行完整基础设施层测试套件")
            print("-" * 40)

            all_tests = []
            for tests in all_test_batches.values():
                all_tests.extend(tests)

            for test_cmd in all_tests:
                test_name = test_cmd.split('/')[-1].replace('.py', '')
                success = self.run_test_with_timeout(
                    f"python -m pytest {test_cmd}",
                    test_name,
                    timeout=self.timeout_per_test
                )

                if success:
                    total_passed += 1
                else:
                    total_failed += 1

        # 生成测试报告
        self.generate_report(total_passed, total_failed, total_timeout)

    def run_coverage_analysis(self):
        """运行覆盖率分析"""
        print("\n📊 执行覆盖率分析")
        print("-" * 40)

        coverage_cmd = (
            "python -m pytest tests/unit/infrastructure/ "
            "--cov=src/infrastructure "
            "--cov-report=html "
            "--cov-report=term-missing "
            "--cov-report=json:reports/infrastructure_coverage.json "
            f"--timeout={self.timeout_per_test} "
            "--durations=10 "
            "--durations-min=2.0 "
            "-q"
        )

        success = self.run_test_with_timeout(
            coverage_cmd,
            "infrastructure_coverage",
            timeout=self.timeout_per_test * 5  # 覆盖率分析给更多时间
        )

        if success:
            print("✅ 覆盖率分析完成")
            # 读取覆盖率结果
            coverage_file = self.project_root / "reports" / "infrastructure_coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    print(
                        f"📈 覆盖率统计: {coverage_data.get('totals', {}).get('percent_covered', 'N/A')}%")
                except Exception as e:
                    print(f"⚠️  无法读取覆盖率数据: {e}")
        else:
            print("❌ 覆盖率分析失败")

    def generate_report(self, passed, failed, timeout):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📋 基础设施层测试报告")
        print("=" * 60)

        total_tests = passed + failed + timeout

        print(f"总测试数: {total_tests}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"⏰ 超时: {timeout}")

        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(".1f")
        else:
            print("成功率: N/A")

        # 保存详细结果
        report_file = self.project_root / "reports" / "infrastructure_test_report.json"
        report_file.parent.mkdir(exist_ok=True)

        report_data = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'timeout': timeout,
                'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0
            },
            'details': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"详细报告已保存: {report_file}")

        # 输出耗时最长的测试
        if self.test_results:
            sorted_results = sorted(
                self.test_results.items(),
                key=lambda x: x[1].get('duration', 0),
                reverse=True
            )

            print("\n⏱️  耗时最长的测试:")
            for i, (test_name, result) in enumerate(sorted_results[:5]):
                duration = result.get('duration', 0)
                status = result.get('status', 'unknown')
                print("2d")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基础设施层测试运行器')
    parser.add_argument('--timeout', type=int, default=120,
                        help='单个测试超时时间(秒)')
    parser.add_argument('--batch', action='store_true',
                        help='启用分批执行模式')
    parser.add_argument('--coverage', action='store_true',
                        help='执行覆盖率分析')
    parser.add_argument('--workers', type=int, default=4,
                        help='最大并行工作数')

    args = parser.parse_args()

    runner = InfrastructureTestRunner(
        timeout_per_test=args.timeout,
        max_workers=args.workers
    )

    if args.coverage:
        runner.run_coverage_analysis()
    else:
        runner.run_infrastructure_tests(batch_mode=args.batch)


if __name__ == "__main__":
    main()
