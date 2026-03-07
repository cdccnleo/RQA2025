#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险层测试运行脚本
提供带超时控制的风险层测试执行
"""

import subprocess
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime


class RiskTestRunner:
    """风险层测试运行器"""

    def __init__(self):
        self.test_results = {}
        self.start_time = None

    def run_tests_with_timeout(self, test_path, timeout=60, marker=None):
        """
        运行带超时的测试

        Args:
            test_path: 测试路径
            timeout: 超时时间(秒)
            marker: pytest标记
        """
        print(f"\n{'='*60}")
        print(f"运行风险层测试: {test_path}")
        print(f"超时设置: {timeout}秒")
        if marker:
            print(f"测试标记: {marker}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        self.start_time = time.time()

        # 构建pytest命令
        cmd = [
            sys.executable, '-m', 'pytest',
            test_path,
            '--tb=short',
            '--timeout={}'.format(timeout),
            '--timeout-method=thread',
            '--durations=20',
            '--durations-min=1.0',
            '-v'
        ]

        # 如果有标记，添加标记过滤
        if marker:
            cmd.extend(['-m', marker])

        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=timeout * 10  # 整体超时是单个测试的10倍
            )

            end_time = time.time()
            duration = end_time - self.start_time

            # 解析结果
            passed = 0
            failed = 0
            skipped = 0

            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line and 'skipped' in line:
                    parts = line.split(',')
                    for part in parts:
                        if 'passed' in part:
                            passed = int(part.strip().split()[0])
                        elif 'failed' in part:
                            failed = int(part.strip().split()[0])
                        elif 'skipped' in part:
                            skipped = int(part.strip().split()[0])

            self.test_results[test_path] = {
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'duration': duration,
                'timeout': timeout,
                'marker': marker,
                'exit_code': result.returncode,
                'success': result.returncode == 0
            }

            # 输出结果
            self._print_test_summary(test_path)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"❌ 测试超时: {test_path} (超过 {timeout * 10} 秒)")
            return False
        except Exception as e:
            print(f"❌ 测试执行出错: {e}")
            return False

    def _print_test_summary(self, test_path):
        """打印测试总结"""
        result = self.test_results[test_path]
        print(f"\n📊 测试结果汇总 - {test_path}")
        print(f"✅ 通过: {result['passed']}")
        print(f"❌ 失败: {result['failed']}")
        print(f"⏭️  跳过: {result['skipped']}")
        print(f"⏱️  耗时: {result['duration']:.2f}秒")
        print(f"🎯 超时设置: {result['timeout']}秒")
        if result['marker']:
            print(f"🏷️  标记: {result['marker']}")

        total_tests = result['passed'] + result['failed'] + result['skipped']
        if total_tests > 0:
            pass_rate = (result['passed'] / total_tests) * 100
            print(f"📊 通过率: {pass_rate:.1f}%")
        if result['success']:
            print("✅ 测试执行成功")
        else:
            print("❌ 测试执行失败")

    def run_risk_layer_tests(self, categories=None):
        """
        运行风险层测试，按类别分组

        Args:
            categories: 要运行的测试类别列表
        """
        if categories is None:
            categories = ['basic', 'threading', 'complex', 'integration']

        test_configs = {
            'basic': {
                'path': 'tests/unit/risk/',
                'timeout': 30,
                'marker': None,
                'description': '基础风险计算测试'
            },
            'threading': {
                'path': 'tests/unit/risk/',
                'timeout': 60,
                'marker': 'risk_threading',
                'description': '多线程风险监控测试'
            },
            'complex': {
                'path': 'tests/unit/risk/',
                'timeout': 90,
                'marker': 'risk_complex',
                'description': '复杂风险计算测试'
            },
            'integration': {
                'path': 'tests/unit/risk/',
                'timeout': 120,
                'marker': None,
                'description': '集成测试'
            }
        }

        overall_start = time.time()
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        print("🚀 开始风险层测试覆盖率统计")
        print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for category in categories:
            if category in test_configs:
                config = test_configs[category]
                print(f"\n🎯 执行测试类别: {category} - {config['description']}")

                success = self.run_tests_with_timeout(
                    config['path'],
                    config['timeout'],
                    config['marker']
                )

                if category in self.test_results:
                    result = self.test_results[category]
                    total_passed += result['passed']
                    total_failed += result['failed']
                    total_skipped += result['skipped']

        # 输出总体统计
        overall_end = time.time()
        total_duration = overall_end - overall_start

        print(f"\n{'='*80}")
        print("📈 风险层测试总体统计")
        print(f"{'='*80}")
        print(f"✅ 总通过数: {total_passed}")
        print(f"❌ 总失败数: {total_failed}")
        print(f"⏭️  总跳过数: {total_skipped}")
        print(f"⏱️  总耗时: {total_duration:.2f}秒")
        print(f"🎯 测试类别: {', '.join(categories)}")

        total_tests = total_passed + total_failed + total_skipped
        if total_tests > 0:
            overall_pass_rate = (total_passed / total_tests) * 100
            print(f"📊 总体通过率: {overall_pass_rate:.1f}%")
            if overall_pass_rate >= 75:
                print("🎉 优秀! 测试通过率达到目标")
            elif overall_pass_rate >= 60:
                print("👍 良好! 测试通过率可接受")
            else:
                print("⚠️  需要改进! 测试通过率偏低")

        # 保存结果到文件
        self._save_results_to_file()

    def _save_results_to_file(self):
        """保存测试结果到文件"""
        results_file = Path("test_results_risk_layer.json")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_data = {
            'timestamp': timestamp,
            'test_results': self.test_results,
            'summary': {
                'total_passed': sum(r['passed'] for r in self.test_results.values()),
                'total_failed': sum(r['failed'] for r in self.test_results.values()),
                'total_skipped': sum(r['skipped'] for r in self.test_results.values()),
                'total_duration': sum(r['duration'] for r in self.test_results.values())
            }
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 测试结果已保存到: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='风险层测试运行器')
    parser.add_argument('--category', '-c', choices=['basic', 'threading', 'complex', 'integration', 'all'],
                        default='all', help='测试类别')
    parser.add_argument('--timeout', '-t', type=int, default=60,
                        help='单个测试超时时间(秒)')
    parser.add_argument('--file', '-f', help='指定测试文件路径')

    args = parser.parse_args()

    runner = RiskTestRunner()

    if args.file:
        # 运行指定文件
        runner.run_tests_with_timeout(args.file, args.timeout)
    else:
        # 运行类别测试
        categories = ['basic', 'threading', 'complex',
                      'integration'] if args.category == 'all' else [args.category]
        runner.run_risk_layer_tests(categories)


if __name__ == '__main__':
    main()
