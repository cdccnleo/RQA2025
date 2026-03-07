#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据库测试运行脚本
"""

from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.infrastructure.config.unified_manager import UnifiedConfigManager as ConfigManager
import sys
import os
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DatabaseTestRunner:
    """数据库测试运行器"""

    def __init__(self):
        """初始化测试运行器"""
        self.config_manager = ConfigManager()
        self.monitor = ApplicationMonitor()
        self.test_results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'summary': {}
        }

    def run_unit_tests(self):
        """运行单元测试"""
        print("🔍 运行数据库单元测试...")

        unit_test_files = [
            'tests/unit/infrastructure/database/test_unified_data_manager.py',
            'tests/unit/infrastructure/database/test_data_consistency_manager.py',
            'tests/unit/infrastructure/database/test_database_health_monitor.py',
            'tests/unit/infrastructure/database/test_postgresql_adapter.py',
            'tests/unit/infrastructure/database/test_influxdb_adapter.py',
            'tests/unit/infrastructure/database/test_redis_adapter.py'
        ]

        results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for test_file in unit_test_files:
            if os.path.exists(test_file):
                print(f"  📋 运行 {test_file}")

                try:
                    # 运行pytest
                    result = subprocess.run([
                        'python', '-m', 'pytest', test_file,
                        '-v', '--tb=short', '--timeout=30'
                    ], capture_output=True, text=True, timeout=60)

                    # 解析结果
                    if result.returncode == 0:
                        status = 'PASSED'
                        passed_tests += 1
                    else:
                        status = 'FAILED'
                        failed_tests += 1

                    total_tests += 1
                    results[test_file] = {
                        'status': status,
                        'output': result.stdout,
                        'error': result.stderr,
                        'return_code': result.returncode
                    }

                    print(f"    ✅ {status}")

                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        'status': 'TIMEOUT',
                        'output': '',
                        'error': 'Test timeout',
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ⏰ TIMEOUT")

                except Exception as e:
                    results[test_file] = {
                        'status': 'ERROR',
                        'output': '',
                        'error': str(e),
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ❌ ERROR: {e}")
            else:
                print(f"  ⚠️  文件不存在: {test_file}")

        self.test_results['unit_tests'] = {
            'results': results,
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }

        print(
            f"  📊 单元测试结果: {passed_tests}/{total_tests} 通过 ({self.test_results['unit_tests']['summary']['success_rate']:.1f}%)")

    def run_integration_tests(self):
        """运行集成测试"""
        print("🔗 运行数据库集成测试...")

        integration_test_files = [
            'tests/integration/database/test_database_integration.py'
        ]

        results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for test_file in integration_test_files:
            if os.path.exists(test_file):
                print(f"  📋 运行 {test_file}")

                try:
                    # 运行pytest
                    result = subprocess.run([
                        'python', '-m', 'pytest', test_file,
                        '-v', '--tb=short', '--timeout=60'
                    ], capture_output=True, text=True, timeout=120)

                    # 解析结果
                    if result.returncode == 0:
                        status = 'PASSED'
                        passed_tests += 1
                    else:
                        status = 'FAILED'
                        failed_tests += 1

                    total_tests += 1
                    results[test_file] = {
                        'status': status,
                        'output': result.stdout,
                        'error': result.stderr,
                        'return_code': result.returncode
                    }

                    print(f"    ✅ {status}")

                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        'status': 'TIMEOUT',
                        'output': '',
                        'error': 'Test timeout',
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ⏰ TIMEOUT")

                except Exception as e:
                    results[test_file] = {
                        'status': 'ERROR',
                        'output': '',
                        'error': str(e),
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ❌ ERROR: {e}")
            else:
                print(f"  ⚠️  文件不存在: {test_file}")

        self.test_results['integration_tests'] = {
            'results': results,
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }

        print(
            f"  📊 集成测试结果: {passed_tests}/{total_tests} 通过 ({self.test_results['integration_tests']['summary']['success_rate']:.1f}%)")

    def run_performance_tests(self):
        """运行性能测试"""
        print("⚡ 运行数据库性能测试...")

        performance_test_files = [
            'tests/performance/database/test_database_performance.py'
        ]

        results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for test_file in performance_test_files:
            if os.path.exists(test_file):
                print(f"  📋 运行 {test_file}")

                try:
                    # 运行pytest
                    result = subprocess.run([
                        'python', '-m', 'pytest', test_file,
                        '-v', '--tb=short', '--timeout=120'
                    ], capture_output=True, text=True, timeout=300)

                    # 解析结果
                    if result.returncode == 0:
                        status = 'PASSED'
                        passed_tests += 1
                    else:
                        status = 'FAILED'
                        failed_tests += 1

                    total_tests += 1
                    results[test_file] = {
                        'status': status,
                        'output': result.stdout,
                        'error': result.stderr,
                        'return_code': result.returncode
                    }

                    print(f"    ✅ {status}")

                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        'status': 'TIMEOUT',
                        'output': '',
                        'error': 'Test timeout',
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ⏰ TIMEOUT")

                except Exception as e:
                    results[test_file] = {
                        'status': 'ERROR',
                        'output': '',
                        'error': str(e),
                        'return_code': -1
                    }
                    failed_tests += 1
                    total_tests += 1
                    print(f"    ❌ ERROR: {e}")
            else:
                print(f"  ⚠️  文件不存在: {test_file}")

        self.test_results['performance_tests'] = {
            'results': results,
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }

        print(
            f"  📊 性能测试结果: {passed_tests}/{total_tests} 通过 ({self.test_results['performance_tests']['summary']['success_rate']:.1f}%)")

    def generate_test_report(self):
        """生成测试报告"""
        print("📝 生成测试报告...")

        # 计算总体统计
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for test_type in ['unit_tests', 'integration_tests', 'performance_tests']:
            if test_type in self.test_results:
                summary = self.test_results[test_type]['summary']
                total_tests += summary['total']
                total_passed += summary['passed']
                total_failed += summary['failed']

        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        self.test_results['summary'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'overall_success_rate': overall_success_rate,
            'timestamp': datetime.now().isoformat(),
            'test_duration': time.time() - self.start_time
        }

        # 保存报告
        report_file = f"reports/testing/database_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        print(f"  📄 测试报告已保存: {report_file}")

        return report_file

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 数据库测试总结")
        print("="*60)

        summary = self.test_results['summary']
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过测试: {summary['total_passed']}")
        print(f"失败测试: {summary['total_failed']}")
        print(f"成功率: {summary['overall_success_rate']:.1f}%")
        print(f"测试耗时: {summary['test_duration']:.2f}秒")

        print("\n详细结果:")
        for test_type in ['unit_tests', 'integration_tests', 'performance_tests']:
            if test_type in self.test_results:
                test_summary = self.test_results[test_type]['summary']
                print(f"  {test_type.replace('_', ' ').title()}: "
                      f"{test_summary['passed']}/{test_summary['total']} "
                      f"({test_summary['success_rate']:.1f}%)")

        print("="*60)

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始运行数据库测试套件...")
        self.start_time = time.time()

        # 运行各类测试
        self.run_unit_tests()
        print()

        self.run_integration_tests()
        print()

        self.run_performance_tests()
        print()

        # 生成报告
        report_file = self.generate_test_report()

        # 打印总结
        self.print_summary()

        # 记录监控指标
        self.monitor.record_metric(
            'database_test_suite',
            self.test_results['summary']['test_duration'],
            {
                'total_tests': self.test_results['summary']['total_tests'],
                'success_rate': self.test_results['summary']['overall_success_rate']
            }
        )

        return self.test_results['summary']['overall_success_rate'] >= 80.0


def main():
    """主函数"""
    runner = DatabaseTestRunner()

    try:
        success = runner.run_all_tests()

        if success:
            print("✅ 数据库测试套件执行成功!")
            sys.exit(0)
        else:
            print("❌ 数据库测试套件执行失败!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试执行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
