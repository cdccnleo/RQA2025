#!/usr/bin/env python3
"""
基础设施层自动化测试运行器

自动发现和运行所有测试，生成测试报告。

作者: RQA2025 Team
版本: 1.0.0
"""

import sys
import os
import unittest
import json
import time
from datetime import datetime
from pathlib import Path
from unittest.runner import TextTestResult
from unittest.loader import TestLoader


class TestResultCollector(TextTestResult):
    """自定义测试结果收集器"""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []

    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.test_start_time
        self.test_results.append({
            'name': str(test),
            'status': 'PASS',
            'duration': duration,
            'error': None
        })

    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.test_start_time
        self.test_results.append({
            'name': str(test),
            'status': 'ERROR',
            'duration': duration,
            'error': str(err[1])
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.test_start_time
        self.test_results.append({
            'name': str(test),
            'status': 'FAIL',
            'duration': duration,
            'error': str(err[1])
        })


class InfrastructureTestRunner:
    """基础设施层测试运行器"""

    def __init__(self, test_directory="src/infrastructure/tests"):
        self.test_directory = Path(test_directory)
        self.project_root = Path.cwd()

        # 添加项目路径
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def discover_tests(self):
        """自动发现测试"""
        loader = TestLoader()

        # 发现tests目录下的所有测试
        if self.test_directory.exists():
            suite = loader.discover(str(self.test_directory), pattern="test_*.py")
        else:
            # 如果没有tests目录，查找整个infrastructure目录下的测试文件
            suite = unittest.TestSuite()
            infra_dir = Path("src/infrastructure")

            for py_file in infra_dir.rglob("test_*.py"):
                try:
                    module_name = str(py_file.relative_to(self.project_root)).replace('/', '.').replace('\\', '.')[:-3]
                    module = __import__(module_name, fromlist=[''])
                    suite.addTests(loader.loadTestsFromModule(module))
                except Exception as e:
                    print(f"⚠️ 无法加载测试文件 {py_file}: {e}")

        return suite

    def run_tests(self, verbosity=2):
        """运行测试"""
        print("🔍 发现测试用例...")

        # 发现测试
        suite = self.discover_tests()

        if suite.countTestCases() == 0:
            print("❌ 未发现任何测试用例")
            return None

        print(f"📋 发现 {suite.countTestCases()} 个测试用例")

        # 创建结果收集器
        result_collector = TestResultCollector(sys.stdout, descriptions=True, verbosity=verbosity)

        # 运行测试
        print("\n🧪 开始运行测试...")
        start_time = time.time()

        runner = unittest.TextTestRunner(
            stream=sys.stdout,
            verbosity=verbosity,
            resultclass=lambda stream, descriptions, verbosity: result_collector
        )

        result = runner.run(suite)
        end_time = time.time()

        # 生成报告
        report = self._generate_report(result, result_collector.test_results, end_time - start_time)

        return report

    def _generate_report(self, result, test_results, duration):
        """生成测试报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': result.testsRun,
                'passed': len(result.successes) if hasattr(result, 'successes') else 0,
                'failed': len(result.failures),
                'errors': len(result.errors),
                'duration': duration,
                'success_rate': 0.0
            },
            'results': test_results,
            'failures': [
                {
                    'test': str(test),
                    'error': str(err)
                } for test, err in result.failures
            ],
            'errors': [
                {
                    'test': str(test),
                    'error': str(err)
                } for test, err in result.errors
            ]
        }

        # 计算成功率
        total = report['summary']['total_tests']
        if total > 0:
            passed = report['summary']['passed']
            failed = report['summary']['failed']
            errors = report['summary']['errors']
            report['summary']['success_rate'] = (passed / total) * 100

        return report

    def save_report(self, report, output_path=None):
        """保存测试报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"infrastructure_test_report_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 测试报告已保存至: {output_path}")
        return output_path

    def print_summary(self, report):
        """打印测试摘要"""
        summary = report['summary']

        print("\n📊 测试执行摘要:")
        print(f"  执行时间: {summary['duration']:.2f}秒")
        print(f"  总测试数: {summary['total_tests']}")
        print(f"  通过: {summary['passed']} ✅")
        print(f"  失败: {summary['failed']} ❌")
        print(f"  错误: {summary['errors']} ⚠️")
        print(f"  成功率: {summary['success_rate']:.1f}%")

        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\n❌ 失败的测试:")
            for failure in report['failures'][:5]:  # 只显示前5个
                print(f"  - {failure['test']}")

            for error in report['errors'][:5]:  # 只显示前5个
                print(f"  - {error['test']}")

            if len(report['failures']) + len(report['errors']) > 10:
                print(f"  ... 还有更多失败 (总计 {len(report['failures']) + len(report['errors'])} 个)")


def main():
    """主函数"""
    print("🚀 基础设施层自动化测试运行器")
    print("=" * 50)

    runner = InfrastructureTestRunner()

    try:
        # 运行测试
        report = runner.run_tests()

        if report:
            # 保存报告
            report_file = runner.save_report(report)

            # 打印摘要
            runner.print_summary(report)

            # 返回退出码
            success_rate = report['summary']['success_rate']
            if success_rate >= 90:
                print("\n✅ 测试质量优秀!")
                return 0
            elif success_rate >= 75:
                print("\n⚠️ 测试质量良好")
                return 0
            else:
                print("\n❌ 测试质量需要改进")
                return 1
        else:
            print("❌ 测试执行失败")
            return 1

    except Exception as e:
        print(f"❌ 测试运行器异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
