#!/usr/bin/env python3
"""
质量测试运行器

执行项目的单元测试和集成测试，确保代码质量。
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QualityTestRunner:
    """质量测试运行器"""

    def __init__(self, test_dir: str = "tests", coverage_dir: str = "coverage_reports"):
        self.test_dir = Path(test_dir)
        self.coverage_dir = Path(coverage_dir)
        self.coverage_dir.mkdir(exist_ok=True)

    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试

        Returns:
            Dict[str, Any]: 测试结果
        """
        print("🧪 开始运行质量测试...")
        print("=" * 40)

        results = {
            'unit_tests': self._run_unit_tests(),
            'integration_tests': self._run_integration_tests(),
            'quality_tests': self._run_quality_tests(),
            'coverage_analysis': self._run_coverage_analysis(),
            'performance_tests': self._run_performance_tests()
        }

        # 生成综合报告
        summary = self._generate_test_summary(results)

        results['summary'] = summary

        print("=" * 40)
        print("✅ 测试运行完成")

        return results

    def _run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        print("1️⃣ 运行单元测试...")

        try:
            # 使用pytest运行单元测试
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.test_dir / "unit"),
                "--tb=short",
                "--quiet",
                "--disable-warnings"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_count': self._extract_test_count(result.stdout),
                'passed_count': self._extract_passed_count(result.stdout),
                'failed_count': self._extract_failed_count(result.stdout)
            }

        except Exception as e:
            print(f"❌ 单元测试运行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        print("2️⃣ 运行集成测试...")

        try:
            # 使用pytest运行集成测试
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.test_dir / "integration"),
                "--tb=short",
                "--quiet",
                "--disable-warnings"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except Exception as e:
            print(f"❌ 集成测试运行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_quality_tests(self) -> Dict[str, Any]:
        """运行质量相关测试"""
        print("3️⃣ 运行质量测试...")

        try:
            # 运行自动化质量检查作为测试
            quality_script = project_root / "scripts" / "automated_quality_check.py"
            cmd = [
                sys.executable, str(quality_script),
                "--quiet",
                "--output", str(self.coverage_dir / "quality_test_report.json"),
                str(project_root / "src")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

            # 读取质量报告
            quality_report = {}
            try:
                import json
                with open(self.coverage_dir / "quality_test_report.json", 'r', encoding='utf-8') as f:
                    quality_report = json.load(f)
            except:
                pass

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'quality_score': quality_report.get('summary', {}).get('overall_quality_score', 0),
                'quality_level': quality_report.get('summary', {}).get('quality_level', 'unknown'),
                'gates_passed': quality_report.get('quality_gates', {}).get('overall_passed', False)
            }

        except Exception as e:
            print(f"❌ 质量测试运行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """运行覆盖率分析"""
        print("4️⃣ 运行覆盖率分析...")

        try:
            # 使用coverage运行测试并生成报告
            coverage_file = self.coverage_dir / ".coverage"
            coverage_report = self.coverage_dir / "coverage_report.xml"

            # 先收集覆盖率数据
            env = os.environ.copy()
            env['COVERAGE_FILE'] = str(coverage_file)

            cmd_collect = [
                sys.executable, "-m", "coverage", "run",
                "--source", "src",
                "-m", "pytest", str(self.test_dir),
                "--tb=no", "--quiet"
            ]

            result_collect = subprocess.run(cmd_collect, env=env, cwd=project_root)

            if result_collect.returncode == 0:
                # 生成覆盖率报告
                cmd_report = [
                    sys.executable, "-m", "coverage", "xml",
                    "-o", str(coverage_report)
                ]

                result_report = subprocess.run(cmd_report, env=env, cwd=project_root,
                                               capture_output=True, text=True)

                # 解析覆盖率报告
                coverage_percentage = self._parse_coverage_report(coverage_report)

                return {
                    'success': True,
                    'coverage_percentage': coverage_percentage,
                    'report_file': str(coverage_report)
                }
            else:
                return {
                    'success': False,
                    'error': 'Coverage collection failed'
                }

        except Exception as e:
            print(f"❌ 覆盖率分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        print("5️⃣ 运行性能测试...")

        try:
            # 这里可以运行性能基准测试
            # 暂时返回基本信息
            return {
                'success': True,
                'tests_run': 0,
                'avg_response_time': 0.0,
                'memory_usage': 0.0,
                'note': '性能测试暂未实现'
            }

        except Exception as e:
            print(f"❌ 性能测试运行失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_test_count(self, output: str) -> int:
        """从pytest输出中提取测试数量"""
        try:
            # 查找类似 "5 passed, 2 failed" 的模式
            import re
            match = re.search(r'(\d+)\s+(?:passed|failed|error)', output)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0

    def _extract_passed_count(self, output: str) -> int:
        """从pytest输出中提取通过测试数量"""
        try:
            import re
            match = re.search(r'(\d+)\s+passed', output)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0

    def _extract_failed_count(self, output: str) -> int:
        """从pytest输出中提取失败测试数量"""
        try:
            import re
            match = re.search(r'(\d+)\s+failed', output)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0

    def _parse_coverage_report(self, report_file: Path) -> float:
        """解析覆盖率报告XML文件"""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(report_file)
            root = tree.getroot()

            # 查找总覆盖率
            coverage_attr = root.get('line-rate')
            if coverage_attr:
                return float(coverage_attr) * 100

        except Exception as e:
            print(f"解析覆盖率报告失败: {e}")

        return 0.0

    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            'overall_success': True,
            'tests_passed': 0,
            'tests_failed': 0,
            'coverage_percentage': 0.0,
            'quality_score': 0.0,
            'recommendations': []
        }

        # 检查各个测试类型的成功状态
        for test_type, result in results.items():
            if test_type != 'summary' and isinstance(result, dict):
                if not result.get('success', False):
                    summary['overall_success'] = False

        # 收集统计信息
        unit_tests = results.get('unit_tests', {})
        summary['tests_passed'] = unit_tests.get('passed_count', 0)
        summary['tests_failed'] = unit_tests.get('failed_count', 0)

        coverage = results.get('coverage_analysis', {})
        summary['coverage_percentage'] = coverage.get('coverage_percentage', 0.0)

        quality = results.get('quality_tests', {})
        summary['quality_score'] = quality.get('quality_score', 0.0)

        # 生成建议
        recommendations = []

        if summary['tests_failed'] > 0:
            recommendations.append(f"🔴 修复 {summary['tests_failed']} 个失败的单元测试")

        if summary['coverage_percentage'] < 80:
            recommendations.append(f"🟡 提高测试覆盖率，当前 {summary['coverage_percentage']:.1f}%")

        if summary['quality_score'] < 0.7:
            recommendations.append(f"🟡 改善代码质量，当前评分 {summary['quality_score']:.3f}")

        if not recommendations:
            recommendations.append("✅ 所有测试和质量检查都通过")

        summary['recommendations'] = recommendations

        return summary

    def save_report(self, results: Dict[str, Any], output_path: str):
        """保存测试报告"""
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 测试报告已保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

    def print_summary(self, results: Dict[str, Any]):
        """打印测试总结"""
        summary = results.get('summary', {})

        print("\n" + "="*60)
        print("🧪 质量测试报告摘要")
        print("="*60)

        status = "✅ 通过" if summary.get('overall_success') else "❌ 失败"
        print(f"整体状态: {status}")

        print(f"单元测试: {summary.get('tests_passed', 0)} 通过, {summary.get('tests_failed', 0)} 失败")
        print(f"覆盖率: {summary.get('coverage_percentage', 0):.1f}%")
        print(f"质量评分: {summary.get('quality_score', 0):.3f}")

        print(f"\n💡 建议:")
        for rec in summary.get('recommendations', []):
            print(f"  • {rec}")

        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="质量测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/run_quality_tests.py
  python scripts/run_quality_tests.py --output test_report.json --coverage-only
        """
    )

    parser.add_argument(
        '--output', '-o',
        help='输出报告文件路径'
    )

    parser.add_argument(
        '--test-dir',
        default='tests',
        help='测试目录路径'
    )

    parser.add_argument(
        '--coverage-only',
        action='store_true',
        help='仅运行覆盖率分析'
    )

    args = parser.parse_args()

    # 创建测试运行器
    runner = QualityTestRunner(args.test_dir)

    # 运行测试
    if args.coverage_only:
        results = {'coverage_analysis': runner._run_coverage_analysis()}
    else:
        results = runner.run_all_tests()

    # 保存报告
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'quality_test_report_{timestamp}.json'

    runner.save_report(results, output_path)

    # 打印摘要
    runner.print_summary(results)

    # 返回适当的退出码
    summary = results.get('summary', {})
    if summary.get('overall_success', False):
        print("\n✅ 质量测试通过")
        return 0
    else:
        print("\n❌ 质量测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
