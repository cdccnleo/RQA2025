#!/usr/bin/env python3
"""
简化版基础设施层质量监控系统

定期审查测试覆盖情况，建立质量趋势分析
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


class SimpleQualityMonitor:
    """简化版质量监控器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"

    def run_quick_test_check(self) -> dict:
        """运行快速测试检查"""
        print("🧪 运行快速测试检查...")

        test_cmd = [
            "python", "-m", "pytest",
            "tests/unit/infrastructure/test_cache_system.py",
            "tests/unit/infrastructure/test_logging_system.py",
            "-v", "--tb=no", "-q"
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                test_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            end_time = time.time()
            duration = end_time - start_time

            # 简单解析结果
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout + stderr
            passed = output.count('PASSED')
            failed = output.count('FAILED')
            errors = output.count('ERROR')

            return {
                'success': result.returncode == 0,
                'total_tests': passed + failed + errors,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'duration': duration,
                'exit_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': 'timeout',
                'duration': 300
            }

    def run_coverage_check(self) -> dict:
        """运行覆盖率检查"""
        print("📊 运行覆盖率检查...")

        coverage_cmd = [
            "python", "-m", "pytest",
            "tests/unit/infrastructure/",
            "--cov=src/infrastructure",
            "--cov-report=term-missing",
            "--cov-report=json:reports/coverage.json",
            "-q"
        ]

        try:
            result = subprocess.run(
                coverage_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            if result.returncode == 0:
                # 读取覆盖率结果
                coverage_file = self.reports_dir / "coverage.json"
                if coverage_file.exists():
                    try:
                        with open(coverage_file, 'r') as f:
                            data = json.load(f)
                        coverage = data.get('totals', {}).get('percent_covered', 0)
                        return {
                            'success': True,
                            'coverage_percent': coverage,
                            'coverage_data': data
                        }
                    except:
                        pass

            return {
                'success': False,
                'error': 'Failed to get coverage data'
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Coverage check timeout'
            }

    def generate_quality_report(self, test_results: dict, coverage_results: dict) -> dict:
        """生成质量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'coverage_results': coverage_results,
            'quality_score': 0,
            'recommendations': []
        }

        # 计算质量分数
        score = 0

        # 测试通过率评分 (40%)
        if test_results.get('total_tests', 0) > 0:
            success_rate = test_results.get('passed', 0) / test_results.get('total_tests', 0)
            score += success_rate * 40

        # 覆盖率评分 (40%)
        if coverage_results.get('success', False):
            coverage = coverage_results.get('coverage_percent', 0)
            score += (coverage / 100) * 40

        # 执行时间评分 (20%)
        duration = test_results.get('duration', 0)
        if duration < 60:  # 1分钟内完成
            score += 20
        elif duration < 120:  # 2分钟内完成
            score += 15
        elif duration < 300:  # 5分钟内完成
            score += 10
        else:
            score += 5

        report['quality_score'] = score

        # 生成建议
        if test_results.get('failed', 0) > 0:
            report['recommendations'].append("修复失败的测试用例")

        if coverage_results.get('coverage_percent', 0) < 50:
            report['recommendations'].append("提高测试覆盖率")

        if test_results.get('duration', 0) > 120:
            report['recommendations'].append("优化测试执行时间")

        return report

    def print_report(self, report: dict):
        """打印质量报告"""
        print("\n" + "=" * 60)
        print("📋 基础设施层质量监控报告")
        print("=" * 60)

        # 测试结果
        test_results = report.get('test_results', {})
        print("🧪 测试结果:")
        print(f"   总测试数: {test_results.get('total_tests', 0)}")
        print(f"   通过: {test_results.get('passed', 0)}")
        print(f"   失败: {test_results.get('failed', 0)}")
        print(f"   执行时间: {test_results.get('duration', 0):.1f}秒")

        # 覆盖率结果
        coverage_results = report.get('coverage_results', {})
        if coverage_results.get('success', False):
            print(f"\n📊 覆盖率: {coverage_results.get('coverage_percent', 0):.1f}%")
        else:
            print("\n📊 覆盖率: 获取失败")

        # 质量分数
        score = report.get('quality_score', 0)
        print(".1f")
        if score >= 80:
            print("   等级: 优秀 🏆")
        elif score >= 60:
            print("   等级: 良好 👍")
        elif score >= 40:
            print("   等级: 需要改进 ⚠️")
        else:
            print("   等级: 需紧急处理 ❌")

        # 建议
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\n💡 改进建议:")
            for rec in recommendations:
                print(f"   • {rec}")

        print(f"\n📅 生成时间: {report.get('timestamp', 'unknown')}")

    def save_report(self, report: dict):
        """保存报告"""
        self.reports_dir.mkdir(exist_ok=True)

        report_file = self.reports_dir / "quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📁 报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版基础设施层质量监控系统')
    parser.add_argument('--full', action='store_true',
                        help='运行完整检查（包括覆盖率）')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    monitor = SimpleQualityMonitor(project_root)

    print("🔄 开始质量监控...")

    # 运行测试检查
    test_results = monitor.run_quick_test_check()

    coverage_results = {}
    if args.full:
        # 运行覆盖率检查
        coverage_results = monitor.run_coverage_check()
    else:
        coverage_results = {'success': False, 'message': '跳过覆盖率检查'}

    # 生成报告
    report = monitor.generate_quality_report(test_results, coverage_results)

    # 打印和保存报告
    monitor.print_report(report)
    monitor.save_report(report)

    return 0


if __name__ == "__main__":
    exit(main())
