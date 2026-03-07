#!/usr/bin/env python3
"""
RQA2025 项目最终质量验证脚本
验证项目整体测试覆盖率和质量达标情况
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any


class QualityValidator:
    """质量验证器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}

    def run_test_coverage_analysis(self) -> Dict[str, Any]:
        """运行测试覆盖率分析"""
        print("🔍 运行测试覆盖率分析...")

        try:
            # 运行pytest并生成覆盖率报告
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/unit/",
                "--tb=no",
                "--quiet",
                "--cov=src",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 解析覆盖率报告
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)

                coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)

                # 解析测试结果
                passed_tests = 0
                failed_tests = 0
                skipped_tests = 0

                for line in result.stdout.split('\n'):
                    if 'passed' in line and 'failed' in line and 'skipped' in line:
                        # 解析类似 "==== 104 passed, 5 failed, 38 skipped, 130 deselected in 114.58s ====" 的行
                        parts = line.split()
                        for part in parts:
                            if 'passed' in part:
                                passed_tests = int(parts[parts.index(part) - 1])
                            elif 'failed' in part:
                                failed_tests = int(parts[parts.index(part) - 1])
                            elif 'skipped' in part:
                                skipped_tests = int(parts[parts.index(part) - 1])

                return {
                    'coverage_percent': coverage_percent,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'skipped_tests': skipped_tests,
                    'total_tests': passed_tests + failed_tests + skipped_tests,
                    'success_rate': (passed_tests / max(passed_tests + failed_tests, 1)) * 100
                }
            else:
                return {'error': 'Coverage report not generated'}

        except subprocess.TimeoutExpired:
            return {'error': 'Test execution timed out'}
        except Exception as e:
            return {'error': f'Test execution failed: {str(e)}'}

    def validate_layer_coverage(self) -> Dict[str, Any]:
        """验证各层级覆盖情况"""
        print("🔍 验证各层级覆盖情况...")

        layers = {
            'infrastructure': 'src/infrastructure',
            'data': 'src/data',
            'ml': 'src/ml',
            'strategy': 'src/strategy',
            'trading': 'src/trading',
            'risk': 'src/risk'
        }

        layer_results = {}

        for layer_name, layer_path in layers.items():
            try:
                # 检查目录是否存在
                layer_dir = self.project_root / layer_path
                if not layer_dir.exists():
                    layer_results[layer_name] = {'status': 'not_found'}
                    continue

                # 查找对应的测试文件
                test_dir = self.project_root / f"tests/unit/{layer_name}"
                if test_dir.exists():
                    test_files = list(test_dir.rglob("test_*.py"))
                    layer_results[layer_name] = {
                        'status': 'has_tests',
                        'test_files_count': len(test_files),
                        'test_files': [f.name for f in test_files[:5]]  # 只显示前5个
                    }
                else:
                    layer_results[layer_name] = {'status': 'no_tests'}

            except Exception as e:
                layer_results[layer_name] = {'status': 'error', 'message': str(e)}

        return layer_results

    def check_integration_tests(self) -> Dict[str, Any]:
        """检查集成测试"""
        print("🔍 检查集成测试...")

        integration_dir = self.project_root / "tests/integration"
        if integration_dir.exists():
            integration_files = list(integration_dir.glob("test_*.py"))
            return {
                'integration_tests_exist': True,
                'integration_test_files': len(integration_files),
                'files': [f.name for f in integration_files]
            }
        else:
            return {
                'integration_tests_exist': False,
                'message': 'Integration tests directory not found'
            }

    def validate_quality_standards(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证质量标准"""
        print("🔍 验证质量标准...")

        standards = {
            'minimum_coverage': 15.0,  # 最低覆盖率要求
            'minimum_success_rate': 90.0,  # 最低成功率要求
            'maximum_failed_tests': 10  # 最大失败测试数
        }

        validation_results = {}

        # 检查覆盖率
        coverage = test_results.get('coverage_percent', 0)
        validation_results['coverage_check'] = {
            'actual': coverage,
            'required': standards['minimum_coverage'],
            'passed': coverage >= standards['minimum_coverage']
        }

        # 检查成功率
        success_rate = test_results.get('success_rate', 0)
        validation_results['success_rate_check'] = {
            'actual': success_rate,
            'required': standards['minimum_success_rate'],
            'passed': success_rate >= standards['minimum_success_rate']
        }

        # 检查失败测试数
        failed_tests = test_results.get('failed_tests', 0)
        validation_results['failed_tests_check'] = {
            'actual': failed_tests,
            'maximum_allowed': standards['maximum_failed_tests'],
            'passed': failed_tests <= standards['maximum_failed_tests']
        }

        # 总体评估
        all_checks_passed = all(
            check['passed'] for check in validation_results.values()
            if isinstance(check, dict) and 'passed' in check
        )

        validation_results['overall_assessment'] = {
            'production_ready': all_checks_passed,
            'recommendation': 'Ready for production' if all_checks_passed else 'Needs improvement'
        }

        return validation_results

    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        print("📊 生成最终质量验证报告...")

        # 运行所有验证
        test_results = self.run_test_coverage_analysis()
        layer_results = self.validate_layer_coverage()
        integration_results = self.check_integration_tests()
        quality_validation = self.validate_quality_standards(test_results)

        # 编译最终报告
        final_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'project': 'RQA2025',
            'validation_type': 'Final Quality Assessment',

            'test_coverage_analysis': test_results,
            'layer_coverage_validation': layer_results,
            'integration_tests_check': integration_results,
            'quality_standards_validation': quality_validation,

            'summary': {
                'overall_coverage': test_results.get('coverage_percent', 0),
                'total_tests': test_results.get('total_tests', 0),
                'success_rate': test_results.get('success_rate', 0),
                'production_ready': quality_validation.get('overall_assessment', {}).get('production_ready', False),
                'recommendation': quality_validation.get('overall_assessment', {}).get('recommendation', 'Unknown')
            }
        }

        return final_report

    def print_report(self, report: Dict[str, Any]):
        """打印报告"""
        print("\n" + "="*80)
        print("🎯 RQA2025 项目最终质量验证报告")
        print("="*80)

        print(f"\n📅 生成时间: {report['timestamp']}")
        print(f"🏗️ 项目: {report['project']}")
        print(f"🔍 验证类型: {report['validation_type']}")

        # 测试覆盖率分析
        test_analysis = report['test_coverage_analysis']
        print(f"\n📊 测试覆盖率分析:")
        print(f"   - 总体覆盖率: {test_analysis.get('coverage_percent', 0):.1f}%")
        print(f"   - 通过测试: {test_analysis.get('passed_tests', 0)}")
        print(f"   - 失败测试: {test_analysis.get('failed_tests', 0)}")
        print(f"   - 跳过测试: {test_analysis.get('skipped_tests', 0)}")
        print(f"   - 成功率: {test_analysis.get('success_rate', 0):.1f}%")

        # 质量标准验证
        quality_check = report['quality_standards_validation']
        print(f"\n✅ 质量标准验证:")
        for check_name, check_result in quality_check.items():
            if check_name != 'overall_assessment' and isinstance(check_result, dict):
                status = "✅" if check_result.get('passed', False) else "❌"
                print(f"   {status} {check_name}: {check_result.get('actual', 'N/A')} (要求: {check_result.get('required', 'N/A')})")

        # 总体评估
        summary = report['summary']
        print(f"\n🏆 总体评估:")
        production_ready = summary['production_ready']
        status_icon = "🟢" if production_ready else "🔴"
        print(f"   {status_icon} 生产就绪: {'是' if production_ready else '否'}")
        print(f"   📋 建议: {summary['recommendation']}")

        print("\n" + "="*80)
        if production_ready:
            print("🎉 恭喜！项目已达到投产质量标准，可以安全上线！")
        else:
            print("⚠️ 项目需要进一步优化以达到投产质量标准。")
        print("="*80)


def main():
    """主函数"""
    print("🚀 开始RQA2025项目最终质量验证...")

    validator = QualityValidator()
    report = validator.generate_final_report()
    validator.print_report(report)

    # 保存报告到文件
    report_file = Path(__file__).parent.parent / "final_quality_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细报告已保存至: {report_file}")

    # 返回退出码
    production_ready = report['summary']['production_ready']
    return 0 if production_ready else 1


if __name__ == "__main__":
    sys.exit(main())
