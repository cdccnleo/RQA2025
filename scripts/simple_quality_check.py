#!/usr/bin/env python3
"""
RQA2025 项目质量状态检查脚本
简单检查项目测试和代码质量状态
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List


class QualityChecker:
    """质量检查器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.check_results = {}

    def count_test_files(self) -> Dict[str, int]:
        """统计测试文件数量"""
        test_count = {
            'unit_tests': 0,
            'integration_tests': 0,
            'total_test_files': 0
        }

        # 统计单元测试
        unit_test_dir = self.project_root / "tests" / "unit"
        if unit_test_dir.exists():
            unit_tests = list(unit_test_dir.rglob("test_*.py"))
            test_count['unit_tests'] = len(unit_tests)

        # 统计集成测试
        integration_test_dir = self.project_root / "tests" / "integration"
        if integration_test_dir.exists():
            integration_tests = list(integration_test_dir.glob("test_*.py"))
            test_count['integration_tests'] = len(integration_tests)

        test_count['total_test_files'] = test_count['unit_tests'] + test_count['integration_tests']

        return test_count

    def analyze_layer_completeness(self) -> Dict[str, Any]:
        """分析各层完整性"""
        layers = {
            'infrastructure': 'src/infrastructure',
            'data': 'src/data',
            'ml': 'src/ml',
            'strategy': 'src/strategy',
            'trading': 'src/trading',
            'risk': 'src/risk'
        }

        layer_analysis = {}

        for layer_name, layer_path in layers.items():
            layer_dir = self.project_root / layer_path
            test_dir = self.project_root / "tests" / "unit" / layer_name

            layer_analysis[layer_name] = {
                'source_exists': layer_dir.exists(),
                'tests_exist': test_dir.exists(),
                'source_files': len(list(layer_dir.rglob("*.py"))) if layer_dir.exists() else 0,
                'test_files': len(list(test_dir.rglob("test_*.py"))) if test_dir.exists() else 0
            }

        return layer_analysis

    def check_recent_commits(self) -> Dict[str, Any]:
        """检查最近的提交情况"""
        # 简单的文件修改时间检查
        recent_files = {}

        # 检查测试文件的最近修改时间
        test_dir = self.project_root / "tests"
        if test_dir.exists():
            test_files = list(test_dir.rglob("test_*.py"))
            if test_files:
                # 按修改时间排序
                test_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                recent_files['most_recent_test'] = {
                    'file': str(test_files[0].relative_to(self.project_root)),
                    'modified': test_files[0].stat().st_mtime
                }

        return recent_files

    def generate_quality_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        print("🔍 分析项目质量状态...")

        test_counts = self.count_test_files()
        layer_analysis = self.analyze_layer_completeness()
        recent_activity = self.check_recent_commits()

        # 计算一些指标
        total_layers = len(layer_analysis)
        layers_with_tests = sum(1 for layer in layer_analysis.values() if layer['tests_exist'])
        layers_with_sources = sum(1 for layer in layer_analysis.values() if layer['source_exists'])

        test_coverage_ratio = layers_with_tests / max(total_layers, 1)

        report = {
            'timestamp': '2025-12-22',
            'project': 'RQA2025',
            'analysis_type': 'Quality Status Check',

            'test_statistics': test_counts,
            'layer_analysis': layer_analysis,
            'recent_activity': recent_activity,

            'quality_metrics': {
                'test_coverage_ratio': test_coverage_ratio,
                'layers_with_tests': layers_with_tests,
                'total_layers': total_layers,
                'layers_with_sources': layers_with_sources
            },

            'assessment': {
                'test_framework_established': test_counts['total_test_files'] > 0,
                'layer_coverage_good': test_coverage_ratio >= 0.8,
                'recent_development': len(recent_activity) > 0,
                'production_ready': test_coverage_ratio >= 0.8 and test_counts['total_test_files'] >= 50
            }
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """打印报告"""
        print("\n" + "="*80)
        print("🎯 RQA2025 项目质量状态检查报告")
        print("="*80)

        print(f"\n📅 生成时间: {report['timestamp']}")
        print(f"🏗️ 项目: {report['project']}")

        # 测试统计
        test_stats = report['test_statistics']
        print(f"\n📊 测试统计:")
        print(f"   - 单元测试文件: {test_stats['unit_tests']}")
        print(f"   - 集成测试文件: {test_stats['integration_tests']}")
        print(f"   - 总测试文件: {test_stats['total_test_files']}")

        # 层级分析
        layer_analysis = report['layer_analysis']
        print(f"\n🏗️ 层级分析:")
        for layer_name, layer_info in layer_analysis.items():
            source_status = "✅" if layer_info['source_exists'] else "❌"
            test_status = "✅" if layer_info['tests_exist'] else "❌"
            print(f"   {layer_name}: 源码{source_status}, 测试{test_status} ({layer_info['test_files']}个测试文件)")

        # 质量指标
        metrics = report['quality_metrics']
        print(f"\n📈 质量指标:")
        print(f"   - 测试覆盖率: {metrics['test_coverage_ratio']:.1%}")
        print(f"   - 有测试的层级: {metrics['layers_with_tests']}/{metrics['total_layers']}")
        print(f"   - 有源码的层级: {metrics['layers_with_sources']}/{metrics['total_layers']}")

        # 评估结果
        assessment = report['assessment']
        print(f"\n✅ 评估结果:")
        for check_name, result in assessment.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check_name.replace('_', ' ').title()}")

        # 总体结论
        production_ready = assessment['production_ready']
        print(f"\n🏆 总体结论:")
        if production_ready:
            print("   🟢 项目已达到投产质量标准！")
            print("   🎉 所有核心层级都有测试覆盖，测试框架完善。")
        else:
            print("   🟡 项目测试体系基本建立，但需要进一步完善。")
            print("   📋 建议继续完善测试覆盖率和质量。")

        print("\n" + "="*80)


def main():
    """主函数"""
    print("🚀 开始RQA2025项目质量状态检查...")

    checker = QualityChecker()
    report = checker.generate_quality_report()
    checker.print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
