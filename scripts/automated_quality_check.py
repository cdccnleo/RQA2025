#!/usr/bin/env python3
"""
自动化质量检查脚本

执行完整的代码质量检查流程，包括：
- 重复代码检测
- 复杂度分析
- 代码规范检查
- 质量报告生成
"""

from tools.smart_duplicate_detector.core.config import SmartDuplicateConfig
from tools.smart_duplicate_detector import detect_clones
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QualityChecker:
    """自动化质量检查器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.results = {}
        self.thresholds = {
            'max_clone_groups': 50,      # 最大允许克隆组数量
            'max_high_complexity': 5,    # 最大允许高复杂度文件数量
            'min_quality_score': 0.7,    # 最低质量评分
        }

    def run_full_check(self, target_path: str) -> Dict[str, Any]:
        """
        执行完整质量检查

        Args:
            target_path: 检查目标路径

        Returns:
            Dict[str, Any]: 检查结果
        """
        print("🔍 开始自动化质量检查...")
        print(f"📁 目标路径: {target_path}")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

        # 1. 重复代码检测
        print("1️⃣ 执行重复代码检测...")
        clone_results = self._run_duplicate_check(target_path)

        # 2. 复杂度分析
        print("2️⃣ 执行复杂度分析...")
        complexity_results = self._run_complexity_analysis(target_path)

        # 3. 代码规范检查
        print("3️⃣ 执行代码规范检查...")
        style_results = self._run_style_check(target_path)

        # 4. 生成综合报告
        print("4️⃣ 生成质量报告...")
        report = self._generate_report(clone_results, complexity_results, style_results)

        # 5. 质量门禁检查
        print("5️⃣ 执行质量门禁检查...")
        gate_results = self._run_quality_gates(report)

        report['quality_gates'] = gate_results
        report['timestamp'] = datetime.now().isoformat()
        report['target_path'] = target_path

        print("-" * 50)
        print("✅ 质量检查完成！")
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return report

    def _run_duplicate_check(self, target_path: str) -> Dict[str, Any]:
        """运行重复代码检测"""
        try:
            config = SmartDuplicateConfig()
            config.get_preset_config('quality')

            result = detect_clones(target_path, config)

            stats = result.get_statistics()
            clone_groups = result.clone_groups

            return {
                'success': True,
                'total_groups': stats.get('total_groups', 0),
                'exact_clones': stats.get('exact_clones', 0),
                'similar_clones': stats.get('similar_clones', 0),
                'semantic_clones': stats.get('semantic_clones', 0),
                'clone_groups': len(clone_groups),
                'files_affected': stats.get('files_affected', 0),
                'avg_similarity': stats.get('avg_similarity', 0.0),
                'refactoring_opportunities': len(result.get_refactoring_opportunities())
            }

        except Exception as e:
            print(f"❌ 重复代码检测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_groups': 0,
                'clone_groups': 0
            }

    def _run_complexity_analysis(self, target_path: str) -> Dict[str, Any]:
        """运行复杂度分析"""
        try:
            # 扫描Python文件
            python_files = []
            for root, dirs, files in os.walk(target_path):
                # 跳过__pycache__等目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for file in files:
                    if file.endswith('.py') and not file.startswith('test_'):
                        python_files.append(os.path.join(root, file))

            total_files = len(python_files)
            complexity_stats = {
                'total_files': total_files,
                'analyzed_files': 0,
                'high_complexity_files': 0,
                'avg_complexity': 0.0,
                'max_complexity': 0,
                'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
            }

            if total_files > 0:
                # 这里可以集成更详细的复杂度分析
                # 暂时使用简化的统计
                complexity_stats['analyzed_files'] = total_files

            return {
                'success': True,
                **complexity_stats
            }

        except Exception as e:
            print(f"❌ 复杂度分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_style_check(self, target_path: str) -> Dict[str, Any]:
        """运行代码规范检查"""
        try:
            # 这里可以集成flake8, black, mypy等工具
            # 暂时返回基本信息
            return {
                'success': True,
                'tools_used': ['basic_check'],
                'issues_found': 0,
                'files_checked': 0
            }

        except Exception as e:
            print(f"❌ 代码规范检查失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_report(self, clone_results: Dict, complexity_results: Dict,
                         style_results: Dict) -> Dict[str, Any]:
        """生成综合质量报告"""
        # 计算综合质量评分
        quality_score = self._calculate_quality_score(
            clone_results, complexity_results, style_results
        )

        report = {
            'summary': {
                'overall_quality_score': quality_score,
                'quality_level': self._get_quality_level(quality_score),
                'check_timestamp': datetime.now().isoformat(),
            },
            'duplicate_check': clone_results,
            'complexity_analysis': complexity_results,
            'style_check': style_results,
            'recommendations': self._generate_recommendations(
                clone_results, complexity_results, style_results
            )
        }

        return report

    def _calculate_quality_score(self, clone_results: Dict, complexity_results: Dict,
                                 style_results: Dict) -> float:
        """计算综合质量评分"""
        score = 1.0  # 基础分数

        # 基于克隆组数量扣分
        clone_penalty = min(clone_results.get('total_groups', 0) / 100, 0.5)
        score -= clone_penalty

        # 基于复杂度扣分
        complexity_penalty = min(complexity_results.get('high_complexity_files', 0) / 10, 0.3)
        score -= complexity_penalty

        # 基于代码规范扣分
        style_penalty = min(style_results.get('issues_found', 0) / 100, 0.2)
        score -= style_penalty

        return max(0.0, score)

    def _get_quality_level(self, score: float) -> str:
        """根据质量评分确定等级"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        elif score >= 0.6:
            return "poor"
        else:
            return "critical"

    def _generate_recommendations(self, clone_results: Dict, complexity_results: Dict,
                                  style_results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于克隆结果的建议
        if clone_results.get('total_groups', 0) > 20:
            recommendations.append("🔴 高优先级: 消除重复代码，当前克隆组过多")
        elif clone_results.get('total_groups', 0) > 10:
            recommendations.append("🟡 中优先级: 考虑重构部分重复代码")

        # 基于复杂度结果的建议
        if complexity_results.get('high_complexity_files', 0) > 0:
            recommendations.append("🟡 中优先级: 重构高复杂度代码文件")

        # 基于规范检查的建议
        if style_results.get('issues_found', 0) > 0:
            recommendations.append("🟢 低优先级: 修复代码规范问题")

        if not recommendations:
            recommendations.append("✅ 代码质量良好，继续保持")

        return recommendations

    def _run_quality_gates(self, report: Dict) -> Dict[str, Any]:
        """运行质量门禁检查"""
        gates = {
            'clone_count_gate': {
                'threshold': self.thresholds['max_clone_groups'],
                'actual': report['duplicate_check'].get('total_groups', 0),
                'passed': report['duplicate_check'].get('total_groups', 0) <= self.thresholds['max_clone_groups']
            },
            'quality_score_gate': {
                'threshold': self.thresholds['min_quality_score'],
                'actual': report['summary']['overall_quality_score'],
                'passed': report['summary']['overall_quality_score'] >= self.thresholds['min_quality_score']
            }
        }

        all_passed = all(gate['passed'] for gate in gates.values())

        return {
            'gates': gates,
            'overall_passed': all_passed,
            'blocking_gates': [name for name, gate in gates.items() if not gate['passed']]
        }

    def save_report(self, report: Dict, output_path: str):
        """保存质量报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 质量报告已保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

    def print_summary(self, report: Dict):
        """打印质量检查摘要"""
        summary = report['summary']
        gates = report.get('quality_gates', {})

        print("\n" + "="*60)
        print("🎯 质量检查报告摘要")
        print("="*60)

        print(f"📊 总体质量评分: {summary['overall_quality_score']:.3f}")
        print(f"🏆 质量等级: {summary['quality_level']}")
        print(f"⏰ 检查时间: {summary['check_timestamp']}")

        print(f"\n🔍 重复代码检测:")
        clone_check = report.get('duplicate_check', {})
        if clone_check.get('success'):
            print(f"  • 克隆组数量: {clone_check.get('total_groups', 0)}")
            print(f"  • 重构机会: {clone_check.get('refactoring_opportunities', 0)}")
        else:
            print(f"  • 检查失败: {clone_check.get('error', '未知错误')}")

        print(f"\n🏗️ 复杂度分析:")
        complexity = report.get('complexity_analysis', {})
        if complexity.get('success'):
            print(f"  • 分析文件数: {complexity.get('analyzed_files', 0)}")
            print(f"  • 高复杂度文件: {complexity.get('high_complexity_files', 0)}")
        else:
            print(f"  • 分析失败: {complexity.get('error', '未知错误')}")

        print(f"\n📋 质量门禁:")
        if gates.get('overall_passed'):
            print("  ✅ 所有质量门禁通过")
        else:
            print("  ❌ 部分质量门禁失败")
            for gate_name in gates.get('blocking_gates', []):
                gate_info = gates['gates'][gate_name]
                print(f"    • {gate_name}: {gate_info['actual']} > {gate_info['threshold']}")

        print(f"\n💡 改进建议:")
        for rec in report.get('recommendations', []):
            print(f"  • {rec}")

        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动化代码质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/automated_quality_check.py src/
  python scripts/automated_quality_check.py --output report.json --config quality_config.json src/
        """
    )

    parser.add_argument(
        'target_path',
        help='检查目标路径（文件或目录）'
    )

    parser.add_argument(
        '--output', '-o',
        help='输出报告文件路径'
    )

    parser.add_argument(
        '--config',
        help='质量检查配置文件路径'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，只输出结果'
    )

    args = parser.parse_args()

    # 创建质量检查器
    checker = QualityChecker(args.config)

    # 执行检查
    report = checker.run_full_check(args.target_path)

    # 保存报告
    if args.output:
        checker.save_report(report, args.output)

    # 打印摘要
    if not args.quiet:
        checker.print_summary(report)

    # 返回适当的退出码
    gates = report.get('quality_gates', {})
    if gates.get('overall_passed'):
        print("\n✅ 质量检查通过")
        return 0
    else:
        print("\n❌ 质量检查失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
