#!/usr/bin/env python3
"""
RQA2025 质量门禁报告生成器

生成综合质量报告并执行质量门禁检查
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class QualityGateGenerator:
    """质量门禁生成器"""

    def __init__(self):
        self.quality_gates = {
            'consistency_score': {
                'name': '代码文档一致性评分',
                'threshold': 95,
                'operator': '>=',
                'critical': True
            },
            'sync_success_rate': {
                'name': '文档同步成功率',
                'threshold': 95,
                'operator': '>=',
                'critical': True
            },
            'version_consistency': {
                'name': '版本一致性检查',
                'threshold': True,
                'operator': '==',
                'critical': True
            },
            'critical_issues': {
                'name': '严重问题数量',
                'threshold': 0,
                'operator': '<=',
                'critical': True
            },
            'documentation_coverage': {
                'name': '文档覆盖率',
                'threshold': 90,
                'operator': '>=',
                'critical': False
            }
        }

    def load_report(self, file_path: str) -> Dict[str, Any]:
        """加载报告文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 报告文件不存在 {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析JSON文件 {file_path}: {e}")
            return {}

    def evaluate_gate(self, gate_config: Dict, value: Any) -> Dict[str, Any]:
        """评估质量门禁"""
        threshold = gate_config['threshold']
        operator = gate_config['operator']

        if operator == '>=':
            passed = value >= threshold
        elif operator == '<=':
            passed = value <= threshold
        elif operator == '==':
            passed = value == threshold
        elif operator == '>':
            passed = value > threshold
        elif operator == '<':
            passed = value < threshold
        else:
            passed = False

        return {
            'name': gate_config['name'],
            'threshold': threshold,
            'operator': operator,
            'actual_value': value,
            'passed': passed,
            'critical': gate_config['critical'],
            'message': f"{gate_config['name']}: {value} {operator} {threshold} = {'✅ 通过' if passed else '❌ 失败'}"
        }

    def generate_comprehensive_report(self,
                                      consistency_report_path: str,
                                      doc_sync_report_path: str,
                                      version_report_path: str,
                                      output_path: str) -> Dict[str, Any]:
        """生成综合质量报告"""

        # 加载各个报告
        consistency_report = self.load_report(consistency_report_path)
        doc_sync_report = self.load_report(doc_sync_report_path)
        version_report = self.load_report(version_report_path)

        # 提取关键指标
        metrics = self._extract_metrics(
            consistency_report,
            doc_sync_report,
            version_report
        )

        # 评估质量门禁
        quality_gates = {}
        failed_gates = []

        for gate_name, gate_config in self.quality_gates.items():
            if gate_name in metrics:
                gate_result = self.evaluate_gate(gate_config, metrics[gate_name])
                quality_gates[gate_name] = gate_result

                if not gate_result['passed']:
                    failed_gates.append(gate_result)

        # 生成综合报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'quality_gate_report',
            'summary': {
                'consistency_score': metrics.get('consistency_score', 0),
                'sync_success_rate': metrics.get('sync_success_rate', 0),
                'version_consistency': metrics.get('version_consistency', False),
                'critical_issues': metrics.get('critical_issues', 0),
                'documentation_coverage': metrics.get('documentation_coverage', 0),
                'total_gates': len(quality_gates),
                'passed_gates': sum(1 for g in quality_gates.values() if g['passed']),
                'failed_gates': len(failed_gates),
                'critical_failures': sum(1 for g in failed_gates if g['critical'])
            },
            'metrics': metrics,
            'quality_gates': quality_gates,
            'failed_gates': failed_gates,
            'recommendations': self._generate_recommendations(failed_gates, metrics)
        }

        # 保存报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _extract_metrics(self,
                         consistency_report: Dict,
                         doc_sync_report: Dict,
                         version_report: Dict) -> Dict[str, Any]:
        """提取关键指标"""

        metrics = {}

        # 一致性评分
        if consistency_report:
            summary = consistency_report.get('summary', {})
            metrics['consistency_score'] = summary.get('consistency_score', 0)
            metrics['critical_issues'] = summary.get('critical_issues', 0)
            metrics['documentation_coverage'] = summary.get('documentation_coverage', 0)

        # 文档同步成功率
        if doc_sync_report:
            sync_results = doc_sync_report.get('sync_results', {})
            total_layers = len(sync_results)
            successful_syncs = sum(1 for r in sync_results.values() if r.get('success', False))
            metrics['sync_success_rate'] = (
                successful_syncs / total_layers * 100) if total_layers > 0 else 0

        # 版本一致性
        if version_report:
            summary = version_report.get('summary', {})
            metrics['version_consistency'] = summary.get('consistency_check_passed', False)

        return metrics

    def _generate_recommendations(self,
                                  failed_gates: List[Dict],
                                  metrics: Dict) -> List[str]:
        """生成改进建议"""

        recommendations = []

        for gate in failed_gates:
            gate_name = gate['name']

            if gate_name == '代码文档一致性评分':
                recommendations.append("🔧 提高代码文档一致性：更新过时的文档，确保文档与代码实现同步")
            elif gate_name == '文档同步成功率':
                recommendations.append("🔧 修复文档同步问题：检查文档模板和同步脚本，确保所有层都能正确同步")
            elif gate_name == '版本一致性检查':
                recommendations.append("🔧 修复版本不一致：更新版本号，确保所有组件使用相同版本")
            elif gate_name == '严重问题数量':
                recommendations.append("🔧 解决严重问题：优先修复关键一致性问题和架构问题")
            elif gate_name == '文档覆盖率':
                recommendations.append("📝 提高文档覆盖率：为未文档化的组件添加架构文档")

        # 基于指标的通用建议
        consistency_score = metrics.get('consistency_score', 0)
        if consistency_score < 80:
            recommendations.append("🚨 紧急：一致性评分严重偏低，需要立即进行全面的代码文档同步")

        if not metrics.get('version_consistency', False):
            recommendations.append("🔄 版本管理：建立统一的版本管理流程，确保版本号一致性")

        return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 质量门禁报告生成器')
    parser.add_argument('--consistency-report', required=True,
                        help='一致性检查报告路径')
    parser.add_argument('--doc-sync-report', required=True,
                        help='文档同步报告路径')
    parser.add_argument('--version-report', required=True,
                        help='版本管理报告路径')
    parser.add_argument('--output', required=True,
                        help='输出报告路径')

    args = parser.parse_args()

    generator = QualityGateGenerator()
    report = generator.generate_comprehensive_report(
        args.consistency_report,
        args.doc_sync_report,
        args.version_report,
        args.output
    )

    print("质量门禁报告生成完成")
    print(f"输出文件: {args.output}")

    summary = report['summary']
    print("
          == = 报告摘要 == =")
    print(f"一致性评分: {summary['consistency_score']}%")
    print(f"同步成功率: {summary['sync_success_rate']}%")
    print(f"版本一致性: {'✅' if summary['version_consistency'] else '❌'}")
    print(f"质量门禁: {summary['passed_gates']}/{summary['total_gates']} 通过")

    if summary['failed_gates'] > 0:
        print(f"\n❌ 失败门禁: {summary['failed_gates']} 个")
        for gate in report['failed_gates']:
            print(f"  - {gate['message']}")
    else:
        print("\n✅ 所有质量门禁通过")


if __name__ == "__main__":
    main()
