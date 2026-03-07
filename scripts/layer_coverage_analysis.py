#!/usr/bin/env python3
"""
RQA2025 分层覆盖率分析工具
系统性分析各个层级的测试覆盖率状态
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

class LayerCoverageAnalyzer:
    """分层覆盖率分析器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.layers = {
            'infrastructure': 'src/infrastructure',
            'data': 'src/data',
            'core': 'src/core',
            'risk': 'src/risk',
            'trading': 'src/trading',
            'ml': 'src/ml'
        }
        self.results = {}

    def analyze_layer(self, layer_name: str, layer_path: str) -> Dict[str, Any]:
        """分析单个层级的覆盖率"""
        print(f"🔍 分析 {layer_name} 层...")

        try:
            # 检查测试目录
            test_dir = self.project_root / f"tests/unit/{layer_name}"
            if not test_dir.exists():
                return {
                    'layer': layer_name,
                    'status': 'missing_tests',
                    'error': f'测试目录不存在: {test_dir}'
                }

            # 运行覆盖率测试
            cmd = [
                sys.executable, '-m', 'pytest',
                f'tests/unit/{layer_name}/',
                '--cov', layer_path,
                '--cov-report', 'json:temp_coverage.json',
                '--tb=no',
                '-q'
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )

            # 解析结果
            analysis = self._parse_pytest_output(result.stdout, result.stderr)

            # 读取覆盖率数据
            coverage_file = self.project_root / 'temp_coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
                analysis['coverage'] = coverage_data
                coverage_file.unlink()  # 删除临时文件

            analysis['layer'] = layer_name
            analysis['status'] = 'success'

            return analysis

        except Exception as e:
            return {
                'layer': layer_name,
                'status': 'error',
                'error': str(e)
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """解析pytest输出"""
        analysis = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'total': 0,
            'pass_rate': 0.0,
            'output': stdout[-500:] if len(stdout) > 500 else stdout,  # 最后500字符
            'warnings': []
        }

        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'passed' in line and 'failed' in line:
                # 解析类似 "30 passed, 2 failed, 1 skipped"
                parts = line.replace(',', '').split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        if i > 0 and 'passed' in parts[i-1]:
                            analysis['passed'] = int(part)
                        elif i > 0 and 'failed' in parts[i-1]:
                            analysis['failed'] = int(part)
                        elif i > 0 and 'skipped' in parts[i-1]:
                            analysis['skipped'] = int(part)
                        elif i > 0 and 'errors' in parts[i-1]:
                            analysis['errors'] = int(part)

                analysis['total'] = analysis['passed'] + analysis['failed'] + analysis['skipped'] + analysis['errors']
                if analysis['total'] > 0:
                    analysis['pass_rate'] = round((analysis['passed'] / analysis['total']) * 100, 1)
                break

        # 提取警告信息
        warning_lines = [line for line in lines if 'WARNING' in line.upper() or 'WARN' in line.upper()]
        analysis['warnings'] = warning_lines[:5]  # 只保留前5个警告

        return analysis

    def analyze_all_layers(self) -> Dict[str, Any]:
        """分析所有层级"""
        print("🚀 开始分层覆盖率分析...")
        print("=" * 60)

        all_results = {}
        for layer_name, layer_path in self.layers.items():
            result = self.analyze_layer(layer_name, layer_path)
            all_results[layer_name] = result

            # 实时输出结果
            if result['status'] == 'success':
                passed = result.get('passed', 0)
                failed = result.get('failed', 0)
                total = result.get('total', 0)
                pass_rate = result.get('pass_rate', 0)
                coverage = result.get('coverage', {}).get('totals', {}).get('percent_covered', 0)

                print("2d"
            elif result['status'] == 'missing_tests':
                print("2d"
            else:
                print("2d"
        print("=" * 60)

        # 生成综合报告
        summary = self._generate_summary(all_results)

        return {
            'timestamp': datetime.now().isoformat(),
            'layers': all_results,
            'summary': summary
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合摘要"""
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_tests = 0
        coverage_values = []

        layer_status = {}

        for layer_name, result in results.items():
            if result['status'] == 'success':
                total_passed += result.get('passed', 0)
                total_failed += result.get('failed', 0)
                total_skipped += result.get('skipped', 0)
                total_errors += result.get('errors', 0)
                total_tests += result.get('total', 0)

                coverage = result.get('coverage', {}).get('totals', {}).get('percent_covered', 0)
                if coverage > 0:
                    coverage_values.append(coverage)

                layer_status[layer_name] = {
                    'status': 'healthy' if result.get('pass_rate', 0) >= 95 else 'needs_attention',
                    'pass_rate': result.get('pass_rate', 0),
                    'coverage': coverage,
                    'tests': result.get('total', 0)
                }
            else:
                layer_status[layer_name] = {
                    'status': 'problematic',
                    'error': result.get('error', '未知错误')
                }

        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0

        # 优先级排序
        priority_order = self._calculate_priority_order(layer_status)

        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_skipped': total_skipped,
            'total_errors': total_errors,
            'overall_pass_rate': round(overall_pass_rate, 1),
            'average_coverage': round(avg_coverage, 1),
            'layer_status': layer_status,
            'improvement_priority': priority_order,
            'recommendations': self._generate_recommendations(layer_status, priority_order)
        }

    def _calculate_priority_order(self, layer_status: Dict[str, Any]) -> List[str]:
        """计算改进优先级"""
        priority_scores = {}

        for layer_name, status in layer_status.items():
            if status['status'] == 'healthy':
                priority_scores[layer_name] = 100  # 已健康，优先级最低
            elif status['status'] == 'needs_attention':
                pass_rate = status.get('pass_rate', 0)
                coverage = status.get('coverage', 0)
                # 优先级 = (100 - 通过率) * 0.7 + (100 - 覆盖率) * 0.3
                priority_scores[layer_name] = (100 - pass_rate) * 0.7 + (100 - coverage) * 0.3
            else:  # problematic
                priority_scores[layer_name] = 200  # 有问题的层级优先级最高

        # 按优先级分数降序排序
        return sorted(priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True)

    def _generate_recommendations(self, layer_status: Dict[str, Any], priority_order: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 检查整体状态
        problematic_layers = [layer for layer, status in layer_status.items() if status['status'] == 'problematic']
        if problematic_layers:
            recommendations.append(f"🔴 紧急修复: {', '.join(problematic_layers)} 层存在严重问题")

        # 检查需要关注的层级
        attention_layers = [layer for layer, status in layer_status.items() if status['status'] == 'needs_attention']
        if attention_layers:
            recommendations.append(f"🟡 重点改进: {', '.join(attention_layers)} 层的测试质量需要提升")

        # 优先级改进路径
        if len(priority_order) > 1:
            top_priority = priority_order[0]
            recommendations.append(f"🎯 建议优先改进: {top_priority} 层 (优先级最高)")

        # 覆盖率目标
        healthy_layers = [layer for layer, status in layer_status.items() if status['status'] == 'healthy']
        if healthy_layers:
            recommendations.append(f"✅ 保持优势: {', '.join(healthy_layers)} 层测试质量良好")

        # 总体建议
        if not recommendations:
            recommendations.append("🎉 所有层级测试质量良好，继续保持！")

        return recommendations

    def save_report(self, report: Dict[str, Any], output_file: str = None):
        """保存分析报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"layer_coverage_analysis_{timestamp}.json"

        report_path = self.project_root / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 分析报告已保存: {report_path}")
        return report_path

    def print_summary_table(self, report: Dict[str, Any]):
        """打印摘要表格"""
        print("\n📊 分层测试覆盖率分析报告")
        print("=" * 80)

        summary = report['summary']
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")

        print("\n🎯 各层级状态:")
        print("-" * 80)

        for layer_name in self.layers.keys():
            status = summary['layer_status'].get(layer_name, {})
            if status.get('status') == 'healthy':
                print("<15")
            elif status.get('status') == 'needs_attention':
                print("<15")
            else:
                print("<15")

        print("\n💡 改进建议:")
        for rec in summary.get('recommendations', []):
            print(f"  • {rec}")

        print("\n🎯 改进优先级:")
        for i, layer in enumerate(summary.get('improvement_priority', []), 1):
            print(f"  {i}. {layer}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    analyzer = LayerCoverageAnalyzer(project_root)

    # 执行分析
    report = analyzer.analyze_all_layers()

    # 保存报告
    report_file = analyzer.save_report(report)

    # 打印摘要
    analyzer.print_summary_table(report)

    print("
📈 详细报告已保存到: {}".format(report_file))
    print("🎯 下一步: 按优先级顺序改进各层级测试覆盖率"

if __name__ == "__main__":
    main()
