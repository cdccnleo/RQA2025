#!/usr/bin/env python3
"""
分层分阶段覆盖率验证脚本
验证19个层级的测试覆盖率
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


class LayeredCoverageAnalyzer:
    """分层覆盖率分析器"""

    def __init__(self):
        self.layers = {
            'adapters': 'src/adapters/',
            'async': 'src/async/',
            'automation': 'src/automation/',
            'boundary': 'src/boundary/',
            'core': 'src/core/',
            'data': 'src/data/',
            'distributed': 'src/distributed/',
            'features': 'src/features/',
            'gateway': 'src/gateway/',
            'infrastructure': 'src/infrastructure/',
            'ml': 'src/ml/',
            'mobile': 'src/mobile/',
            'monitoring': 'src/monitoring/',
            'optimization': 'src/optimization/',
            'resilience': 'src/resilience/',
            'risk': 'src/risk/',
            'strategy': 'src/strategy/',
            'streaming': 'src/streaming/',
            'trading': 'src/trading/'
        }

    def run_coverage_for_layer(self, layer_name: str, layer_path: str) -> Dict:
        """为单个层运行覆盖率测试"""
        print(f"🔍 正在分析 {layer_name} 层覆盖率...")

        # 构建测试路径
        test_path = f"tests/unit/{layer_name}/"
        if not os.path.exists(test_path):
            test_path = f"tests/unit/{layer_name.replace('_', '')}/"
        if not os.path.exists(test_path):
            test_path = "tests/unit/"

        try:
            # 运行覆盖率测试
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov", layer_path,
                "--cov-report", "json",
                "--cov-report", "term-missing",
                "--cov-report", "html",
                test_path,
                "-q", "--tb=no"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # 解析覆盖率数据
            coverage_file = f".coverage.{layer_name}.json"
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)

                # 计算统计信息
                total_statements = 0
                covered_statements = 0
                total_branches = 0
                covered_branches = 0

                for file_path, file_data in coverage_data.get('files', {}).items():
                    if file_path.startswith(layer_path):
                        total_statements += file_data.get('summary', {}).get('num_statements', 0)
                        covered_statements += file_data.get('summary', {}).get('covered_lines', 0)
                        total_branches += file_data.get('summary', {}).get('num_branches', 0)
                        covered_branches += file_data.get('summary', {}).get('covered_branches', 0)

                coverage_percent = (covered_statements / total_statements *
                                    100) if total_statements > 0 else 0

                return {
                    'layer': layer_name,
                    'coverage_percent': round(coverage_percent, 2),
                    'total_statements': total_statements,
                    'covered_statements': covered_statements,
                    'total_branches': total_branches,
                    'covered_branches': covered_branches,
                    'status': 'success' if coverage_percent > 0 else 'no_tests'
                }

        except Exception as e:
            print(f"❌ {layer_name} 层分析失败: {str(e)}")

        return {
            'layer': layer_name,
            'coverage_percent': 0.0,
            'total_statements': 0,
            'covered_statements': 0,
            'total_branches': 0,
            'covered_branches': 0,
            'status': 'error'
        }

    def analyze_layer_structure(self, layer_path: str) -> Dict:
        """分析层级结构"""
        if not os.path.exists(layer_path):
            return {'files': 0, 'directories': 0, 'total_lines': 0}

        total_files = 0
        total_lines = 0

        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('.py'):
                    total_files += 1
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            lines = len(f.readlines())
                            total_lines += lines
                    except:
                        pass

        return {
            'files': total_files,
            'directories': len(dirs),
            'total_lines': total_lines
        }

    def generate_coverage_report(self) -> Dict:
        """生成分层覆盖率报告"""
        print("🚀 开始分层分阶段覆盖率验证...")

        results = {}
        summary = {
            'total_layers': len(self.layers),
            'analyzed_layers': 0,
            'total_coverage': 0.0,
            'weighted_coverage': 0.0,
            'total_statements': 0,
            'covered_statements': 0,
            'layers_coverage': []
        }

        for layer_name, layer_path in self.layers.items():
            print(f"\n📊 分析 {layer_name} 层...")

            # 分析层级结构
            structure = self.analyze_layer_structure(layer_path)

            # 运行覆盖率测试
            coverage_result = self.run_coverage_for_layer(layer_name, layer_path)

            # 合并结果
            result = {
                **coverage_result,
                **structure
            }

            results[layer_name] = result
            summary['analyzed_layers'] += 1
            summary['total_statements'] += result['total_statements']
            summary['covered_statements'] += result['covered_statements']

            if result['status'] == 'success':
                summary['layers_coverage'].append(result['coverage_percent'])

        # 计算总体覆盖率
        if summary['total_statements'] > 0:
            summary['total_coverage'] = round(
                summary['covered_statements'] / summary['total_statements'] * 100, 2
            )

        # 计算加权覆盖率
        if summary['layers_coverage']:
            summary['weighted_coverage'] = round(
                sum(summary['layers_coverage']) / len(summary['layers_coverage']), 2
            )

        return {
            'summary': summary,
            'layers': results,
            'timestamp': str(Path.cwd()),
            'report_version': '1.0'
        }

    def save_report(self, report: Dict, filename: str = 'layered_coverage_report.json'):
        """保存覆盖率报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ 覆盖率报告已保存到: {filename}")

    def print_summary_report(self, report: Dict):
        """打印总结报告"""
        print("\n" + "="*80)
        print("🎯 RQA2025 分层分阶段覆盖率验证报告")
        print("="*80)

        summary = report['summary']

        print("
📊 总体统计: "        print(f"   • 总层级数: {summary['total_layers']}")
        print(f"   • 已分析层级: {summary['analyzed_layers']}")
        print(f"   • 总语句数: {summary['total_statements']:,}")
        print(f"   • 已覆盖语句: {summary['covered_statements']:,}")
        print(".2f")
        print(".2f")

        print("
🏆 分层覆盖率排名: "        sorted_layers=sorted(
            report['layers'].items(),
            key=lambda x: x[1]['coverage_percent'],
            reverse=True
        )

        for i, (layer_name, data) in enumerate(sorted_layers[:10], 1):
            status_icon="✅" if data['status'] == 'success' else "❌"
            print("2d"
                  "8.2f")

        print("
📁 层级结构分析: " for layer_name, data in report['layers'].items():
            print(f"   • {layer_name}: {data['files']} 个文件, {data['total_lines']:,} 行代码")

        print("\n🎉 验证完成！")
        print("="*80)

def main():
    """主函数"""
    analyzer=LayeredCoverageAnalyzer()

    try:
        # 生成覆盖率报告
        report=analyzer.generate_coverage_report()

        # 保存报告
        analyzer.save_report(report)

        # 打印总结
        analyzer.print_summary_report(report)

        # 更新 TEST_COVERAGE_IMPROVEMENT_PLAN.md
        update_plan_file(report)

    except Exception as e:
        print(f"❌ 生成覆盖率报告失败: {str(e)}")
        sys.exit(1)

def update_plan_file(report: Dict):
    """更新测试覆盖率改进计划文件"""
    plan_file="TEST_COVERAGE_IMPROVEMENT_PLAN.md"

    if not os.path.exists(plan_file):
        print(f"⚠️  未找到 {plan_file} 文件")
        return

    try:
        with open(plan_file, 'r', encoding='utf-8') as f:
            content=f.read()

        # 生成覆盖率更新内容
        coverage_section=generate_coverage_section(report)

        # 查找并替换覆盖率部分
        if "## 📊 **覆盖率验证报告**" in content:
            # 如果已经有覆盖率部分，替换它
            parts=content.split("## 📊 **覆盖率验证报告**")
            new_content=parts[0] + coverage_section
            if len(parts) > 1:
                # 保留后面的内容
                remaining_parts=parts[1].split("\n## ", 1)
                if len(remaining_parts) > 1:
                    new_content += "\n## " + remaining_parts[1]
        else:
            # 如果没有覆盖率部分，添加到合适位置
            new_content=content + "\n\n" + coverage_section

        # 保存更新后的文件
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 已更新 {plan_file}")

    except Exception as e:
        print(f"❌ 更新 {plan_file} 失败: {str(e)}")

def generate_coverage_section(report: Dict) -> str:
    """生成覆盖率部分内容"""
    summary=report['summary']

    section="## 📊 **覆盖率验证报告**\n\n"
    section += "### 🎯 **总体覆盖率统计**\n\n"
    section += "| 指标 | 值 |\n"
    section += "|------|-----|\n"
    section += f"| 总层级数 | {summary['total_layers']} |\n"
    section += f"| 已分析层级 | {summary['analyzed_layers']} |\n"
    section += f"| 总语句数 | {summary['total_statements']:,} |\n"
    section += f"| 已覆盖语句 | {summary['covered_statements']:,} |\n"
    section += ".2f"    section += ".2f"    section += "\n\n"

    section += "### 🏆 **分层覆盖率详情**\n\n"
    section += "| 层级 | 覆盖率 | 语句数 | 覆盖语句 | 状态 |\n"
    section += "|------|--------|--------|----------|------|\n"

    sorted_layers=sorted(
        report['layers'].items(),
        key=lambda x: x[1]['coverage_percent'],
        reverse=True
    )

    for layer_name, data in sorted_layers:
        status_icon="✅" if data['status'] == 'success' else "❌"
        section += "8.2f"        section += f"| {data['covered_statements']:,} | {status_icon} |\n"

    section += "\n\n"

    section += "### 📈 **覆盖率趋势分析**\n\n"
    section += "```json\n"
    section += json.dumps({
        "summary": summary,
        "top_layers": [
            {
                "name": layer[0],
                "coverage": layer[1]['coverage_percent'],
                "statements": layer[1]['total_statements']
            } for layer in sorted_layers[:5]
        ]
    }, indent=2, ensure_ascii=False)
    section += "\n```\n\n"

    section += "### 📋 **验证完成时间**\n\n"
    section += f"- 生成时间: {report.get('timestamp', '未知')}\n"
    section += f"- 报告版本: {report.get('report_version', '1.0')}\n\n"

    return section

if __name__ == "__main__":
    main()
