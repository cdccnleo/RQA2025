#!/usr/bin/env python3
"""
创建简化的分层覆盖率报告
"""

import json
import os
from pathlib import Path
from typing import Dict


def analyze_coverage_by_layer():
    """按层分析覆盖率"""
    if not os.path.exists('coverage_layered.json'):
        print("❌ 未找到覆盖率文件")
        return {}

    with open('coverage_layered.json', 'r') as f:
        coverage_data = json.load(f)

    # 定义19个层级
    layers = {
        'adapters': [],
        'async': [],
        'automation': [],
        'boundary': [],
        'core': [],
        'data': [],
        'distributed': [],
        'features': [],
        'gateway': [],
        'infrastructure': [],
        'ml': [],
        'mobile': [],
        'monitoring': [],
        'optimization': [],
        'resilience': [],
        'risk': [],
        'strategy': [],
        'streaming': [],
        'trading': []
    }

    # 统计每个文件的覆盖率
    for file_path, file_data in coverage_data.get('files', {}).items():
        # 将Windows路径转换为标准路径
        file_path = file_path.replace('\\', '/')

        # 确定文件属于哪个层级
        layer_found = False
        for layer_name in layers.keys():
            if f'src/{layer_name}/' in file_path:
                layers[layer_name].append(file_data)
                layer_found = True
                break

        # 如果没找到匹配的层级，可能是根级文件
        if not layer_found and file_path.startswith('src/'):
            layers['core'].append(file_data)  # 归类到core层

    # 计算每个层级的统计信息
    layer_stats = {}
    total_statements = 0
    total_covered = 0

    for layer_name, files in layers.items():
        if not files:
            layer_stats[layer_name] = {
                'coverage_percent': 0.0,
                'total_statements': 0,
                'covered_statements': 0,
                'files_count': 0
            }
            continue

        layer_statements = sum(f['summary']['num_statements'] for f in files)
        layer_covered = sum(f['summary']['covered_lines'] for f in files)

        layer_stats[layer_name] = {
            'coverage_percent': round((layer_covered / layer_statements * 100) if layer_statements > 0 else 0, 2),
            'total_statements': layer_statements,
            'covered_statements': layer_covered,
            'files_count': len(files)
        }

        total_statements += layer_statements
        total_covered += layer_covered

    # 计算总体统计
    overall_coverage = round((total_covered / total_statements * 100)
                             if total_statements > 0 else 0, 2)

    return {
        'summary': {
            'total_layers': len(layers),
            'analyzed_layers': len([l for l in layer_stats.values() if l['total_statements'] > 0]),
            'total_statements': total_statements,
            'covered_statements': total_covered,
            'overall_coverage': overall_coverage
        },
        'layers': layer_stats
    }


def generate_markdown_report(stats: Dict):
    """生成Markdown格式的报告"""
    summary = stats['summary']
    layers = stats['layers']

    report = "# 📊 RQA2025 分层覆盖率验证报告\n\n"

    # 总体统计
    report += "## 🎯 总体统计\n\n"
    report += "| 指标 | 值 |\n"
    report += "|------|-----|\n"
    report += f"| 总层级数 | {summary['total_layers']} |\n"
    report += f"| 已分析层级 | {summary['analyzed_layers']} |\n"
    report += f"| 总语句数 | {summary['total_statements']:,} |\n"
    report += f"| 已覆盖语句 | {summary['covered_statements']:,} |\n"
    report += ".2f" + " |\n"
    report += "\n\n"

    # 分层详情
    report += "## 🏆 分层覆盖率详情\n\n"
    report += "| 层级 | 覆盖率 | 语句数 | 覆盖语句 | 文件数 |\n"
    report += "|------|--------|--------|----------|--------|\n"

    # 按覆盖率排序
    sorted_layers = sorted(layers.items(), key=lambda x: x[1]['coverage_percent'], reverse=True)

    for layer_name, data in sorted_layers:
        report += "8.2f" + f"| {data['files_count']} |\n"

    report += "\n\n"

    # 覆盖率分布
    report += "## 📈 覆盖率分布分析\n\n"
    report += "```json\n"
    report += json.dumps({
        "coverage_ranges": {
            "high": [l for l, d in layers.items() if d['coverage_percent'] >= 80],
            "medium": [l for l, d in layers.items() if 50 <= d['coverage_percent'] < 80],
            "low": [l for l, d in layers.items() if d['coverage_percent'] < 50]
        },
        "top_performers": [
            {"layer": layer, "coverage": data['coverage_percent']}
            for layer, data in sorted_layers[:3]
        ],
        "needs_attention": [
            {"layer": layer, "coverage": data['coverage_percent']}
            for layer, data in sorted_layers[-3:]
            if data['coverage_percent'] < 50
        ]
    }, indent=2, ensure_ascii=False)
    report += "\n```\n\n"

    report += "## 📋 验证完成时间\n\n"
    report += f"- 生成时间: {Path.cwd()}\n"
    report += f"- 覆盖率工具: pytest-cov\n\n"

    return report


def update_plan_file(report_content: str):
    """更新TEST_COVERAGE_IMPROVEMENT_PLAN.md文件"""
    plan_file = "TEST_COVERAGE_IMPROVEMENT_PLAN.md"

    if not os.path.exists(plan_file):
        print(f"⚠️  未找到 {plan_file} 文件")
        return

    try:
        with open(plan_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找覆盖率验证部分
        if "## 📊 **覆盖率验证报告**" in content:
            # 替换现有部分
            parts = content.split("## 📊 **覆盖率验证报告**")
            new_content = parts[0] + report_content
            if len(parts) > 1:
                # 保留后面的内容
                remaining_parts = parts[1].split("\n## ", 1)
                if len(remaining_parts) > 1:
                    new_content += "\n## " + remaining_parts[1]
        else:
            # 添加新部分
            new_content = content + "\n\n" + report_content

        # 保存更新后的文件
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 已更新 {plan_file}")

    except Exception as e:
        print(f"❌ 更新 {plan_file} 失败: {str(e)}")


def main():
    """主函数"""
    print("🚀 开始生成分层覆盖率报告...")

    try:
        # 分析覆盖率数据
        stats = analyze_coverage_by_layer()

        # 生成Markdown报告
        report_content = generate_markdown_report(stats)

        # 更新计划文件
        update_plan_file(report_content)

        # 打印总结
        print("\n" + "="*60)
        print("🎯 分层覆盖率分析完成")
        print("="*60)
        print(f"总覆盖率: {stats['summary']['overall_coverage']}%")
        print(f"分析层级: {stats['summary']['analyzed_layers']}/{stats['summary']['total_layers']}")
        print(f"总语句数: {stats['summary']['total_statements']:,}")
        print("="*60)

    except Exception as e:
        print(f"❌ 生成报告失败: {str(e)}")


if __name__ == "__main__":
    main()
