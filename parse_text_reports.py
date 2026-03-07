#!/usr/bin/env python3
"""
RQA2025 19层架构代码质量综合分析报告生成器 - 文本报告解析器
"""

import glob
import re
from typing import Dict, Any


def parse_text_report(file_path: str) -> Dict[str, Any]:
    """解析单个文本格式的AI智能代码分析报告"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取层级名称
    layer_match = re.search(r'🎯 AI智能化代码分析报告 - ([^\n]+)', content)
    layer_name = layer_match.group(1).replace('src/', '') if layer_match else file_path

    # 提取总体统计
    files_match = re.search(r'📁 文件数量: (\d+)', content)
    lines_match = re.search(r'📝 总代码行数: ([\d,]+)', content)
    complexity_match = re.search(r'🔄 总复杂度: ([\d.]+)', content)
    maintainability_match = re.search(r'📈 平均可维护性: ([\d.]+)', content)

    # 提取问题统计
    high_match = re.search(r'HIGH: (\d+)', content)
    medium_match = re.search(r'MEDIUM: (\d+)', content)
    low_match = re.search(r'LOW: (\d+)', content)

    return {
        'layer': layer_name,
        'files': int(files_match.group(1)) if files_match else 0,
        'lines': int(lines_match.group(1).replace(',', '')) if lines_match else 0,
        'complexity': float(complexity_match.group(1)) if complexity_match else 0.0,
        'maintainability': float(maintainability_match.group(1)) if maintainability_match else 0.0,
        'high_issues': int(high_match.group(1)) if high_match else 0,
        'medium_issues': int(medium_match.group(1)) if medium_match else 0,
        'low_issues': int(low_match.group(1)) if low_match else 0,
        'total_issues': 0  # 将在后面计算
    }


def analyze_all_text_reports():
    """分析所有文本格式的架构层AI智能代码分析报告"""

    # 获取所有分析报告文件
    report_files = glob.glob('architecture_layer_analysis_*_final.json')
    print(f'找到 {len(report_files)} 个分析报告文件')

    total_stats = {
        'files': 0,
        'lines': 0,
        'complexity': 0.0,
        'maintainability': 0.0,
        'high_issues': 0,
        'medium_issues': 0,
        'low_issues': 0,
        'total_issues': 0
    }

    layer_stats = {}

    valid_reports = 0
    for report_file in sorted(report_files):
        try:
            stats = parse_text_report(report_file)
            layer_stats[stats['layer']] = stats

            total_stats['files'] += stats['files']
            total_stats['lines'] += stats['lines']
            total_stats['complexity'] += stats['complexity']
            total_stats['maintainability'] += stats['maintainability']
            total_stats['high_issues'] += stats['high_issues']
            total_stats['medium_issues'] += stats['medium_issues']
            total_stats['low_issues'] += stats['low_issues']

            valid_reports += 1

        except Exception as e:
            print(f'解析 {report_file} 时出错: {e}')

    # 计算平均值和总数
    total_stats['avg_maintainability'] = total_stats['maintainability'] / \
        valid_reports if valid_reports else 0
    total_stats['total_issues'] = total_stats['high_issues'] + \
        total_stats['medium_issues'] + total_stats['low_issues']

    print('')
    print('=== RQA2025 19层架构代码质量综合分析报告 ===')
    print(f'📁 总文件数: {total_stats["files"]}')
    print(f'📝 总代码行数: {total_stats["lines"]:,}')
    print(f'🔄 总复杂度: {total_stats["complexity"]:.2f}')
    print(f'📈 平均可维护性: {total_stats["avg_maintainability"]:.2f}')
    print('')
    print('🚨 问题统计:')
    print(f'   HIGH: {total_stats["high_issues"]}')
    print(f'   MEDIUM: {total_stats["medium_issues"]}')
    print(f'   LOW: {total_stats["low_issues"]}')
    print(f'   总计: {total_stats["total_issues"]}')

    print('')
    print('=== 分层问题统计 (按严重程度排序) ===')
    # 按问题总数排序
    sorted_layers = sorted(layer_stats.items(),
                           key=lambda x: (x[1]['medium_issues'] + x[1]['low_issues']),
                           reverse=True)

    for layer, stats in sorted_layers:
        total_layer_issues = stats['medium_issues'] + stats['low_issues']
        if total_layer_issues > 0:
            print(
                f'{layer}: MEDIUM={stats["medium_issues"]}, LOW={stats["low_issues"]}, 总计={total_layer_issues}')

    return total_stats, layer_stats


if __name__ == '__main__':
    analyze_all_text_reports()
