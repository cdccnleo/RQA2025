#!/usr/bin/env python3
"""
RQA2025 19层架构代码质量综合分析报告生成器
"""

import json
import glob


def analyze_all_reports():
    """分析所有架构层的AI智能代码分析报告"""

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

    for report_file in sorted(report_files):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            layer_name = report_file.replace(
                'architecture_layer_analysis_', '').replace('_final.json', '')
            layer_stats[layer_name] = data

            if 'summary' in data:
                summary = data['summary']
                total_stats['files'] += summary.get('total_files', 0)
                total_stats['lines'] += summary.get('total_lines', 0)
                total_stats['complexity'] += summary.get('total_complexity', 0)
                total_stats['maintainability'] += summary.get('avg_maintainability', 0)

                issues = summary.get('issues', {})
                total_stats['high_issues'] += issues.get('HIGH', 0)
                total_stats['medium_issues'] += issues.get('MEDIUM', 0)
                total_stats['low_issues'] += issues.get('LOW', 0)
                total_stats['total_issues'] += issues.get('total', 0)

        except Exception as e:
            print(f'读取 {report_file} 时出错: {e}')

    # 计算平均值
    total_stats['avg_maintainability'] = total_stats['maintainability'] / \
        len(report_files) if report_files else 0

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
    print('=== 分层问题统计 ===')
    for layer, data in sorted(layer_stats.items()):
        if 'summary' in data:
            summary = data['summary']
            issues = summary.get('issues', {})
            print(
                f'{layer}: MEDIUM={issues.get("MEDIUM", 0)}, LOW={issues.get("LOW", 0)}, 总计={issues.get("total", 0)}')

    return total_stats, layer_stats


if __name__ == '__main__':
    analyze_all_reports()
