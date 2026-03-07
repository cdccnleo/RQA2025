#!/usr/bin/env python3
"""
分析剩余的重构机会
"""

import json
from collections import defaultdict

def analyze_remaining_issues():
    """分析剩余的重构机会"""

    with open('core_service_layer_final_refactored_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('=== 剩余重构机会分析 ===')
    print(f'总重构机会: {len(data["opportunities"])}')

    # 按严重程度统计
    severity_count = defaultdict(int)
    for opp in data['opportunities']:
        severity = opp.get('severity', 'unknown')
        severity_count[severity] += 1

    print('\n按严重程度统计:')
    for severity, count in sorted(severity_count.items()):
        print(f'  {severity}: {count}')

    # 按问题类型统计
    type_count = defaultdict(int)
    for opp in data['opportunities']:
        title = opp.get('title', '')
        if '长参数列表' in title:
            type_count['长参数列表'] += 1
        elif '长函数' in title:
            type_count['长函数'] += 1
        elif '大类' in title:
            type_count['大类'] += 1
        elif '复杂方法' in title:
            type_count['复杂方法'] += 1
        else:
            type_count['其他'] += 1

    print('\n按问题类型统计:')
    for problem_type, count in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
        print(f'  {problem_type}: {count}')

    # 按文件统计
    file_count = defaultdict(int)
    for opp in data['opportunities']:
        file_path = opp.get('file_path', '').replace('\\', '/')
        file_count[file_path] += 1

    print('\n按文件统计 (前10个):')
    for file_path, count in sorted(file_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {file_path}: {count}')

    # 高严重度问题详情
    high_severity = [opp for opp in data['opportunities'] if opp.get('severity') == 'high']
    print(f'\n高严重度问题 ({len(high_severity)}个):')
    for i, opp in enumerate(high_severity[:5]):  # 只显示前5个
        print(f'  {i+1}. {opp["title"]} - {opp["file_path"]}:{opp["line_number"]}')

    # 长参数列表问题详情
    long_params = [opp for opp in data['opportunities'] if '长参数列表' in opp.get('title', '')]
    print(f'\n长参数列表问题 ({len(long_params)}个):')
    for i, opp in enumerate(long_params[:10]):  # 只显示前10个
        print(f'  {i+1}. {opp["title"]} - {opp["file_path"]}:{opp["line_number"]}')

    # 生成优化建议
    print('\n=== 优化建议 ===')
    print('1. 优先处理长参数列表问题 (69个)')
    print('2. 处理复杂方法重构 (2个)')
    print('3. 处理长函数重构 (10个)')
    print('4. 检查剩余大类问题 (19个)')

    # 重点关注的文件
    print('\n重点关注的文件:')
    priority_files = [
        'src/core/foundation/exceptions/unified_exceptions.py',
        'src/core/core_optimization/optimizations/long_term_optimizations.py',
        'src/core/event_bus/core.py'
    ]

    for file_path in priority_files:
        if file_path in file_count:
            print(f'  • {file_path}: {file_count[file_path]} 个问题')

if __name__ == "__main__":
    analyze_remaining_issues()
