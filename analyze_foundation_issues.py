#!/usr/bin/env python3
"""
分析foundation模块的具体重构机会
"""

import json
from collections import defaultdict

def analyze_foundation_issues():
    """分析foundation模块的问题"""

    with open('submodule_foundation_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('Foundation模块重构机会汇总:')
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
        else:
            type_count['其他'] += 1

    print('\n按问题类型统计:')
    for problem_type, count in sorted(type_count.items()):
        print(f'  {problem_type}: {count}')

    # 显示前15个重构机会
    print('\n前15个重构机会:')
    for i, opp in enumerate(data['opportunities'][:15]):
        print(f'{i+1:2d}. {opp["title"]} - {opp["severity"]} - {opp["file_path"]}:{opp["line_number"]}')

if __name__ == "__main__":
    analyze_foundation_issues()
