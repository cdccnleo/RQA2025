#!/usr/bin/env python3
"""
分析core_optimization模块的高严重度问题
"""

import json

def analyze_high_severity():
    """分析高严重度问题"""

    with open('submodule_core_optimization_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('Core Optimization模块高严重度问题:')
    high_severity = [opp for opp in data['opportunities'] if opp.get('severity') == 'high']

    for i, opp in enumerate(high_severity):
        print(f'{i+1}. {opp["title"]} - {opp["file_path"]}:{opp["line_number"]}')
        print(f'   {opp["description"]}')

if __name__ == "__main__":
    analyze_high_severity()
