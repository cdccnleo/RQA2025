#!/usr/bin/env python3
import json


def find_long_param_functions():
    """查找长参数函数"""
    with open('phase1_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    long_param_functions = []
    for opp in data['opportunities']:
        if 'parameter' in opp['title'].lower() or '参数' in opp['title']:
            long_param_functions.append({
                'title': opp['title'],
                'file': opp['file_path'],
                'severity': opp['severity'],
                'risk_level': opp['risk_level']
            })

    print("🎯 发现的长参数函数问题:")
    print("=" * 60)
    for func in long_param_functions:
        print(f"📋 {func['title']}")
        print(f"   文件: {func['file']}")
        print(f"   严重程度: {func['severity']}")
        print(f"   风险等级: {func['risk_level']}")
        print("   ---")


if __name__ == "__main__":
    find_long_param_functions()
