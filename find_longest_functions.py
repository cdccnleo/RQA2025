#!/usr/bin/env python3
import json


def find_longest_functions():
    """查找最长的函数"""
    with open('phase2_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 查找最长的函数
    long_functions = []
    for opp in data['opportunities']:
        if 'long_function' in opp['title']:
            # 从描述中提取行数
            desc = opp['description']
            if '(' in desc and '行' in desc:
                try:
                    lines = int(desc.split('(')[1].split('行')[0].strip())
                    long_functions.append((lines, opp))
                except:
                    pass

    # 按行数排序
    long_functions.sort(key=lambda x: x[0], reverse=True)

    print('🎯 Phase 3 需要重构的最长函数:')
    print('=' * 50)
    for i, (lines, opp) in enumerate(long_functions[:10]):
        print(f'{i+1}. {opp["title"]} - {lines}行')
        print(f'   文件: {opp["file_path"]}')
        print(f'   建议: {opp["suggested_fix"]}')
        print()


if __name__ == "__main__":
    find_longest_functions()
