#!/usr/bin/env python3
import json


def check_remaining_params():
    """检查剩余的长参数列表问题"""
    with open('phase3_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🎯 Phase 3剩余的长参数列表问题:')
    print('=' * 50)
    param_count = 0
    for opp in data['opportunities']:
        if '长参数列表' in opp['title']:
            param_count += 1
            print(f'{param_count}. {opp["title"]}')
            print(f'   文件: {opp["file_path"]}')
            print(f'   行号: {opp["line_number"]}')
            print(f'   建议: {opp["suggested_fix"]}')
            print()

    print(f'总共发现 {param_count} 个长参数列表问题需要解决')

    # 按文件分组统计
    file_stats = {}
    for opp in data['opportunities']:
        if '长参数列表' in opp['title']:
            file_path = opp['file_path']
            file_stats[file_path] = file_stats.get(file_path, 0) + 1

    print('\n📊 按文件分组统计:')
    for file_path, count in sorted(file_stats.items(), key=lambda x: x[1], reverse=True):
        print(f'  {file_path}: {count}个问题')


if __name__ == "__main__":
    check_remaining_params()
