#!/usr/bin/env python3
import json


def show_long_functions():
    """显示长函数重构机会"""
    with open('phase2_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🎯 长函数重构机会:')
    print('=' * 40)
    count = 0
    for opp in data['opportunities']:
        if '长函数重构' in opp['title']:
            count += 1
            print(f'{count}. {opp["title"]}')
            print(f'   文件: {opp["file_path"]}')
            print(f'   行号: {opp["line_number"]}')
            print(f'   描述: {opp["description"]}')
            print()


if __name__ == "__main__":
    show_long_functions()
