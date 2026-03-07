#!/usr/bin/env python3
import json


def check_monitoring_params():
    """查看monitoring_alert_system.py中的长参数问题"""
    with open('phase5_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('📋 monitoring_alert_system.py 中的长参数问题:')
    print('=' * 60)

    count = 0
    for opp in data['opportunities']:
        if '长参数列表' in opp['title'] and 'monitoring_alert_system.py' in opp['file_path']:
            count += 1
            print(f'{count}. {opp["title"]}')
            print(f'   行号: {opp["line_number"]}')
            print(f'   建议: {opp["suggested_fix"]}')
            print()

    print(f'共发现 {count} 个问题')


if __name__ == "__main__":
    check_monitoring_params()
