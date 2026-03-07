#!/usr/bin/env python3
"""
修复财联社ID的脚本
"""

import json
import os

def fix_cls_id():
    config_file = 'data/data_sources_config.json'

    # 确保配置文件存在
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return

    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('修复前:')
    for source in data['data_sources']:
        if source['name'] == '财联社':
            print(f"财联社ID: {repr(source['id'])}")

    # 修复财联社的ID
    fixed = False
    for source in data['data_sources']:
        if source['name'] == '财联社':
            if source['id'] is None or str(source['id']).lower() in ['null', 'none']:
                source['id'] = 'cls'
                print(f"已修复财联社ID为: {source['id']}")
                fixed = True

    if fixed:
        # 保存修复后的配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('配置文件已保存')
    else:
        print('财联社ID已经是正确的，无需修复')

if __name__ == "__main__":
    fix_cls_id()
