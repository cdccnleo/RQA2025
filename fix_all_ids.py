#!/usr/bin/env python3
"""
修复所有数据源的null ID
"""

import json

def fix_all_ids():
    config_file = 'data/data_sources_config.json'

    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('修复前:')
    for source in data['data_sources']:
        print(f"{source['name']}: id={repr(source['id'])}")

    # 修复所有null ID
    for source in data['data_sources']:
        if source['id'] is None or str(source['id']).lower() in ['null', 'none']:
            name = source['name']
            if '新浪财经' in name:
                source['id'] = 'sinafinance'
            elif '宏观经济' in name:
                source['id'] = 'macrodata'
            elif '财联社' in name:
                source['id'] = 'cls'
            else:
                source['id'] = name.lower().replace(' ', '_')
            print(f"修复 {name}: {source['id']}")

    # 保存修复后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print('配置文件已保存')

if __name__ == "__main__":
    fix_all_ids()
