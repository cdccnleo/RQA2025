#!/usr/bin/env python3
"""
调试财联社ID修复问题
"""

import json
import os

def load_data_sources_debug():
    """调试版本的load_data_sources"""
    config_file = "data/data_sources_config.json"

    print(f"尝试加载配置文件: {config_file}")
    print(f"文件存在: {os.path.exists(config_file)}")

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            data_sources = config_data.get('data_sources', [])
            print(f"从文件加载了 {len(data_sources)} 个数据源")

            # 显示所有数据源
            for i, source in enumerate(data_sources):
                id_val = source.get('id')
                name = source.get('name', 'unknown')
                print(f"  {i}: {name} (ID: {repr(id_val)})")

            # 修复null ID
            print("\n开始修复null ID...")
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')

                needs_fix = (
                    id_value is None or
                    str(id_value).lower() in ['null', 'none', ''] or
                    id_value == 'null' or
                    id_value == 'None'
                )

                print(f"检查 {name}: ID={repr(id_value)}, needs_fix={needs_fix}")

                if needs_fix:
                    print(f"  发现需要修复: {name}")
                    if '新浪财经' in name:
                        source['id'] = 'sinafinance'
                        print("  修复为: sinafinance")
                    elif '宏观经济' in name:
                        source['id'] = 'macrodata'
                        print("  修复为: macrodata")
                    elif '财联社' in name:
                        source['id'] = 'cls'
                        print("  修复为: cls")
                    else:
                        source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')
                        print(f"  修复为: {source['id']}")

            print("\n修复后的数据源:")
            for i, source in enumerate(data_sources):
                id_val = source.get('id')
                name = source.get('name', 'unknown')
                print(f"  {i}: {name} (ID: {repr(id_val)})")

            return data_sources
        else:
            print("配置文件不存在")
            return []
    except Exception as e:
        print(f"加载数据源配置失败: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("=== 调试财联社ID修复 ===\n")
    sources = load_data_sources_debug()

    print("\n=== 最终结果 ===")
    for source in sources:
        if source['name'] == '财联社':
            print(f"财联社最终ID: {repr(source['id'])}")
            break
    else:
        print("未找到财联社数据源")
