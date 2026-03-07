#!/usr/bin/env python3
"""
直接测试load_data_sources函数
"""

import json
import os

DATA_SOURCES_CONFIG_FILE = "data/data_sources_config.json"

def load_data_sources():
    """加载数据源配置"""
    try:
        if os.path.exists(DATA_SOURCES_CONFIG_FILE):
            with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data_sources = data.get('data_sources', [])

            print(f"加载数据源配置，共有 {len(data_sources)} 个数据源")

            # 修复null ID
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')

                print(f"检查数据源: {name}, ID: {repr(id_value)} (type: {type(id_value).__name__})")

                # 强制修复任何形式的null ID
                needs_fix = (
                    id_value is None or
                    str(id_value).lower() in ['null', 'none', ''] or
                    id_value == 'null' or
                    id_value == 'None'
                )

                print(f"  needs_fix: {needs_fix}")

                if needs_fix:
                    print(f"  准备修复数据源: {name}")
                    if '新浪财经' in name:
                        source['id'] = 'sinafinance'
                        print("  匹配新浪财经规则")
                    elif '宏观经济' in name:
                        source['id'] = 'macrodata'
                        print("  匹配宏观经济规则")
                    elif '财联社' in name:
                        source['id'] = 'cls'
                        print("  匹配财联社规则")
                    else:
                        source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')
                        print(f"  使用默认规则: {source['id']}")
                    print(f"🔧 修复数据源 {name} 的null ID: {repr(id_value)} -> {source['id']}")
                else:
                    print(f"✅ 数据源 {name} ID正常: {repr(id_value)}")

            print(f"修复后数据源列表:")
            for source in data_sources:
                print(f"  {source.get('name')}: {repr(source.get('id'))}")

            return data_sources
        else:
            return []
    except Exception as e:
        print(f"加载数据源配置失败: {e}")
        return []

if __name__ == "__main__":
    sources = load_data_sources()
    print(f"\n返回 {len(sources)} 个数据源")
