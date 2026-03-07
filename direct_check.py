#!/usr/bin/env python3
"""
直接检查数据源配置，绕过所有缓存
"""

import json
import os

def direct_check():
    """直接检查所有数据源配置文件"""
    print("🔍 直接检查数据源配置")
    print("=" * 40)

    # 检查所有可能的文件
    files_to_check = [
        "data/data_sources_config.json",
        "src/data/data_sources_config.json",
        "data/production/data_sources_config.json"
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n📄 检查文件: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    sources = data
                    print("  格式: 列表")
                elif isinstance(data, dict):
                    sources = data.get('data_sources', [])
                    print("  格式: 字典")
                else:
                    print("  格式: 未知")
                    continue

                print(f"  数据源数量: {len(sources)}")

                # 详细列出每个数据源
                for i, source in enumerate(sources):
                    if isinstance(source, dict):
                        name = source.get('name', 'Unknown')
                        id_val = source.get('id')
                        print(f"    {i}: {name} (ID: {repr(id_val)})")

                        if name == '财联社':
                            print(f"       ⚠️  发现财联社数据源!")
                            print(f"           类型: {repr(source.get('type'))}")
                            print(f"           URL: {repr(source.get('url'))}")
                            print(f"           启用: {source.get('enabled')}")
                    else:
                        print(f"    {i}: 无效数据源格式")

                # 统计财联社
                cls_count = sum(1 for s in sources if isinstance(s, dict) and s.get('name') == '财联社')
                if cls_count > 0:
                    print(f"  ❌ 财联社数据源: {cls_count} 个")
                else:
                    print("  ✅ 无财联社数据源")

            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        else:
            print(f"\n📄 文件不存在: {file_path}")

    print(f"\n{'='*40}")
    print("检查完成")

if __name__ == "__main__":
    direct_check()