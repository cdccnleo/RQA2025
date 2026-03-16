#!/usr/bin/env python3
import json

with open('/app/data/data_sources_config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('数据源列表:')
print(f"数据类型: {type(data)}")
if isinstance(data, list):
    for s in data:
        print(f"  - 键: {list(s.keys())}")
        if 'source_id' in s:
            print(f"    source_id: {s['source_id']}")
        elif 'id' in s:
            print(f"    id: {s['id']}")
        else:
            print(f"    内容: {s}")
elif isinstance(data, dict):
    for key in data:
        print(f"  - {key}")
