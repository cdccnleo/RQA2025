#!/usr/bin/env python3
"""检查调度器状态"""
import requests
import json

# 检查数据源列表
response = requests.get('http://localhost:8000/api/v1/data/sources')
print('=== 数据源列表 ===')
data = response.json()
if data.get('success'):
    sources = data.get('data', [])
    print(f"总数据源数量: {len(sources)}")
    for source in sources[:10]:
        name = source.get('source_name')
        stype = source.get('source_type')
        enabled = source.get('enabled')
        last_time = source.get('last_collection_time')
        print(f"- {name}: {stype} | enabled={enabled} | last={last_time}")
else:
    print(json.dumps(data, indent=2, ensure_ascii=False))
