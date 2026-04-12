#!/usr/bin/env python3
import json
import os

config_path = '/app/data/data_sources_config.json'
output_path = '/app/data/data_sources_config.json'

# Read current config
with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

data = json.loads(content)
sources = data if isinstance(data, list) else data.get('data_sources', data)

print(f'Found {len(sources)} data sources')

# Update rate limits
updated = 0
for s in sources:
    old = s.get('rate_limit', 'N/A')
    if old != '1次/天':
        s['rate_limit'] = '1次/天'
        print(f'Updated {s["id"]}: {old} -> 1次/天')
        updated += 1

print(f'Total updated: {updated}')

# Save
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('Config saved')
