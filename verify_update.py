#!/usr/bin/env python3
import json

with open('/app/data/data_sources_config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

source = [x for x in data if x['id']=='akshare_stock_a'][0]
print(f"source_id: {source['id']}")
print(f"last_test: {source['last_test']}")
print(f"status: {source['status']}")
