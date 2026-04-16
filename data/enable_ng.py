#!/usr/bin/env python3
import json, os

config_path = os.path.join(os.path.dirname(__file__), 'data', 'data_sources_config.json')
config_path = r"C:\PythonProject\RQA2025\data\data_sources_config.json"

with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse carefully
data = json.loads(content)
print(f"Top keys: {list(data.keys())}")

# Navigate to sources
sources = data.get('data_sources', [])
print(f"Sources count: {len(sources)}")

# Find natural gas
for s in sources:
    sid = s.get('id', '')
    if 'natural' in sid.lower():
        print(f"Found: {sid}")
        print(f"  enabled: {s.get('enabled')}")
        print(f"  akshare_function: {s.get('config', {}).get('akshare_function')}")
        s['enabled'] = True
        s['status'] = 'healthy'
        print(f"  Updated: enabled=True")

# Save
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved!")

# Now reload in the running container
import subprocess
result = subprocess.run(
    ['curl', '-s', '-X', 'POST', 'http://localhost:8000/api/v1/data/sources/reload'],
    capture_output=True, text=True, timeout=10
)
print(f"Reload API: {result.stdout} {result.stderr}")
