#!/usr/bin/env python3
import subprocess, json

# Find natural gas in the array using a loop
for i in range(22):
    result = subprocess.run(
        ['docker', 'exec', '-t', 'rqa2025-postgres', 'psql', '-U', 'rqa2025_admin', '-d', 'rqa2025_prod', '-t', '-c',
         f"SELECT config_data->'data_sources'->>{i}->>'id' FROM data_source_configs WHERE config_key='data_sources_production';"],
        capture_output=True, timeout=30, encoding='utf-8', errors='replace'
    )
    sid = result.stdout.strip()
    if 'natural' in sid.lower():
        print(f"Found natural gas at index {i}: {sid}")
        # Get full config
        result2 = subprocess.run(
            ['docker', 'exec', '-t', 'rqa2025-postgres', 'psql', '-U', 'rqa2025_admin', '-d', 'rqa2025_prod', '-t', '-c',
             f"SELECT config_data->'data_sources'->>{i} FROM data_source_configs WHERE config_key='data_sources_production';"],
            capture_output=True, timeout=30, encoding='utf-8', errors='replace'
        )
        print(f"Full config: {result2.stdout.strip()[:300]}")
        break
