#!/usr/bin/env python3
"""验证调度器和自动采集状态"""
import requests

print('=== 调度器和自动采集状态验证 ===')

# 检查调度器状态
response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
data = response.json()

print('调度器状态:')
print('  running:', data['scheduler']['running'])
print('  uptime:', data['scheduler']['uptime'])
print('  is_running:', data['unified_scheduler']['is_running'])

# 检查自动采集状态
response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
data = response.json()

print('\n自动采集状态:')
print('  success:', data['success'])
if data['success']:
    print('  running:', data['data']['running'])
    print('  check_interval:', data['data']['check_interval'])
    print('  total_checks:', data['data']['total_checks'])
    print('  sources_checked:', data['data']['sources_checked'])
