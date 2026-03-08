#!/usr/bin/env python3
"""验证自动采集状态"""
import requests
import time

print('=== 自动采集状态验证 ===')

# 等待几秒让采集开始
time.sleep(3)

# 检查自动采集状态
response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
data = response.json()

print('自动采集运行:', data['data']['running'])
print('检查间隔:', data['data']['check_interval'], '秒')
print('总检查次数:', data['data']['total_checks'])
print('已提交任务:', data['data']['tasks_submitted'])
print('已检查数据源:', data['data']['sources_checked'])

# 检查调度器任务
response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
data = response.json()

print('\n调度器状态:')
print('  running:', data['scheduler']['running'])
print('  active_tasks:', data['scheduler']['active_tasks'])
print('  total_tasks:', data['unified_scheduler']['total_tasks'])
print('  pending_tasks:', data['unified_scheduler']['pending_tasks'])
print('  running_tasks:', data['unified_scheduler']['running_tasks'])
