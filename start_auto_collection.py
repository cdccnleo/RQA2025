#!/usr/bin/env python3
"""启动自动采集服务"""
import requests

print('=== 启动自动采集服务 ===')

# 启动自动采集
response = requests.post('http://localhost:8000/api/v1/data/scheduler/auto-collection/start', timeout=10)
print('状态码:', response.status_code)
print('响应:', response.json())

# 验证状态
print('\n=== 验证自动采集状态 ===')
response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
data = response.json()

print('自动采集运行:', data['data']['running'])
print('检查间隔:', data['data']['check_interval'], '秒')
print('总检查次数:', data['data']['total_checks'])
print('已检查数据源:', data['data']['sources_checked'])
