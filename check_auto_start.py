#!/usr/bin/env python3
"""验证自动采集服务自动启动"""
import requests
import time

time.sleep(5)

print('=== 验证自动采集服务自动启动 ===')

response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
data = response.json()

if data.get('success'):
    auto_data = data.get('data', {})
    print(f"自动采集运行中: {auto_data.get('running', False)}")
    print(f"总检查次数: {auto_data.get('total_checks', 0)}")
    print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
    print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
    
    if auto_data.get('running', False):
        print("\n✅ 自动采集服务已成功自动启动！")
    else:
        print("\n❌ 自动采集服务未运行")
else:
    print(f"获取状态失败: {data.get('detail', '未知错误')}")
