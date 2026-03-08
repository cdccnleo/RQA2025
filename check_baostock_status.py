#!/usr/bin/env python3
"""检查baostock_a股数据采集状态"""
import requests
import json

print('=== baostock_a股数据采集状态检查 ===')

# 1. 检查调度器任务列表
print('\n1. 调度器任务状态')
print('-' * 50)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    unified = data.get('unified_scheduler', {})
    print(f"总任务数: {unified.get('total_tasks', 0)}")
    print(f"待处理任务: {unified.get('pending_tasks', 0)}")
    print(f"运行中任务: {unified.get('running_tasks', 0)}")
    print(f"已完成任务: {unified.get('completed_tasks', 0)}")
    print(f"失败任务: {unified.get('failed_tasks', 0)}")
except Exception as e:
    print(f"获取失败: {e}")

# 2. 检查自动采集状态
print('\n2. 自动采集状态')
print('-' * 50)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"运行中: {auto_data.get('running', False)}")
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
        print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
        print(f"待处理任务: {auto_data.get('pending_tasks_count', 0)}")
except Exception as e:
    print(f"获取失败: {e}")

# 3. 检查baostock数据源详情
print('\n3. baostock_a股数据源详情')
print('-' * 50)
try:
    # 获取所有数据源
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    # 查找baostock数据源
    baostock_source = None
    for source in sources:
        if 'baostock' in source.get('id', '').lower() or 'baostock' in source.get('name', '').lower():
            baostock_source = source
            break
    
    if baostock_source:
        print(f"数据源ID: {baostock_source.get('id')}")
        print(f"数据源名称: {baostock_source.get('name')}")
        print(f"是否启用: {baostock_source.get('is_active', False)}")
        print(f"最后采集时间: {baostock_source.get('last_collection_time') or '从未采集'}")
        print(f"状态: {baostock_source.get('status', '未知')}")
    else:
        print("未找到baostock数据源")
        print(f"\n所有数据源:")
        for s in sources[:5]:
            print(f"  - {s.get('id')}: {s.get('name')} (启用: {s.get('is_active')})")
except Exception as e:
    print(f"获取失败: {e}")

# 4. 检查最近日志
print('\n4. 检查应用日志')
print('-' * 50)
print("查看最近的数据采集相关日志...")
