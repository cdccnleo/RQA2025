#!/usr/bin/env python3
"""检查运行中的应用调度器状态"""
import requests

print('=== 检查运行中的应用调度器状态 ===')

# 1. 检查调度器仪表盘
print('\n1. 调度器仪表盘')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    scheduler_info = data.get('scheduler', {})
    unified_info = data.get('unified_scheduler', {})
    
    print(f"调度器运行: {scheduler_info.get('running', False)}")
    print(f"活跃数据源: {scheduler_info.get('active_sources', 0)}")
    print(f"活跃任务: {scheduler_info.get('active_tasks', 0)}")
    print(f"统一调度器运行: {unified_info.get('is_running', False)}")
    print(f"总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"已完成任务: {unified_info.get('completed_tasks', 0)}")
    print(f"数据采集器数量: {unified_info.get('data_collectors_count', 0)}")
    
except Exception as e:
    print(f"获取失败: {e}")

# 2. 检查自动采集状态
print('\n2. 自动采集状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"自动采集运行中: {auto_data.get('running', False)}")
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
        print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
        print(f"待处理任务: {auto_data.get('pending_tasks_count', 0)}")
    else:
        print(f"获取失败: {data.get('detail', '未知错误')}")
        
except Exception as e:
    print(f"获取失败: {e}")

# 3. 检查baostock数据源
print('\n3. baostock数据源状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    for source in sources.get('data', []):
        if 'baostock' in source.get('id', '').lower():
            print(f"数据源ID: {source.get('id')}")
            print(f"启用: {source.get('is_active', False)}")
            print(f"状态: {source.get('status', '未知')}")
            print(f"最后采集时间: {source.get('last_collection_time') or '从未采集'}")
            break
    else:
        print("未找到baostock数据源")
        
except Exception as e:
    print(f"获取失败: {e}")

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
