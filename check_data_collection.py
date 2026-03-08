#!/usr/bin/env python3
"""检查自动数据采集状态"""
import requests
import time

print('=== 检查自动数据采集状态 ===')

# 等待应用启动
time.sleep(10)

# 1. 检查自动采集服务状态
print('\n1. 自动采集服务状态')
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
        
        is_running = auto_data.get('running', False)
        if is_running:
            print("\n✅ 自动采集服务运行正常！")
        else:
            print("\n❌ 自动采集服务未运行")
    else:
        print(f"获取状态失败: {data.get('detail', '未知错误')}")
        is_running = False
except Exception as e:
    print(f"获取失败: {e}")
    is_running = False

# 2. 检查调度器状态
print('\n2. 调度器状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    scheduler_info = data.get('scheduler', {})
    unified_info = data.get('unified_scheduler', {})
    
    print(f"调度器运行: {scheduler_info.get('running', False)}")
    print(f"活跃数据源: {scheduler_info.get('active_sources', 0)}")
    print(f"数据采集器数量: {unified_info.get('data_collectors_count', 0)}")
    print(f"总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"已完成任务: {unified_info.get('completed_tasks', 0)}")
except Exception as e:
    print(f"获取失败: {e}")

# 3. 如果自动采集在运行，等待并检查任务执行
if is_running:
    print('\n3. 等待任务执行（30秒）...')
    print('-' * 60)
    time.sleep(30)
    
    print('\n4. 任务执行结果')
    print('-' * 60)
    try:
        # 再次检查自动采集状态
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
        data = response.json()
        
        if data.get('success'):
            auto_data = data.get('data', {})
            print(f"总检查次数: {auto_data.get('total_checks', 0)}")
            print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
            print(f"待处理任务: {auto_data.get('pending_tasks_count', 0)}")
        
        # 检查调度器状态
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
        data = response.json()
        
        unified_info = data.get('unified_scheduler', {})
        print(f"\n调度器任务统计:")
        print(f"  总任务数: {unified_info.get('total_tasks', 0)}")
        print(f"  待处理任务: {unified_info.get('pending_tasks', 0)}")
        print(f"  运行中任务: {unified_info.get('running_tasks', 0)}")
        print(f"  已完成任务: {unified_info.get('completed_tasks', 0)}")
        
        if auto_data.get('tasks_submitted', 0) > 0:
            print("\n✅ 自动数据采集正常工作！任务已提交")
        else:
            print("\n⚠️ 暂无任务提交")
            
    except Exception as e:
        print(f"检查失败: {e}")

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
