#!/usr/bin/env python3
"""验证数据采集修复效果"""
import requests
import time

print('=== 验证数据采集修复效果 ===')

# 等待应用启动
time.sleep(15)

# 1. 检查自动采集状态
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
    else:
        print(f"获取失败: {data.get('detail', '未知错误')}")
except Exception as e:
    print(f"获取失败: {e}")

# 2. 检查调度器状态
print('\n2. 调度器状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    unified_info = data.get('unified_scheduler', {})
    print(f"统一调度器运行: {unified_info.get('is_running', False)}")
    print(f"总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"已完成任务: {unified_info.get('completed_tasks', 0)}")
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
except Exception as e:
    print(f"获取失败: {e}")

# 4. 等待任务执行
print('\n4. 等待任务执行（60秒）...')
print('-' * 60)
time.sleep(60)

# 5. 再次检查状态
print('\n5. 任务执行后状态')
print('-' * 60)
try:
    # 检查自动采集状态
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
    
    # 检查调度器状态
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    unified_info = data.get('unified_scheduler', {})
    print(f"\n调度器任务统计:")
    print(f"  总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"  待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"  运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"  已完成任务: {unified_info.get('completed_tasks', 0)}")
    
    # 检查baostock数据源
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    for source in sources.get('data', []):
        if 'baostock' in source.get('id', '').lower():
            print(f"\nbaostock数据源:")
            print(f"  最后采集时间: {source.get('last_collection_time') or '从未采集'}")
            
            if source.get('last_collection_time'):
                print("\n🎉 修复成功！baostock数据采集完成！")
            else:
                print("\n⏳ 采集尚未完成，可能需要更多时间")
            break
    
except Exception as e:
    print(f"检查失败: {e}")

print('\n' + '=' * 60)
print('验证完成')
print('=' * 60)
