#!/usr/bin/env python3
"""验证修复效果"""
import requests
import time

print('=== 验证修复效果 ===')

# 等待应用完全启动
time.sleep(5)

# 1. 检查调度器仪表盘（验证工作节点注册）
print('\n1. 检查工作节点注册')
print('-' * 50)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    scheduler_info = data.get('scheduler', {})
    unified_info = data.get('unified_scheduler', {})
    
    print(f"调度器运行: {scheduler_info.get('running', False)}")
    print(f"活跃数据源: {scheduler_info.get('active_sources', 0)}")
    print(f"数据采集器数量: {unified_info.get('data_collectors_count', 0)}")
    print(f"总任务数: {unified_info.get('total_tasks', 0)}")
    
    if unified_info.get('data_collectors_count', 0) > 0:
        print("✅ 工作节点注册成功！")
    else:
        print("⚠️ 工作节点数量为0")
        
except Exception as e:
    print(f"获取失败: {e}")

# 2. 启动自动采集
print('\n2. 启动自动采集')
print('-' * 50)
try:
    response = requests.post('http://localhost:8000/api/v1/data/scheduler/auto-collection/start', timeout=10)
    data = response.json()
    
    print(f"启动结果: {data.get('message', '未知')}")
    
except Exception as e:
    print(f"启动失败: {e}")

# 3. 等待一段时间后检查任务状态
print('\n3. 等待任务执行...')
print('-' * 50)
time.sleep(10)

try:
    # 检查自动采集状态
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"自动采集运行: {auto_data.get('running', False)}")
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
        print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
    
    # 检查调度器任务
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    unified_info = data.get('unified_scheduler', {})
    print(f"\n调度器任务统计:")
    print(f"  总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"  待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"  运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"  已完成任务: {unified_info.get('completed_tasks', 0)}")
    
    if unified_info.get('total_tasks', 0) > 0:
        print("\n✅ 任务已成功提交到调度器！")
    else:
        print("\n⚠️ 调度器中暂无任务")
        
except Exception as e:
    print(f"检查失败: {e}")

print('\n' + '=' * 50)
print('验证完成')
print('=' * 50)
