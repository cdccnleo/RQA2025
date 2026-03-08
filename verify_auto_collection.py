#!/usr/bin/env python3
"""验证自动采集服务状态和完整数据采集流程"""
import requests
import time

print('=== 验证自动采集服务和数据采集流程 ===')

# 1. 检查自动采集服务状态
print('\n1. 检查自动采集服务状态')
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
        print(f"上次检查: {auto_data.get('last_check_time') or '无'}")
        print(f"下次检查: {auto_data.get('next_check_time') or '无'}")
        
        is_running = auto_data.get('running', False)
        if is_running:
            print("\n✅ 自动采集服务已在运行！")
        else:
            print("\n⚠️ 自动采集服务未运行，需要手动启动")
    else:
        print(f"获取状态失败: {data.get('detail', '未知错误')}")
        is_running = False
except Exception as e:
    print(f"获取状态失败: {e}")
    is_running = False

# 2. 如果未运行，启动自动采集
if not is_running:
    print('\n2. 启动自动采集服务')
    print('-' * 60)
    try:
        response = requests.post('http://localhost:8000/api/v1/data/scheduler/auto-collection/start', timeout=10)
        data = response.json()
        print(f"启动结果: {data.get('message', '未知')}")
        
        # 等待启动
        time.sleep(3)
        
        # 再次检查状态
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
        data = response.json()
        
        if data.get('success') and data.get('data', {}).get('running'):
            print("✅ 自动采集服务启动成功！")
            is_running = True
        else:
            print("❌ 自动采集服务启动失败")
    except Exception as e:
        print(f"启动失败: {e}")

# 3. 检查调度器仪表盘
print('\n3. 检查调度器仪表盘')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    scheduler_info = data.get('scheduler', {})
    unified_info = data.get('unified_scheduler', {})
    
    print(f"调度器运行: {scheduler_info.get('running', False)}")
    print(f"活跃数据源: {scheduler_info.get('active_sources', 0)}")
    print(f"总数据源: {scheduler_info.get('total_sources', 0)}")
    print(f"活跃任务: {scheduler_info.get('active_tasks', 0)}")
    print(f"统一调度器运行: {unified_info.get('is_running', False)}")
    print(f"总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"已完成任务: {unified_info.get('completed_tasks', 0)}")
    print(f"数据采集器数量: {unified_info.get('data_collectors_count', 0)}")
except Exception as e:
    print(f"获取失败: {e}")

# 4. 等待一段时间后检查任务执行
if is_running:
    print('\n4. 等待任务执行...')
    print('-' * 60)
    print("等待30秒让自动采集执行任务...")
    time.sleep(30)
    
    print('\n5. 检查任务执行结果')
    print('-' * 60)
    try:
        # 检查自动采集状态
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
        
        if unified_info.get('completed_tasks', 0) > 0:
            print("\n✅ 任务已成功执行！")
        elif unified_info.get('pending_tasks', 0) > 0 or auto_data.get('tasks_submitted', 0) > 0:
            print("\n⏳ 任务已提交，正在执行中...")
        else:
            print("\n⚠️ 暂无任务执行记录")
            
    except Exception as e:
        print(f"检查失败: {e}")

print('\n' + '=' * 60)
print('验证完成')
print('=' * 60)
