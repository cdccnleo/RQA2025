#!/usr/bin/env python3
"""检查数据采集器工作节点和监控仪表板数据"""
import requests
import json

print('=== 数据采集器工作节点和监控仪表板检查 ===')

# 1. 检查工作节点注册状态
print('\n1. 工作节点注册状态')
print('-' * 60)
try:
    from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
    
    registry = get_unified_worker_registry()
    
    # 获取所有类型的工作节点
    print("工作节点统计:")
    for worker_type in WorkerType:
        workers = registry.get_workers_by_type(worker_type)
        print(f"  {worker_type.value}: {len(workers)} 个")
        
        # 显示前3个工作节点详情
        for i, (worker_id, info) in enumerate(list(workers.items())[:3]):
            print(f"    - {worker_id}: {info.get('status', '未知')}")
    
    # 特别关注数据采集器
    data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
    print(f"\n数据采集器详情:")
    if data_collectors:
        for worker_id, info in data_collectors.items():
            print(f"  - {worker_id}")
            print(f"    状态: {info.get('status', '未知')}")
            print(f"    能力: {info.get('capabilities', [])}")
            print(f"    注册时间: {info.get('registered_at', '未知')}")
    else:
        print("  ⚠️ 没有注册的数据采集器工作节点！")
        print("  建议: 启动数据采集器工作进程")
        
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 检查调度器队列状态
print('\n2. 调度器队列状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    stats = scheduler.get_statistics()
    
    print(f"调度器运行: {stats.get('is_running', False)}")
    print(f"总任务数: {stats.get('total_tasks', 0)}")
    print(f"待处理任务: {stats.get('pending_tasks', 0)}")
    print(f"运行中任务: {stats.get('running_tasks', 0)}")
    print(f"已完成任务: {stats.get('completed_tasks', 0)}")
    print(f"失败任务: {stats.get('failed_tasks', 0)}")
    
    # 检查队列大小
    if hasattr(scheduler, '_task_manager'):
        task_manager = scheduler._task_manager
        if hasattr(task_manager, '_tasks'):
            tasks = task_manager._tasks
            print(f"\n任务队列中的任务: {len(tasks)}")
            for task_id, task in list(tasks.items())[:5]:
                print(f"  - {task_id}: {task.get('status', '未知')}")
                
except Exception as e:
    print(f"检查失败: {e}")

# 3. 检查监控仪表板API
print('\n3. 监控仪表板API检查')
print('-' * 60)

# 3.1 检查调度器仪表盘
print("\n3.1 调度器仪表盘API:")
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
    data = response.json()
    
    scheduler_info = data.get('scheduler', {})
    unified_info = data.get('unified_scheduler', {})
    
    print(f"  调度器运行: {scheduler_info.get('running', False)}")
    print(f"  活跃数据源: {scheduler_info.get('active_sources', 0)}")
    print(f"  总数据源: {scheduler_info.get('total_sources', 0)}")
    print(f"  活跃任务: {scheduler_info.get('active_tasks', 0)}")
    print(f"  统一调度器运行: {unified_info.get('is_running', False)}")
    print(f"  总任务数: {unified_info.get('total_tasks', 0)}")
    print(f"  待处理任务: {unified_info.get('pending_tasks', 0)}")
    print(f"  运行中任务: {unified_info.get('running_tasks', 0)}")
    print(f"  数据采集器数量: {unified_info.get('data_collectors_count', 0)}")
    
except Exception as e:
    print(f"  获取失败: {e}")

# 3.2 检查自动采集状态
print("\n3.2 自动采集状态API:")
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"  运行中: {auto_data.get('running', False)}")
        print(f"  总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"  已提交任务: {auto_data.get('tasks_submitted', 0)}")
        print(f"  已检查数据源: {auto_data.get('sources_checked', 0)}")
        print(f"  待处理任务: {auto_data.get('pending_tasks_count', 0)}")
    else:
        print(f"  获取失败: {data.get('detail', '未知错误')}")
except Exception as e:
    print(f"  获取失败: {e}")

# 3.3 检查数据源列表
print("\n3.3 数据源列表API:")
try:
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    print(f"  总数据源: {len(sources)}")
    
    active = [s for s in sources if s.get('is_active', False)]
    inactive = [s for s in sources if not s.get('is_active', False)]
    
    print(f"  活跃数据源: {len(active)}")
    print(f"  非活跃数据源: {len(inactive)}")
    
    if active:
        print(f"\n  活跃数据源详情:")
        for source in active[:3]:
            print(f"    - {source.get('name', 'N/A')}: {source.get('status', '未知')}")
    
except Exception as e:
    print(f"  获取失败: {e}")

# 4. 检查监控仪表板HTML文件
print('\n4. 监控仪表板HTML文件检查')
print('-' * 60)

import os
html_paths = [
    '/app/web-static/data-collection-monitor.html',
    '/app/web-static/monitoring/data-collection-monitor.html',
    'web-static/data-collection-monitor.html',
]

for path in html_paths:
    if os.path.exists(path):
        print(f"✅ 找到HTML文件: {path}")
        
        # 检查文件内容
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键API调用
        if 'api/v1/data/scheduler' in content:
            print("  ✅ 包含调度器API调用")
        else:
            print("  ⚠️ 可能缺少调度器API调用")
            
        if 'worker' in content.lower() or '工作节点' in content:
            print("  ✅ 包含工作节点显示")
        else:
            print("  ⚠️ 可能缺少工作节点显示")
            
        if 'queue' in content.lower() or '队列' in content:
            print("  ✅ 包含队列显示")
        else:
            print("  ⚠️ 可能缺少队列显示")
            
        break
else:
    print("❌ 未找到监控仪表板HTML文件")

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
