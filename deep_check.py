#!/usr/bin/env python3
"""深度检查任务提交和执行流程"""
import requests
import json

print('=== 深度检查任务提交和执行流程 ===')

# 1. 检查任务提交详情
print('\n1. 检查自动采集管理器状态')
print('-' * 60)
try:
    # 尝试获取更详细的自动采集状态
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"运行中: {auto_data.get('running', False)}")
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
        print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
        print(f"待处理任务: {auto_data.get('pending_tasks_count', 0)}")
        print(f"上次检查: {auto_data.get('last_check_time') or '无'}")
        print(f"下次检查: {auto_data.get('next_check_time') or '无'}")
except Exception as e:
    print(f"获取失败: {e}")

# 2. 检查调度器内部状态
print('\n2. 检查统一调度器内部状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    
    # 获取统计信息
    stats = scheduler.get_statistics()
    print(f"调度器运行: {stats.get('is_running', False)}")
    print(f"总任务数: {stats.get('total_tasks', 0)}")
    print(f"待处理任务: {stats.get('pending_tasks', 0)}")
    print(f"运行中任务: {stats.get('running_tasks', 0)}")
    print(f"已完成任务: {stats.get('completed_tasks', 0)}")
    print(f"失败任务: {stats.get('failed_tasks', 0)}")
    
    # 检查工作节点
    from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
    registry = get_unified_worker_registry()
    
    data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
    print(f"\n数据采集器工作节点: {len(data_collectors)}")
    for worker_id, worker_info in data_collectors.items():
        print(f"  - {worker_id}: {worker_info.get('status', '未知')}")
        
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查任务队列
print('\n3. 检查任务队列状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    
    # 尝试获取队列信息
    if hasattr(scheduler, '_task_manager'):
        task_manager = scheduler._task_manager
        print(f"任务管理器类型: {type(task_manager).__name__}")
        
        # 检查是否有任务
        if hasattr(task_manager, '_tasks'):
            tasks = task_manager._tasks
            print(f"任务字典中的任务数: {len(tasks)}")
            
            for task_id, task in list(tasks.items())[:5]:
                print(f"  - {task_id}: {task.get('status', '未知')}")
except Exception as e:
    print(f"检查失败: {e}")

# 4. 检查数据源配置
print('\n4. 检查baostock数据源配置')
print('-' * 60)
try:
    from src.gateway.web.config_manager import load_data_sources
    
    sources = load_data_sources()
    print(f"加载的数据源数量: {len(sources)}")
    
    # 查找baostock
    for source in sources:
        source_id = source.get('id', '')
        if 'baostock' in source_id.lower():
            print(f"\n找到baostock数据源:")
            print(f"  ID: {source_id}")
            print(f"  名称: {source.get('name')}")
            print(f"  启用: {source.get('enabled', False)}")
            print(f"  状态: {source.get('status', '未知')}")
            print(f"  配置: {source.get('config', {})}")
            break
    else:
        print("未找到baostock数据源")
        print(f"\n所有数据源ID:")
        for s in sources[:10]:
            print(f"  - {s.get('id')}")
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 尝试手动提交测试任务
print('\n5. 尝试手动提交测试任务')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    import asyncio
    
    scheduler = get_unified_scheduler()
    
    async def submit_test_task():
        try:
            task_id = await scheduler.submit_task(
                task_type="DATA_COLLECTION",
                payload={
                    "source_id": "baostock_a股数据",
                    "test": True
                },
                priority=5
            )
            print(f"测试任务提交成功: {task_id}")
            return task_id
        except Exception as e:
            print(f"测试任务提交失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 运行异步任务
    task_id = asyncio.run(submit_test_task())
    
    if task_id:
        print(f"\n任务提交成功，检查任务状态...")
        # 等待一下再检查
        import time
        time.sleep(2)
        
        stats = scheduler.get_statistics()
        print(f"提交后 - 总任务数: {stats.get('total_tasks', 0)}")
        print(f"提交后 - 待处理任务: {stats.get('pending_tasks', 0)}")
        
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
