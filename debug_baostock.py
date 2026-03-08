#!/usr/bin/env python3
"""深度检查baostock数据源采集问题"""
import requests
import time

print('=== 深度检查baostock数据源采集问题 ===')

# 1. 检查baostock数据源详情
print('\n1. 检查baostock数据源详情')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    baostock_source = None
    for source in sources.get('data', []):
        if 'baostock' in source.get('id', '').lower():
            baostock_source = source
            break
    
    if baostock_source:
        print(f"数据源ID: {baostock_source.get('id')}")
        print(f"数据源名称: {baostock_source.get('name')}")
        print(f"是否启用: {baostock_source.get('is_active', False)}")
        print(f"状态: {baostock_source.get('status', '未知')}")
        print(f"最后采集时间: {baostock_source.get('last_collection_time') or '从未采集'}")
        print(f"配置: {baostock_source.get('config', {})}")
    else:
        print("❌ 未找到baostock数据源")
        
except Exception as e:
    print(f"获取失败: {e}")

# 2. 检查自动采集任务提交详情
print('\n2. 检查自动采集任务提交详情')
print('-' * 60)
try:
    from src.gateway.web.data_collection_scheduler_manager import get_scheduler_manager
    
    manager = get_scheduler_manager()
    stats = manager.get_stats()
    
    print(f"调度管理器运行中: {stats.get('running', False)}")
    print(f"总检查次数: {stats.get('total_checks', 0)}")
    print(f"已提交任务: {stats.get('tasks_submitted', 0)}")
    print(f"已检查数据源: {stats.get('sources_checked', 0)}")
    print(f"活跃数据源: {stats.get('active_sources', 0)}")
    
    # 检查已提交的任务
    submitted_tasks = stats.get('submitted_tasks', [])
    print(f"\n已提交任务列表: {len(submitted_tasks)}个")
    for task in submitted_tasks[-5:]:  # 最近5个
        print(f"  - {task}")
        
except Exception as e:
    print(f"获取失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查调度器任务状态
print('\n3. 检查调度器任务状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    
    # 获取任务管理器
    task_manager = scheduler._task_manager
    
    # 检查所有任务
    if hasattr(task_manager, '_tasks'):
        tasks = task_manager._tasks
        print(f"任务字典中的任务数: {len(tasks)}")
        
        for task_id, task in tasks.items():
            print(f"\n任务: {task_id}")
            print(f"  类型: {task.type}")
            print(f"  状态: {task.status}")
            print(f"  优先级: {task.priority}")
            print(f"  创建时间: {task.created_at}")
            print(f"  payload: {task.payload}")
            
            # 检查是否是baostock任务
            payload = task.payload or {}
            if 'baostock' in str(payload).lower():
                print(f"  ✅ 这是baostock相关任务")
                
except Exception as e:
    print(f"获取失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 检查工作管理器队列
print('\n4. 检查工作管理器队列')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    worker_manager = scheduler._worker_manager
    
    # 检查队列大小
    if hasattr(worker_manager, '_task_queue'):
        queue = worker_manager._task_queue
        print(f"队列类型: {type(queue).__name__}")
        
        # 尝试查看队列内容
        import queue as queue_module
        temp_queue = queue_module.Queue()
        
        task_count = 0
        while not queue.empty() and task_count < 10:
            try:
                task = queue.get_nowait()
                temp_queue.put(task)
                task_count += 1
                print(f"\n队列任务 {task_count}:")
                print(f"  ID: {task.get('id')}")
                print(f"  类型: {task.get('type')}")
                print(f"  payload: {task.get('payload')}")
            except:
                break
        
        # 将任务放回队列
        while not temp_queue.empty():
            queue.put(temp_queue.get())
        
        print(f"\n队列中的任务数: {task_count}")
    
    # 检查任务处理器
    if hasattr(worker_manager, '_task_handlers'):
        handlers = worker_manager._task_handlers
        print(f"\n已注册的任务处理器: {list(handlers.keys())}")
        
except Exception as e:
    print(f"获取失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 检查工作节点状态
print('\n5. 检查工作节点状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    worker_manager = scheduler._worker_manager
    
    # 检查工作节点
    if hasattr(worker_manager, '_workers'):
        workers = worker_manager._workers
        print(f"工作节点数量: {len(workers)}")
        
        for worker_id, worker in workers.items():
            print(f"\n工作节点: {worker_id}")
            print(f"  状态: {worker.status}")
            print(f"  当前任务: {worker.current_task}")
            print(f"  完成任务数: {worker.task_count}")
            print(f"  最后心跳: {worker.last_heartbeat}")
            
except Exception as e:
    print(f"获取失败: {e}")
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('检查完成')
print('=' * 60)
