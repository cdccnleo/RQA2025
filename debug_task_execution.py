#!/usr/bin/env python3
"""调试任务执行流程"""
import asyncio
import time

print('=== 调试任务执行流程 ===')

# 1. 检查调度器状态
print('\n1. 检查调度器状态')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    
    # 检查调度器是否运行
    stats = scheduler.get_statistics()
    print(f"调度器统计: {stats}")
    
    # 检查任务管理器
    task_manager = scheduler._task_manager
    print(f"\n任务管理器任务数: {len(task_manager._tasks)}")
    
    # 检查工作管理器
    worker_manager = scheduler._worker_manager
    print(f"工作管理器运行中: {worker_manager._running}")
    print(f"工作节点数: {len(worker_manager._workers)}")
    print(f"任务处理器: {list(worker_manager._task_handlers.keys())}")
    
    # 检查队列
    import queue
    temp_queue = queue.Queue()
    queue_size = 0
    while not worker_manager._task_queue.empty() and queue_size < 10:
        try:
            task = worker_manager._task_queue.get_nowait()
            temp_queue.put(task)
            queue_size += 1
        except:
            break
    
    # 放回队列
    while not temp_queue.empty():
        worker_manager._task_queue.put(temp_queue.get())
    
    print(f"队列中的任务数: {queue_size}")
    
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 手动提交一个测试任务
print('\n2. 手动提交测试任务')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    
    async def submit_test():
        task_id = await scheduler.submit_task(
            task_type="DATA_COLLECTION",
            payload={
                "source_id": "baostock_a股数据",
                "test": True,
                "timestamp": time.time()
            },
            priority=5
        )
        return task_id
    
    # 运行异步任务
    try:
        loop = asyncio.get_event_loop()
        task_id = loop.run_until_complete(submit_test())
    except:
        task_id = asyncio.run(submit_test())
    
    print(f"任务提交成功: {task_id}")
    
    # 检查任务是否被添加
    task_manager = scheduler._task_manager
    print(f"任务字典中的任务数: {len(task_manager._tasks)}")
    
    if task_id in task_manager._tasks:
        task = task_manager._tasks[task_id]
        print(f"任务状态: {task.status}")
        print(f"任务类型: {task.type}")
        print(f"任务payload: {task.payload}")
    
    # 检查队列
    worker_manager = scheduler._worker_manager
    import queue
    temp_queue = queue.Queue()
    queue_size = 0
    while not worker_manager._task_queue.empty() and queue_size < 10:
        try:
            task = worker_manager._task_queue.get_nowait()
            temp_queue.put(task)
            queue_size += 1
            if task.get('id') == task_id:
                print(f"✅ 找到任务在队列中: {task}")
        except:
            break
    
    # 放回队列
    while not temp_queue.empty():
        worker_manager._task_queue.put(temp_queue.get())
    
    print(f"队列中的任务数: {queue_size}")
    
except Exception as e:
    print(f"提交失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 等待任务执行
print('\n3. 等待任务执行（10秒）...')
print('-' * 60)
time.sleep(10)

# 4. 检查任务执行结果
print('\n4. 检查任务执行结果')
print('-' * 60)
try:
    from src.core.orchestration.scheduler import get_unified_scheduler
    
    scheduler = get_unified_scheduler()
    task_manager = scheduler._task_manager
    
    print(f"任务字典中的任务数: {len(task_manager._tasks)}")
    
    for task_id, task in list(task_manager._tasks.items())[:5]:
        print(f"\n任务: {task_id}")
        print(f"  状态: {task.status}")
        print(f"  结果: {task.result}")
        print(f"  错误: {task.error}")
        
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('调试完成')
print('=' * 60)
