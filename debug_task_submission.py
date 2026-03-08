#!/usr/bin/env python3
"""调试任务提交流程"""
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_task_submission():
    """调试任务提交"""
    print('=== 调试任务提交流程 ===')
    
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 1. 检查调度器状态
        print('\n1. 调度器初始状态')
        print('-' * 50)
        stats = scheduler.get_statistics()
        print(f"调度器运行: {stats.get('is_running', False)}")
        print(f"总任务数: {stats.get('total_tasks', 0)}")
        print(f"待处理任务: {stats.get('pending_tasks', 0)}")
        
        # 2. 检查任务管理器
        print('\n2. 任务管理器状态')
        print('-' * 50)
        task_manager = scheduler._task_manager
        print(f"任务管理器类型: {type(task_manager).__name__}")
        
        # 直接访问任务字典
        if hasattr(task_manager, '_tasks'):
            tasks = task_manager._tasks
            print(f"任务字典中的任务数: {len(tasks)}")
        
        # 3. 检查工作管理器
        print('\n3. 工作管理器状态')
        print('-' * 50)
        worker_manager = scheduler._worker_manager
        print(f"工作管理器类型: {type(worker_manager).__name__}")
        
        # 4. 检查工作节点
        print('\n4. 工作节点状态')
        print('-' * 50)
        data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
        print(f"数据采集器数量: {len(data_collectors)}")
        for worker in data_collectors:
            print(f"  - {worker.worker_id}: {worker.status.value}")
        
        # 5. 提交测试任务
        print('\n5. 提交测试任务')
        print('-' * 50)
        
        task_id = await scheduler.submit_task(
            task_type="DATA_COLLECTION",
            payload={
                "source_id": "baostock_a股数据",
                "test": True,
                "debug": True
            },
            priority=5
        )
        
        print(f"任务提交成功，ID: {task_id}")
        
        # 6. 检查任务是否被添加
        print('\n6. 检查任务是否被添加')
        print('-' * 50)
        
        # 检查任务管理器
        if hasattr(task_manager, '_tasks'):
            tasks = task_manager._tasks
            print(f"任务字典中的任务数: {len(tasks)}")
            
            if task_id in tasks:
                task = tasks[task_id]
                print(f"找到任务: {task_id}")
                print(f"  状态: {task.status}")
                print(f"  类型: {task.type}")
                print(f"  优先级: {task.priority}")
            else:
                print(f"❌ 任务 {task_id} 不在任务字典中！")
        
        # 检查调度器统计
        stats = scheduler.get_statistics()
        print(f"\n调度器统计:")
        print(f"  总任务数: {stats.get('total_tasks', 0)}")
        print(f"  待处理任务: {stats.get('pending_tasks', 0)}")
        print(f"  运行中任务: {stats.get('running_tasks', 0)}")
        
        # 7. 检查工作管理器的队列
        print('\n7. 检查工作管理器队列')
        print('-' * 50)
        
        if hasattr(worker_manager, '_task_queue'):
            queue = worker_manager._task_queue
            print(f"任务队列类型: {type(queue).__name__}")
            
            # 尝试获取队列大小
            if hasattr(queue, 'qsize'):
                try:
                    size = queue.qsize()
                    print(f"队列大小: {size}")
                except Exception as e:
                    print(f"获取队列大小失败: {e}")
        
        # 8. 检查工作节点是否收到任务
        print('\n8. 检查工作节点任务分配')
        print('-' * 50)
        
        data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
        for worker in data_collectors:
            print(f"工作节点 {worker.worker_id}:")
            print(f"  状态: {worker.status.value}")
            print(f"  当前任务: {worker.current_task}")
            print(f"  完成任务数: {worker.tasks_completed}")
        
        return True
        
    except Exception as e:
        logger.error(f"调试失败: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_task_submission())
    exit(0 if result else 1)
