#!/usr/bin/env python3
"""
手工触发PostgreSQL中submitted状态的特征提取任务
"""

import sys
import asyncio
sys.path.insert(0, '/app/src')

async def trigger_submitted_tasks():
    """触发所有submitted状态的任务"""
    try:
        # 1. 从PostgreSQL获取submitted状态的任务
        from src.gateway.web.feature_task_persistence import list_feature_tasks
        
        tasks = list_feature_tasks(status='submitted', limit=100)
        
        if not tasks:
            print("✅ 没有submitted状态的任务需要执行")
            return True
        
        print(f"📊 找到 {len(tasks)} 个submitted状态的任务")
        
        # 2. 获取调度器
        from src.core.orchestration.scheduler import get_unified_scheduler
        scheduler = get_unified_scheduler()
        
        if not scheduler:
            print("❌ 调度器未初始化")
            return False
        
        # 3. 提交任务到调度器
        triggered_count = 0
        for task in tasks:
            task_id = task.get('task_id')
            symbol = task.get('symbol', task.get('config', {}).get('symbol', ''))
            
            print(f"🚀 触发任务: {task_id}, 股票: {symbol}")
            
            try:
                # 构建任务payload
                payload = {
                    'task_id': task_id,
                    'symbol': symbol,
                    'config': task.get('config', {}),
                    'indicators': task.get('config', {}).get('indicators', [])
                }
                
                # 提交到调度器
                new_task_id = await scheduler.submit_task(
                    task_type='feature_extraction',
                    payload=payload,
                    priority=5
                )
                
                print(f"✅ 任务已提交到调度器: {new_task_id}")
                triggered_count += 1
                
            except Exception as e:
                print(f"❌ 提交任务失败 {task_id}: {e}")
        
        print(f"\n📊 成功触发 {triggered_count}/{len(tasks)} 个任务")
        return True
        
    except Exception as e:
        print(f"❌ 触发任务失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 手工触发submitted状态的特征提取任务")
    print("=" * 60)
    
    success = asyncio.run(trigger_submitted_tasks())
    
    print("=" * 60)
    sys.exit(0 if success else 1)
