#!/usr/bin/env python3
"""测试调度器启动"""

import sys
import asyncio
sys.path.insert(0, '/app/src')

async def test_scheduler():
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        scheduler = get_unified_scheduler()
        
        print(f"调度器实例: {scheduler}")
        print(f"启动前 - 是否运行中: {scheduler._running}")
        
        print("\n🔧 启动调度器...")
        success = await scheduler.start()
        
        print(f"启动结果: {success}")
        print(f"启动后 - 是否运行中: {scheduler._running}")
        print(f"调度任务: {scheduler._scheduler_task}")
        
        # 等待几秒让调度循环启动
        await asyncio.sleep(2)
        
        print(f"\n2秒后 - 是否运行中: {scheduler._running}")
        
        # 检查是否能从数据库加载任务
        print("\n📊 尝试从数据库加载任务...")
        await scheduler._load_pending_tasks_from_db()
        
        # 停止调度器
        print("\n🛑 停止调度器...")
        await scheduler.stop()
        
        print(f"停止后 - 是否运行中: {scheduler._running}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scheduler())
