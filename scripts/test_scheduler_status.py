#!/usr/bin/env python3
"""
测试调度器状态
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    try:
        from src.core.orchestration.historical_data_scheduler import get_historical_data_scheduler

        print("获取调度器实例...")
        scheduler = get_historical_data_scheduler()

        print(f"调度器类型: {type(scheduler)}")
        print(f"调度器状态: {scheduler.status}")
        print(f"调度器worker_nodes数量: {len(scheduler.worker_nodes)}")

        # 检查是否有注册的工作进程
        if scheduler.worker_nodes:
            print("已注册的工作进程:")
            for worker_id, worker in scheduler.worker_nodes.items():
                print(f"  - {worker_id}: {worker.host}:{worker.port}, 活跃: {worker.is_active}")
        else:
            print("没有注册的工作进程")

        # 检查任务队列
        print(f"待处理任务数量: {scheduler.task_queue.qsize()}")

        print("✅ 调度器状态检查完成")

    except Exception as e:
        print(f"❌ 调度器状态检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()