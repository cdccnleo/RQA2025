"""
调试持久化层
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.feature_task_persistence import _list_from_postgresql


def debug_persistence():
    """调试持久化层"""
    print("\n" + "="*80)
    print("🔍 调试持久化层")
    print("="*80)
    
    # 直接调用 _list_from_postgresql
    print("\n📋 调用 _list_from_postgresql(status=None, limit=100):")
    tasks = _list_from_postgresql(status=None, limit=100)
    
    print(f"   返回任务数: {len(tasks)}")
    
    # 按状态统计
    status_counts = {}
    for task in tasks:
        status = task.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n   状态分布:")
    for status, count in sorted(status_counts.items()):
        print(f"      - {status}: {count}")
    
    # 显示所有任务
    print(f"\n   所有任务列表:")
    for i, task in enumerate(tasks, 1):
        task_id = task.get('task_id', 'N/A')
        status = task.get('status', 'N/A')
        created_at = task.get('created_at', 'N/A')
        print(f"      {i}. {task_id} | {status} | {created_at}")


if __name__ == "__main__":
    debug_persistence()
