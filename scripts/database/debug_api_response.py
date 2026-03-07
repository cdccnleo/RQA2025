"""
调试API响应，检查为什么completed任务没有返回
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.feature_engineering_service import get_feature_tasks
from src.gateway.web.feature_task_persistence import list_feature_tasks


def debug_api():
    """调试API"""
    print("\n" + "="*80)
    print("🔍 调试API响应")
    print("="*80)
    
    # 直接调用 list_feature_tasks
    print("\n📋 直接调用 list_feature_tasks():")
    tasks = list_feature_tasks(limit=100)
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
        print(f"      {i}. {task_id} | {status}")
    
    # 调用 get_feature_tasks
    print("\n📋 调用 get_feature_tasks():")
    tasks2 = get_feature_tasks()
    print(f"   返回任务数: {len(tasks2)}")
    
    status_counts2 = {}
    for task in tasks2:
        status = task.get('status', 'unknown')
        status_counts2[status] = status_counts2.get(status, 0) + 1
    
    print(f"\n   状态分布:")
    for status, count in sorted(status_counts2.items()):
        print(f"      - {status}: {count}")


if __name__ == "__main__":
    debug_api()
