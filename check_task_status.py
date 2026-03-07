#!/usr/bin/env python3
"""检查任务状态"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def check_task_status(task_id):
    """检查任务状态"""
    response = requests.get(f"{BASE_URL}/features/engineering/tasks")
    if response.status_code == 200:
        tasks_data = response.json()
        for task in tasks_data.get("tasks", []):
            if task.get("task_id") == task_id:
                return task
    return None

if __name__ == "__main__":
    task_id = "task_1770961078"
    
    print(f"检查任务 {task_id} 的状态...")
    
    for i in range(10):
        task = check_task_status(task_id)
        if task:
            print(f"\n第 {i+1} 次检查:")
            print(f"  状态: {task.get('status')}")
            print(f"  进度: {task.get('progress')}%")
            print(f"  特征数量: {task.get('feature_count', 0)}")
            
            if task.get('status') == 'completed':
                print("\n✅ 任务已完成！")
                print(f"✅ 特征数量: {task.get('feature_count', 0)}")
                break
            elif task.get('status') == 'failed':
                print("\n❌ 任务失败！")
                print(f"错误: {task.get('error_message', '未知错误')}")
                break
        
        time.sleep(3)
    
    print("\n检查完成")
