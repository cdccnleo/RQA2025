#!/usr/bin/env python3
"""测试特征工程使用真实数据流"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_create_and_monitor_task():
    """创建任务并监控执行过程"""
    
    # 创建任务 - 使用数据库中存在的股票代码 002837
    payload = {
        "task_type": "技术指标",
        "config": {
            "symbol": "002837",
            "start_date": "2025-04-07",
            "end_date": "2026-02-12",
            "indicators": ["SMA", "EMA", "RSI", "MACD"]
        }
    }
    
    print("=== 步骤1: 创建特征任务 ===")
    response = requests.post(
        f"{BASE_URL}/features/engineering/tasks",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"❌ 创建任务失败: {response.text}")
        return None
    
    data = response.json()
    task_id = data.get("task", {}).get("task_id")
    print(f"✅ 任务创建成功: {task_id}")
    
    # 监控任务执行
    print("\n=== 步骤2: 监控任务执行 ===")
    max_attempts = 20
    for i in range(max_attempts):
        time.sleep(3)
        
        status_response = requests.get(f"{BASE_URL}/features/engineering/tasks")
        if status_response.status_code != 200:
            print(f"❌ 查询任务状态失败")
            continue
        
        tasks_data = status_response.json()
        for task in tasks_data.get("tasks", []):
            if task.get("task_id") == task_id:
                status = task.get('status')
                progress = task.get('progress', 0)
                feature_count = task.get('feature_count', 0)
                
                print(f"  检查 {i+1}/{max_attempts}: 状态={status}, 进度={progress}%, 特征数={feature_count}")
                
                if status == 'completed':
                    print(f"\n✅ 任务执行完成!")
                    print(f"✅ 特征数量: {feature_count}")
                    
                    # 验证是否使用了真实数据
                    if feature_count > 5:  # 原始数据有11列
                        print(f"✅ 使用了真实数据（特征数量: {feature_count} > 5）")
                        return task
                    else:
                        print(f"⚠️ 可能未使用真实数据（特征数量: {feature_count} <= 5）")
                        return task
                        
                elif status == 'failed':
                    print(f"\n❌ 任务执行失败!")
                    print(f"错误: {task.get('error_message', '未知错误')}")
                    return task
                
                break
    
    print(f"\n⚠️ 任务执行超时")
    return None

def get_scheduler_status():
    """获取调度器状态"""
    print("\n=== 调度器状态 ===")
    response = requests.get(f"{BASE_URL}/features/engineering/scheduler/status")
    if response.status_code == 200:
        data = response.json()
        print(f"调度器运行中: {data.get('is_running', False)}")
        print(f"工作节点数: {data.get('worker_count', 0)}")
        print(f"待处理任务: {data.get('pending_tasks', 0)}")
        print(f"运行中任务: {data.get('running_tasks', 0)}")
        return data
    else:
        print(f"❌ 获取调度器状态失败: {response.status_code}")
        return None

if __name__ == "__main__":
    # 检查调度器状态
    scheduler_status = get_scheduler_status()
    
    if scheduler_status and not scheduler_status.get('is_running'):
        print("\n启动调度器...")
        requests.post(f"{BASE_URL}/features/engineering/scheduler/start")
        time.sleep(2)
    
    # 创建并监控任务
    task = test_create_and_monitor_task()
    
    if task:
        print("\n=== 任务详情 ===")
        print(json.dumps(task, indent=2, ensure_ascii=False, default=str))
    
    print("\n=== 测试完成 ===")
