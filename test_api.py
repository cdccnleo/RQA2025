#!/usr/bin/env python3
"""测试特征工程API"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_get_tasks():
    """测试获取特征任务列表"""
    response = requests.get(f"{BASE_URL}/features/engineering/tasks")
    print("=== 获取特征任务列表 ===")
    print(f"状态码: {response.status_code}")
    data = response.json()
    print(f"任务数量: {len(data.get('tasks', []))}")
    print(f"统计: {data.get('stats', {})}")
    return data

def test_create_task():
    """测试创建特征任务"""
    payload = {
        "task_type": "技术指标",
        "config": {
            "symbol": "002837",
            "start_date": "2025-04-07",
            "end_date": "2026-02-12",
            "indicators": ["SMA", "EMA", "RSI", "MACD"]
        }
    }
    
    print("\n=== 创建特征任务 ===")
    response = requests.post(
        f"{BASE_URL}/features/engineering/tasks",
        json=payload
    )
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"创建成功: {data}")
        return data
    else:
        print(f"创建失败: {response.text}")
        return None

def test_get_features():
    """测试获取特征列表"""
    response = requests.get(f"{BASE_URL}/features/engineering/features")
    print("\n=== 获取特征列表 ===")
    print(f"状态码: {response.status_code}")
    data = response.json()
    print(f"特征数量: {len(data.get('features', []))}")
    print(f"质量分布: {data.get('quality_distribution', {})}")
    return data

if __name__ == "__main__":
    # 测试获取任务列表
    tasks_data = test_get_tasks()
    
    # 测试创建任务
    new_task = test_create_task()
    
    # 测试获取特征列表
    features_data = test_get_features()
    
    print("\n=== 测试完成 ===")
