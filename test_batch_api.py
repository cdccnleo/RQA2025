#!/usr/bin/env python3
"""测试批量特征任务API"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_get_stock_pools():
    """测试获取股票池列表"""
    print("=== 测试1: 获取股票池列表 ===")
    response = requests.get(f"{BASE_URL}/features/engineering/pools")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ 成功获取股票池列表")
        print(f"   全市场股票数: {data.get('total_stocks', 0)}")
        print(f"   股票池数量: {len(data.get('pools', []))}")
        
        for pool in data.get('pools', []):
            print(f"   - {pool.get('name')} ({pool.get('pool_id')}): {pool.get('symbol_count')} 只股票")
        
        return data
    else:
        print(f"❌ 获取股票池列表失败: {response.status_code}")
        print(f"   错误: {response.text}")
        return None

def test_create_batch_tasks():
    """测试批量创建特征任务"""
    print("\n=== 测试2: 批量创建特征任务 ===")
    
    payload = {
        "stock_pool": "all",  # 使用全市场股票池
        "date_range": {
            "start": "2025-04-07",
            "end": "2026-02-12"
        },
        "task_type": "技术指标",
        "indicators": ["SMA", "EMA", "RSI", "MACD"],
        "batch_size": 50,  # 每批50只股票
        "priority": "medium"
    }
    
    print(f"请求参数: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(
        f"{BASE_URL}/features/engineering/tasks/batch",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ 批量任务创建成功")
        print(f"   股票池: {data.get('pool_name')}")
        print(f"   总股票数: {data.get('total_symbols')}")
        print(f"   批次数量: {data.get('batch_count')}")
        print(f"   成功创建任务数: {len(data.get('created_tasks', []))}")
        print(f"   失败任务数: {len(data.get('failed_tasks', []))}")
        
        # 显示前3个任务
        for task in data.get('created_tasks', [])[:3]:
            print(f"   - 批次 {task.get('batch_index') + 1}: {task.get('task_id')}, "
                  f"股票数: {task.get('symbol_count')}")
        
        return data
    else:
        print(f"❌ 批量创建任务失败: {response.status_code}")
        print(f"   错误: {response.text}")
        return None

def test_get_tasks():
    """测试获取任务列表"""
    print("\n=== 测试3: 获取特征任务列表 ===")
    response = requests.get(f"{BASE_URL}/features/engineering/tasks")
    
    if response.status_code == 200:
        data = response.json()
        tasks = data.get('tasks', [])
        stats = data.get('stats', {})
        
        print(f"✅ 成功获取任务列表")
        print(f"   任务总数: {len(tasks)}")
        print(f"   活跃任务: {stats.get('active', 0)}")
        print(f"   已完成: {stats.get('completed', 0)}")
        print(f"   失败: {stats.get('failed', 0)}")
        
        # 显示最新的3个任务
        for task in tasks[:3]:
            print(f"   - {task.get('task_id')}: {task.get('status')}, "
                  f"进度: {task.get('progress')}%, "
                  f"特征数: {task.get('feature_count', 0)}")
        
        return data
    else:
        print(f"❌ 获取任务列表失败: {response.status_code}")
        return None

if __name__ == "__main__":
    print("开始测试批量特征任务API...\n")
    
    # 测试1: 获取股票池
    pools_data = test_get_stock_pools()
    
    # 测试2: 批量创建任务
    if pools_data and pools_data.get('total_stocks', 0) > 0:
        batch_data = test_create_batch_tasks()
    else:
        print("\n⚠️ 跳过批量任务测试（没有可用股票）")
    
    # 测试3: 获取任务列表
    tasks_data = test_get_tasks()
    
    print("\n=== 测试完成 ===")
