#!/usr/bin/env python3
"""测试特征工程使用真实数据"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_create_task_with_real_data():
    """测试创建特征任务并使用真实数据"""
    
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
    
    print("=== 创建特征任务（使用真实数据） ===")
    response = requests.post(
        f"{BASE_URL}/features/engineering/tasks",
        json=payload
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"创建成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # 获取任务ID
        task_id = data.get("task", {}).get("task_id")
        if task_id:
            print(f"\n任务ID: {task_id}")
            
            # 等待任务完成
            print("\n等待任务执行...")
            time.sleep(5)
            
            # 查询任务状态
            status_response = requests.get(f"{BASE_URL}/features/engineering/tasks")
            if status_response.status_code == 200:
                tasks_data = status_response.json()
                for task in tasks_data.get("tasks", []):
                    if task.get("task_id") == task_id:
                        print(f"\n任务状态: {task.get('status')}")
                        print(f"进度: {task.get('progress')}%")
                        print(f"特征数量: {task.get('feature_count', 0)}")
                        
                        # 检查是否使用了真实数据
                        if task.get("feature_count", 0) > 0:
                            print("\n✅ 特征任务成功执行！")
                            if task.get("feature_count", 0) > 5:  # 原始数据有11列
                                print("✅ 可能使用了真实数据（特征数量较多）")
                        break
        
        return data
    else:
        print(f"创建失败: {response.text}")
        return None

def test_data_summary():
    """测试数据汇总信息"""
    print("\n=== 检查数据库数据汇总 ===")
    
    # 这里应该调用数据加载器的get_data_summary方法
    # 简化测试，直接查询数据库
    import subprocess
    result = subprocess.run(
        ['docker', 'exec', 'rqa2025-postgres', 'psql', '-U', 'rqa2025_admin', 
         '-d', 'rqa2025_prod', '-c', 
         'SELECT COUNT(*) as total, COUNT(DISTINCT symbol) as symbols FROM akshare_stock_data;'],
        capture_output=True, text=True
    )
    print(result.stdout)

if __name__ == "__main__":
    # 测试数据汇总
    test_data_summary()
    
    # 测试创建特征任务
    result = test_create_task_with_real_data()
    
    print("\n=== 测试完成 ===")
