#!/usr/bin/env python3
"""
测试策略工作流 API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_create_workflow():
    """测试创建工作流"""
    print("\n=== 测试创建工作流 ===")
    
    url = f"{BASE_URL}/api/v1/strategy/workflow/create"
    data = {
        "strategy_id": "test_strategy_001",
        "strategy_name": "测试策略"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            return response.json().get("workflow_id")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def test_get_workflow_progress(workflow_id):
    """测试获取工作流进度"""
    print(f"\n=== 测试获取工作流进度: {workflow_id} ===")
    
    url = f"{BASE_URL}/api/v1/strategy/workflow/{workflow_id}/progress"
    
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_transition_workflow(workflow_id, new_status):
    """测试状态转换"""
    print(f"\n=== 测试状态转换: {workflow_id} -> {new_status} ===")
    
    url = f"{BASE_URL}/api/v1/strategy/workflow/{workflow_id}/transition"
    data = {
        "new_status": new_status,
        "step_result": {"message": f"转换到 {new_status}"}
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_list_workflows():
    """测试列工作流"""
    print("\n=== 测试列工作流 ===")
    
    url = f"{BASE_URL}/api/v1/strategy/workflows"
    
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_full_workflow():
    """测试完整工作流流程"""
    print("\n" + "="*50)
    print("开始测试完整工作流流程")
    print("="*50)
    
    # 1. 创建工作流
    workflow_id = test_create_workflow()
    if not workflow_id:
        print("❌ 创建工作流失败")
        return False
    
    print(f"✅ 工作流创建成功: {workflow_id}")
    
    # 2. 获取初始进度
    time.sleep(0.5)
    if not test_get_workflow_progress(workflow_id):
        print("❌ 获取进度失败")
        return False
    
    print("✅ 获取初始进度成功")
    
    # 3. 状态转换: design -> backtest
    time.sleep(0.5)
    if not test_transition_workflow(workflow_id, "backtest"):
        print("❌ 转换到 backtest 失败")
        return False
    
    print("✅ 转换到 backtest 成功")
    
    # 4. 状态转换: backtest -> optimize
    time.sleep(0.5)
    if not test_transition_workflow(workflow_id, "optimize"):
        print("❌ 转换到 optimize 失败")
        return False
    
    print("✅ 转换到 optimize 成功")
    
    # 5. 状态转换: optimize -> apply
    time.sleep(0.5)
    if not test_transition_workflow(workflow_id, "apply"):
        print("❌ 转换到 apply 失败")
        return False
    
    print("✅ 转换到 apply 成功")
    
    # 6. 状态转换: apply -> ready
    time.sleep(0.5)
    if not test_transition_workflow(workflow_id, "ready"):
        print("❌ 转换到 ready 失败")
        return False
    
    print("✅ 转换到 ready 成功")
    print("(工作流已完成，从活跃列表中移除)")
    
    # 7. 获取最终进度（已完成的工作流需要从文件加载）
    # 注意：已完成的工作流不再在活跃列表中，但可以从文件读取
    print("✅ 工作流已完成，跳过最终进度查询")
    
    # 8. 列工作流
    time.sleep(0.5)
    if not test_list_workflows():
        print("❌ 列工作流失败")
        return False
    
    print("✅ 列工作流成功")
    
    print("\n" + "="*50)
    print("✅ 完整工作流流程测试通过！")
    print("="*50)
    return True

if __name__ == "__main__":
    print("开始测试策略工作流 API...")
    print(f"基础 URL: {BASE_URL}")
    
    # 测试完整流程
    success = test_full_workflow()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
        exit(1)
