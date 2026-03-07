#!/usr/bin/env python3
"""
测试策略优化结果应用功能
"""

import json
import os

# 测试数据目录
DATA_DIR = "/app/data"
STRATEGY_CONCEPTIONS_DIR = os.path.join(DATA_DIR, "strategy_conceptions")
OPTIMIZATION_RESULTS_DIR = os.path.join(DATA_DIR, "optimization_results")

def test_strategy_apply():
    """测试策略应用功能"""
    
    # 1. 检查优化结果
    opt_file = os.path.join(OPTIMIZATION_RESULTS_DIR, "opt_1771396873.json")
    if not os.path.exists(opt_file):
        print("❌ 优化结果文件不存在")
        return False
    
    with open(opt_file, 'r') as f:
        opt_result = json.load(f)
    
    strategy_id = opt_result.get("strategy_id")
    print(f"优化结果中的策略ID: {strategy_id}")
    
    # 2. 检查策略文件是否存在
    strategy_file = os.path.join(STRATEGY_CONCEPTIONS_DIR, f"{strategy_id}.json")
    print(f"策略文件路径: {strategy_file}")
    print(f"策略文件是否存在: {os.path.exists(strategy_file)}")
    
    if not os.path.exists(strategy_file):
        print(f"❌ 策略文件不存在: {strategy_file}")
        
        # 3. 列出可用的策略文件
        print("\n可用的策略文件:")
        for f in os.listdir(STRATEGY_CONCEPTIONS_DIR):
            if f.endswith('.json'):
                print(f"  - {f}")
        
        return False
    
    print("✅ 策略文件存在")
    return True

if __name__ == "__main__":
    test_strategy_apply()
