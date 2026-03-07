#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试优化结果持久化功能"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gateway.web.strategy_persistence import (
    save_optimization_result, 
    list_optimization_results
)

def test_optimization_persistence():
    """测试优化结果持久化"""
    print("测试优化结果持久化...")
    
    # 创建测试数据
    test_result = {
        'task_id': 'test_opt_001',
        'strategy_id': 'test_strategy',
        'method': 'grid_search',
        'target': 'sharpe_ratio',
        'results': [{
            'parameters': {'param1': 10},
            'score': 1.5,
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': 0.05
        }],
        'completed_at': 1704643200
    }
    
    # 测试保存
    if save_optimization_result('test_opt_001', test_result):
        print("✅ 优化结果保存成功")
    else:
        print("❌ 优化结果保存失败")
        return False
    
    # 测试列表
    results = list_optimization_results()
    if any(r.get('task_id') == 'test_opt_001' for r in results):
        print(f"✅ 优化结果列表成功，共有 {len(results)} 个结果")
    else:
        print("❌ 优化结果列表失败")
        return False
    
    print("✅ 优化结果持久化测试通过")
    return True

if __name__ == '__main__':
    success = test_optimization_persistence()
    sys.exit(0 if success else 1)

