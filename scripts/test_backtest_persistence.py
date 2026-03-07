#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试回测结果持久化功能"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gateway.web.backtest_persistence import (
    save_backtest_result, 
    load_backtest_result, 
    list_backtest_results
)

def test_backtest_persistence():
    """测试回测结果持久化"""
    print("测试回测结果持久化...")
    
    # 创建测试数据
    test_result = {
        'backtest_id': 'test_backtest_001',
        'strategy_id': 'test_strategy',
        'status': 'completed',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': 100000,
        'final_capital': 110000,
        'total_return': 0.1,
        'annualized_return': 0.1,
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.05,
        'win_rate': 0.6,
        'total_trades': 100,
        'equity_curve': [100000, 105000, 110000],
        'trades': [],
        'metrics': {}
    }
    
    # 测试保存
    if save_backtest_result(test_result):
        print("✅ 回测结果保存成功")
    else:
        print("❌ 回测结果保存失败")
        return False
    
    # 测试加载
    loaded = load_backtest_result('test_backtest_001')
    if loaded and loaded.get('backtest_id') == 'test_backtest_001':
        print("✅ 回测结果加载成功")
    else:
        print("❌ 回测结果加载失败")
        return False
    
    # 测试列表
    results = list_backtest_results(limit=10)
    if any(r.get('backtest_id') == 'test_backtest_001' for r in results):
        print(f"✅ 回测结果列表成功，共有 {len(results)} 个结果")
    else:
        print("❌ 回测结果列表失败")
        return False
    
    print("✅ 回测结果持久化测试通过")
    return True

if __name__ == '__main__':
    success = test_backtest_persistence()
    sys.exit(0 if success else 1)

