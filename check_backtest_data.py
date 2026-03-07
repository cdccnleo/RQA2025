#!/usr/bin/env python3
"""
检查回测数据结构和字段
"""
import sys
sys.path.insert(0, '/app')

from src.gateway.web.backtest_persistence import list_backtest_results, load_backtest_result
import json

def check_backtest_data():
    """检查回测数据结构"""
    print("=" * 60)
    print("📊 回测数据结构检查报告")
    print("=" * 60)
    print()
    
    # 获取回测记录列表
    results = list_backtest_results(limit=3)
    print(f"找到 {len(results)} 条回测记录")
    print()
    
    if not results:
        print("❌ 没有回测记录")
        return
    
    # 检查第一条记录的字段
    first_result = results[0]
    backtest_id = first_result.get('backtest_id')
    
    print(f"检查回测记录: {backtest_id}")
    print()
    
    # 加载完整数据
    full_result = load_backtest_result(backtest_id)
    
    if not full_result:
        print("❌ 无法加载完整回测数据")
        return
    
    # 检查关键字段
    key_fields = [
        'backtest_id',
        'strategy_id',
        'start_date',
        'end_date',
        'initial_capital',
        'final_capital',
        'total_return',
        'annualized_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'total_trades',
        'equity_curve',
        'trades',
        'metrics'
    ]
    
    print("【字段检查】")
    for field in key_fields:
        value = full_result.get(field)
        if value is not None:
            if field == 'equity_curve':
                if isinstance(value, list):
                    print(f"  ✅ {field}: 数组，长度 {len(value)}")
                    if len(value) > 0:
                        print(f"     前3个值: {value[:3]}")
                        print(f"     后3个值: {value[-3:]}")
                else:
                    print(f"  ❌ {field}: 类型错误 {type(value)}")
            elif field == 'trades':
                if isinstance(value, list):
                    print(f"  ✅ {field}: 数组，长度 {len(value)}")
                else:
                    print(f"  ❌ {field}: 类型错误 {type(value)}")
            elif field == 'metrics':
                if isinstance(value, dict):
                    print(f"  ✅ {field}: 字典，键 {list(value.keys())}")
                else:
                    print(f"  ⚠️ {field}: {value}")
            else:
                print(f"  ✅ {field}: {value}")
        else:
            print(f"  ❌ {field}: 缺失")
    
    print()
    print("【equity_curve 详细检查】")
    equity_curve = full_result.get('equity_curve', [])
    if equity_curve:
        print(f"  数据类型: {type(equity_curve)}")
        print(f"  数组长度: {len(equity_curve)}")
        if len(equity_curve) > 0:
            print(f"  第一个元素类型: {type(equity_curve[0])}")
            print(f"  第一个元素值: {equity_curve[0]}")
            print(f"  最后一个元素值: {equity_curve[-1]}")
            
            # 检查是否所有元素都是数字
            all_numbers = all(isinstance(x, (int, float)) for x in equity_curve)
            print(f"  所有元素都是数字: {all_numbers}")
    else:
        print("  ❌ equity_curve 为空")
    
    print()
    print("【trades 详细检查】")
    trades = full_result.get('trades', [])
    if trades:
        print(f"  交易数量: {len(trades)}")
        if len(trades) > 0:
            first_trade = trades[0]
            print(f"  第一条交易记录字段: {list(first_trade.keys())}")
    else:
        print("  ⚠️ trades 为空")

if __name__ == "__main__":
    check_backtest_data()
