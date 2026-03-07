#!/usr/bin/env python3
"""
检查交易记录的手续费数据
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771323986'
result = load_backtest_result(backtest_id)

if not result:
    print('回测记录未找到')
    sys.exit(1)

print('=' * 80)
print('🔍 交易记录手续费检查')
print('=' * 80)
print()

trades = result.get('trades', [])
print(f"交易记录数: {len(trades)}")
print()

for i, trade in enumerate(trades):
    print(f"交易{i+1}: {trade.get('timestamp', '')[:10]} {trade.get('symbol', '')} {trade.get('type', '')}")
    print(f"  价格: {trade.get('price')}")
    print(f"  数量: {trade.get('quantity')}")
    print(f"  手续费(cost): {trade.get('cost')}")
    print(f"  手续费(fee): {trade.get('fee')}")
    print(f"  手续费类型: {type(trade.get('cost'))}")
    print()
